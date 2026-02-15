"""ComfyUI API client — RunPod Serverless version."""

import asyncio
import base64
import io
import json
import logging
import uuid

import aiohttp
from PIL import Image, ImageOps

import config

logger = logging.getLogger(__name__)


def _image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def build_flux_fill_workflow(
    prompt: str,
    negative: str = "blurry, ugly, deformed, watermark, text, low quality, cartoon",
    cfg: float = 3.5,
    steps: int = 25,
    seed: int | None = None,
) -> dict:
    """Build the Flux Fill inpaint workflow for ComfyUI API."""
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))

    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "inpaint_input.png",
            },
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-fill-dev.safetensors",
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ae.safetensors",
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["3", 0],
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": ["3", 0],
            },
        },
        "7": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["4", 0],
                "pixels": ["1", 0],
                "mask": ["1", 1],
                "noise_mask": True,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["7", 2],
                "seed": seed,
                "control_after_generate": "randomize",
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["4", 0],
            },
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": "TGBot_Result",
            },
        },
    }

    return workflow


# ============================================================
# RunPod Serverless API
# ============================================================

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


async def submit_job(workflow: dict, images: list[dict]) -> str | None:
    """Submit a job to RunPod Serverless endpoint.
    
    Args:
        workflow: ComfyUI workflow dict
        images: List of {"name": "filename.png", "image": "base64data"}
    
    Returns:
        Job ID or None on failure.
    """
    url = f"{RUNPOD_API_BASE}/{config.RUNPOD_ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "workflow": workflow,
            "images": images,
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error("RunPod submit failed (%d): %s", resp.status, text)
                return None
            result = await resp.json()
            job_id = result.get("id")
            logger.info("RunPod job submitted: %s (status: %s)", job_id, result.get("status"))
            return job_id


async def poll_job(job_id: str, timeout: int = 300) -> dict | None:
    """Poll RunPod Serverless for job completion.
    
    Returns the output dict or None on timeout/failure.
    """
    url = f"{RUNPOD_API_BASE}/{config.RUNPOD_ENDPOINT_ID}/status/{job_id}"
    headers = {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
    }
    
    async with aiohttp.ClientSession() as session:
        for i in range(timeout // 3):
            await asyncio.sleep(3)
            try:
                async with session.get(url, headers=headers) as resp:
                    data = await resp.json()
                    status = data.get("status")
                    
                    if status == "COMPLETED":
                        logger.info("Job %s completed!", job_id)
                        return data.get("output")
                    elif status == "FAILED":
                        logger.error("Job %s failed: %s", job_id, data.get("error"))
                        return None
                    elif status in ("IN_QUEUE", "IN_PROGRESS"):
                        if i % 5 == 0:
                            logger.info("Job %s status: %s", job_id, status)
                    else:
                        logger.warning("Job %s unknown status: %s", job_id, status)
            except Exception as e:
                logger.warning("Poll error: %s", e)

    logger.error("Job %s timed out after %d seconds", job_id, timeout)
    return None


async def run_inpaint(
    image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    negative: str = "blurry, ugly, deformed, watermark, text, low quality, cartoon",
    cfg: float = 3.5,
    steps: int = 25,
) -> bytes | None:
    """Full inpaint pipeline via RunPod Serverless.
    
    1. Combine image + mask into RGBA PNG
    2. Submit workflow + image to RunPod Serverless
    3. Poll for result
    4. Return result image bytes
    """

    # Combine image and mask into a single PNG with alpha mask
    # ComfyUI LoadImage extracts mask from alpha channel
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    # Resize mask to match image if needed
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.LANCZOS)

    # Create RGBA image with inverted mask as alpha
    # Frontend mask: white (255) = inpaint, black (0) = keep
    # ComfyUI LoadImage alpha: 0 (transparent) → mask 1.0 (inpaint), 255 (opaque) → mask 0.0 (keep)
    # So we INVERT: white (255) → alpha 0 (transparent) = inpaint ✓
    mask_inverted = ImageOps.invert(mask)
    rgba = img.copy()
    rgba.putalpha(mask_inverted)

    # Save as PNG bytes
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    rgba_bytes = buf.getvalue()

    # Encode as base64 for RunPod Serverless API
    image_b64 = _image_to_base64(rgba_bytes)

    # Build workflow
    logger.info("Building workflow (prompt=%s, cfg=%s, steps=%s)...", prompt[:50], cfg, steps)
    workflow = build_flux_fill_workflow(prompt=prompt, negative=negative, cfg=cfg, steps=steps)

    # Submit job with image
    images = [
        {
            "name": "inpaint_input.png",
            "image": image_b64,
        }
    ]

    logger.info("Submitting to RunPod Serverless (endpoint: %s)...", config.RUNPOD_ENDPOINT_ID)
    job_id = await submit_job(workflow, images)

    if not job_id:
        logger.error("Failed to submit job")
        return None

    # Poll for result
    logger.info("Waiting for result (job: %s)...", job_id)
    output = await poll_job(job_id)

    if not output:
        return None

    # Extract result image from output
    # RunPod worker-comfyui returns: {"images": [{"image": "base64...", "type": "base64"}]}
    try:
        output_images = output.get("images", [])
        if output_images:
            img_data = output_images[0]
            if img_data.get("type") == "base64":
                return base64.b64decode(img_data["image"])
            elif "image" in img_data:
                return base64.b64decode(img_data["image"])
        
        # Alternative output format: {"message": "base64..."}
        if "message" in output:
            return base64.b64decode(output["message"])
            
        logger.error("Unexpected output format: %s", list(output.keys()))
        return None
    except Exception as e:
        logger.error("Failed to parse result: %s", e)
        return None
