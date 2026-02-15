"""ComfyUI API client — submit workflows and get results."""

import asyncio
import base64
import io
import json
import logging
import uuid

import aiohttp
from PIL import Image

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
    """Build the Flux Fill inpaint workflow for ComfyUI API.
    
    Note: image and mask are uploaded separately via /upload/image endpoint.
    """
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


async def upload_image(
    image_bytes: bytes, filename: str = "inpaint_input.png", image_type: str = "input", overwrite: bool = True
) -> dict:
    """Upload an image to ComfyUI."""
    url = f"{config.COMFYUI_BASE_URL}/upload/image"

    form = aiohttp.FormData()
    form.add_field(
        "image",
        image_bytes,
        filename=filename,
        content_type="image/png",
    )
    form.add_field("type", image_type)
    form.add_field("overwrite", str(overwrite).lower())

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form) as resp:
            result = await resp.json()
            logger.info("Upload image result: %s", result)
            return result


async def queue_prompt(workflow: dict) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    url = f"{config.COMFYUI_BASE_URL}/prompt"

    payload = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4()),
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            prompt_id = result.get("prompt_id", "")
            logger.info("Queued prompt: %s", prompt_id)
            return prompt_id


async def poll_result(prompt_id: str, timeout: int = 300) -> dict | None:
    """Poll ComfyUI history until the prompt is done."""
    url = f"{config.COMFYUI_BASE_URL}/history/{prompt_id}"

    async with aiohttp.ClientSession() as session:
        for _ in range(timeout // 2):
            await asyncio.sleep(2)
            try:
                async with session.get(url) as resp:
                    history = await resp.json()
                    if prompt_id in history:
                        return history[prompt_id]
            except Exception as e:
                logger.warning("Poll error: %s", e)

    logger.error("Prompt %s timed out after %d seconds", prompt_id, timeout)
    return None


async def download_image(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes | None:
    """Download a result image from ComfyUI."""
    url = f"{config.COMFYUI_BASE_URL}/view"
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.read()
            logger.error("Failed to download image: %s", resp.status)
            return None


async def run_inpaint(
    image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    negative: str = "blurry, ugly, deformed, watermark, text, low quality, cartoon",
    cfg: float = 3.5,
    steps: int = 25,
) -> bytes | None:
    """Full inpaint pipeline: upload image+mask, run workflow, download result."""

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
    from PIL import ImageOps
    mask_inverted = ImageOps.invert(mask)
    rgba = img.copy()
    rgba.putalpha(mask_inverted)

    # Save as PNG
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    rgba_bytes = buf.getvalue()

    # Upload image with mask
    logger.info("Uploading image with mask...")
    await upload_image(rgba_bytes, filename="inpaint_input.png")

    # Build and submit workflow
    logger.info("Submitting workflow (prompt=%s, cfg=%s, steps=%s)...", prompt[:50], cfg, steps)
    workflow = build_flux_fill_workflow(prompt=prompt, negative=negative, cfg=cfg, steps=steps)
    prompt_id = await queue_prompt(workflow)

    if not prompt_id:
        logger.error("Failed to queue prompt")
        return None

    # Poll for result
    logger.info("Waiting for result...")
    result = await poll_result(prompt_id)

    if not result:
        return None

    # Extract output image filename
    try:
        outputs = result["outputs"]
        for node_id, output in outputs.items():
            if "images" in output:
                img_info = output["images"][0]
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                logger.info("Downloading result: %s", filename)
                return await download_image(filename, subfolder)
    except (KeyError, IndexError) as e:
        logger.error("Failed to parse result: %s", e)

    return None
