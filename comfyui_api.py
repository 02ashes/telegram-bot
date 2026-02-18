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


def build_flux_klein_edit_workflow(
    prompt: str,
    negative: str = "",
    denoise: float = 1.0,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int | None = None,
    has_reference: bool = False,
    crop_x: int = 0,
    crop_y: int = 0,
    crop_w: int = 0,
    crop_h: int = 0,
    lora_name: str = "",
    lora_strength: float = 0.7,
) -> dict:
    """Build Flux 2 Klein 9B advanced image editing workflow for ComfyUI API.

    Uses ReferenceLatent conditioning with depth and canny edge maps to
    preserve face and body shape while allowing strong edits (denoise=1.0).

    Pipeline:
        1. Load + resize image to 1024 (keep proportions)
        2. Extract depth map (DepthAnything V2) and canny edges
        3. Encode original, depth, canny as latents
        4. Chain ReferenceLatent nodes for positive conditioning:
           prompt → ref(original) → ref(depth) → ref(canny) → KSampler
        5. Chain ReferenceLatent nodes for negative conditioning:
           negative → ref(original) → ref(depth) → ref(canny) → KSampler
        6. KSampler (euler/simple, denoise=1.0) → VAEDecode → SaveImage
    """
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))

    workflow = {
        # ---- Models ----
        "106": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux-2-klein-9b.safetensors",
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "107": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_3_8b.safetensors",
                "type": "flux2",
            },
        },
        "110": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "flux2-vae.safetensors"},
        },

        # ---- Load & Resize Image ----
        "139": {
            "class_type": "LoadImage",
            "inputs": {"image": "edit_input.png"},
        },
        "146": {
            "class_type": "ImageResize+",
            "inputs": {
                "image": ["139", 0],
                "width": 1024,
                "height": 1024,
                "interpolation": "lanczos",
                "method": "keep proportion",
                "condition": "always",
                "multiple_of": 0,
            },
        },

        # ---- Encode original image as latent ----
        "141": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["146", 0],
                "vae": ["110", 0],
            },
        },

        # ---- Depth map extraction + encode ----
        "148": {
            "class_type": "DownloadAndLoadDepthAnythingV2Model",
            "inputs": {
                "model_name": "depth_anything_v2_vits_fp16.safetensors",
            },
        },
        "147": {
            "class_type": "DepthAnything_V2",
            "inputs": {
                "da_model": ["148", 0],
                "images": ["146", 0],
            },
        },
        "152": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["147", 0],
                "vae": ["110", 0],
            },
        },

        # ---- Canny edge extraction + encode ----
        "161": {
            "class_type": "Canny",
            "inputs": {
                "image": ["146", 0],
                "low_threshold": 0.2,
                "high_threshold": 0.7,
            },
        },
        "162": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["161", 0],
                "vae": ["110", 0],
            },
        },

        # ---- CLIP Text Encode ----
        "108": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["107", 0]},
        },
        "109": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["107", 0]},
        },

        # ---- Positive conditioning: chain ReferenceLatent (original → depth → canny) ----
        "140": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["108", 0],
                "latent": ["141", 0],
            },
        },
        "151": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["140", 0],
                "latent": ["152", 0],
            },
        },
        "160": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["151", 0],
                "latent": ["162", 0],
            },
        },

        # ---- Negative conditioning: chain ReferenceLatent (original → depth → canny) ----
        "149": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["109", 0],
                "latent": ["141", 0],
            },
        },
        "156": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["149", 0],
                "latent": ["152", 0],
            },
        },
        "159": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["156", 0],
                "latent": ["162", 0],
            },
        },

        # ---- KSampler ----
        "138": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["106", 0],  # updated to LoRA output if lora_name set
                "positive": ["160", 0],
                "negative": ["159", 0],
                "latent_image": ["141", 0],
                "seed": seed,
                "control_after_generate": "randomize",
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise,
            },
        },

        # ---- Decode + Save ----
        "104": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["138", 0], "vae": ["110", 0]},
        },
    }

    # Optional LoRA (e.g. NSFW LoRA for Klein)
    if lora_name:
        workflow["144"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["106", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
            },
        }
        # Redirect KSampler model input to LoRA output
        workflow["138"]["inputs"]["model"] = ["144", 0]

    if has_reference and crop_w > 0 and crop_h > 0:
        # Crop right half of stitched canvas
        workflow["200"] = {
            "class_type": "ImageCrop",
            "inputs": {
                "image": ["104", 0],
                "width": crop_w,
                "height": crop_h,
                "x": crop_x,
                "y": crop_y,
            },
        }
        workflow["201"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["200", 0], "filename_prefix": "TGBot_Edit"},
        }
    else:
        workflow["201"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["104", 0], "filename_prefix": "TGBot_Edit"},
        }

    return workflow


def build_wan_i2v_workflow(
    prompt: str,
    negative: str = "",
    audio_enabled: bool = False,
    audio_prompt: str = "",
    audio_negative: str = "music, speech, talking, noise, static",
    frames: int = 33,
    fps: int = 16,
    width: int = 720,
    height: int = 1280,
    seed: int | None = None,
) -> dict:
    """Build a WAN 2.2 Remix NSFW I2V workflow with optional MMAudio.

    Uses dual-sampler architecture with LoRA 4-step acceleration.
    Based on Wan2.2-Remix-comfy-i2v-workflow.json (FastUnsharpSharpen removed).
    """
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))

    if not negative:
        negative = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

    workflow = {
        # --- Model Loaders ---
        # CLIP (text encoder)
        "97": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "nsfw_wan_umt5-xxl_fp8_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        # UNET high lighting
        "77": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v2.1.safetensors",
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        # UNET low lighting
        "103": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v2.1.safetensors",
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        # LoRA high noise → high lighting model
        "112": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["77", 0],
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "strength_model": 1.0,
            },
        },
        # LoRA low noise → low lighting model
        "113": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["103", 0],
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                "strength_model": 1.0,
            },
        },
        # ModelSamplingSD3 shift=8 for high lighting
        "54": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["112", 0],
                "shift": 8,
            },
        },
        # ModelSamplingSD3 shift=8 for low lighting
        "55": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["113", 0],
                "shift": 8,
            },
        },
        # VAE
        "39": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors",
            },
        },
        # --- Text Encoding ---
        # Positive prompt
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["97", 0],
            },
        },
        # Negative prompt
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": ["97", 0],
            },
        },
        # --- Image Input ---
        "106": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "video_input.png",
            },
        },
        # --- WAN Image to Video ---
        "107": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "vae": ["39", 0],
                "start_image": ["106", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
            },
        },
        # --- Dual Sampler ---
        # KSampler #1 (high lighting, steps 0 → split_step)
        "57": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["54", 0],
                "positive": ["107", 0],
                "negative": ["107", 1],
                "latent_image": ["107", 2],
                "noise_seed": seed,
                "add_noise": "enable",
                "return_with_leftover_noise": "enable",
                "steps": 4,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 2,
            },
        },
        # KSampler #2 (low lighting, steps split_step → end)
        "58": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["55", 0],
                "positive": ["107", 0],
                "negative": ["107", 1],
                "latent_image": ["57", 0],
                "noise_seed": seed,
                "add_noise": "disable",
                "return_with_leftover_noise": "disable",
                "steps": 4,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 2,
                "end_at_step": 10000,
            },
        },
        # --- Decode ---
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["58", 0],
                "vae": ["39", 0],
            },
        },
        # --- Output Video (VHS_VideoCombine → proper h264-mp4) ---
        "99": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "TGBot_Video",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }

    # --- Optional MMAudio ---
    if audio_enabled and audio_prompt:
        # MMAudio model loader
        workflow["109"] = {
            "class_type": "MMAudioModelLoader",
            "inputs": {
                "mmaudio_model": "mmaudio_large_44k_nsfw_gold_8.5k_final.pth",
                "precision": "fp16",
                "base_precision": "fp16",
            },
        }
        # MMAudio feature utils (requires explicit model file names)
        workflow["110"] = {
            "class_type": "MMAudioFeatureUtilsLoader",
            "inputs": {
                "vae_model": "mmaudio_vae_44k_fp16.safetensors",
                "synchformer_model": "mmaudio_synchformer_fp32.safetensors",
                "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
                "precision": "fp16",
                "mode": "44k",
            },
        }
        # MMAudio sampler
        workflow["111"] = {
            "class_type": "MMAudioSampler",
            "inputs": {
                "mmaudio_model": ["109", 0],
                "feature_utils": ["110", 0],
                "images": ["8", 0],
                "duration": 8.0,
                "steps": 25,
                "cfg": 4.5,
                "seed": seed,
                "prompt": audio_prompt,
                "negative_prompt": audio_negative,
                "force_offload": True,
                "mask_away_clip": False,
            },
        }
        # Connect audio to VHS_VideoCombine
        workflow["99"]["inputs"]["audio"] = ["111", 0]


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


async def poll_job(job_id: str, timeout: int = 600) -> dict | None:
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
    # Log the full output structure for debugging
    logger.info("RunPod output type: %s", type(output).__name__)
    if isinstance(output, dict):
        logger.info("RunPod output keys: %s", list(output.keys()))
        for key, val in output.items():
            if isinstance(val, list):
                logger.info("  output[%s]: list of %d items", key, len(val))
                if val:
                    first = val[0]
                    if isinstance(first, dict):
                        logger.info("    first item keys: %s", list(first.keys()))
                    elif isinstance(first, str):
                        logger.info("    first item: str of %d chars", len(first))
            elif isinstance(val, str):
                logger.info("  output[%s]: str of %d chars", key, len(val))
            else:
                logger.info("  output[%s]: %s", key, type(val).__name__)
    elif isinstance(output, str):
        logger.info("RunPod output is a raw string of %d chars", len(output))
    
    try:
        image_bytes = _extract_file_from_output(output)
        if not image_bytes:
            logger.error("Could not extract image from RunPod output")
            return None
        return image_bytes
    except Exception as e:
        logger.error("Failed to parse result: %s (type: %s)", e, type(e).__name__)
        logger.error("Output preview: %s", str(output)[:500])
        return None


async def run_image_edit(
    image_bytes: bytes,
    prompt: str,
    negative: str = "blurry, ugly, deformed, watermark, text, low quality",
    denoise: float = 0.5,
    steps: int = 28,
    cfg: float = 3.5,
    image2_bytes: bytes | None = None,
    lora_name: str = "",
    lora_strength: float = 1.0,
) -> bytes | None:
    """Full image editing pipeline via Flux 2 Klein on RunPod Serverless.

    1 image:  Simple img2img edit by prompt
    2 images: Stitch side-by-side → edit → crop right half as result
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    has_reference = image2_bytes is not None
    crop_x, crop_y, crop_w, crop_h = 0, 0, 0, 0

    if has_reference:
        # Stitch reference (left) + target (right) side by side
        img2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB")
        # Resize img2 to same height as img
        img2 = img2.resize((int(img2.width * orig_h / img2.height), orig_h), Image.LANCZOS)
        stitched_w = img2.width + orig_w
        stitched = Image.new("RGB", (stitched_w, orig_h))
        stitched.paste(img2, (0, 0))
        stitched.paste(img, (img2.width, 0))
        img = stitched
        # Crop params: right half = the target image area
        crop_x = img2.width
        crop_y = 0
        crop_w = orig_w
        crop_h = orig_h
        logger.info("Stitched canvas: %dx%d (ref=%dx%d + target=%dx%d)",
                     stitched_w, orig_h, img2.width, orig_h, orig_w, orig_h)

    # Convert to PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    edit_png_bytes = buf.getvalue()
    edit_b64 = base64.b64encode(edit_png_bytes).decode("utf-8")

    workflow = build_flux_klein_edit_workflow(
        prompt=prompt,
        negative=negative,
        denoise=denoise,
        steps=steps,
        cfg=cfg,
        has_reference=has_reference,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
        lora_name=lora_name,
        lora_strength=lora_strength,
    )

    images = [{"name": "edit_input.png", "image": edit_b64}]

    logger.info("Submitting image edit job (denoise=%.2f, ref=%s)", denoise, has_reference)
    job_id = await submit_job(workflow, images)
    if not job_id:
        return None

    output = await poll_job(job_id, timeout=600)
    if not output:
        return None

    try:
        image_bytes = _extract_file_from_output(output)
        if not image_bytes:
            logger.error("Could not extract image from RunPod output")
            return None
        return image_bytes
    except Exception as e:
        logger.error("Failed to parse edit result: %s", e)
        return None


async def run_video(
    image_bytes: bytes,
    prompt: str,
    negative: str = "",
    audio_enabled: bool = False,
    audio_prompt: str = "",
    audio_negative: str = "music, speech, talking, noise, static",
    frames: int = 33,
    fps: int = 16,
    width: int = 0,
    height: int = 0,
) -> bytes | None:
    """Full video pipeline via RunPod Serverless.

    1. Resize input image to target resolution
    2. Submit WAN I2V workflow + image to RunPod Serverless
    3. Poll for result
    4. Return result video bytes (mp4)
    """

    # Open image and auto-detect resolution if not explicitly set
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    if width == 0 or height == 0:
        # Auto-detect: scale so longest side = 1280, round to multiple of 16
        max_side = 1280
        if max(orig_w, orig_h) > max_side:
            scale = max_side / max(orig_w, orig_h)
            width = int(orig_w * scale)
            height = int(orig_h * scale)
        else:
            width = orig_w
            height = orig_h
        # Round to nearest multiple of 16 (required by WAN model)
        width = max(16, (width // 16) * 16)
        height = max(16, (height // 16) * 16)
        logger.info("Auto-detected resolution: %dx%d → %dx%d", orig_w, orig_h, width, height)

    img = img.resize((width, height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resized_bytes = buf.getvalue()

    # Encode as base64
    image_b64 = _image_to_base64(resized_bytes)

    # Build workflow
    logger.info(
        "Building WAN I2V workflow (prompt=%s, frames=%d, fps=%d, %dx%d, audio=%s)...",
        prompt[:50], frames, fps, width, height, audio_enabled,
    )
    workflow = build_wan_i2v_workflow(
        prompt=prompt,
        negative=negative,
        audio_enabled=audio_enabled,
        audio_prompt=audio_prompt,
        audio_negative=audio_negative,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
    )

    # Submit job with image
    images = [
        {
            "name": "video_input.png",
            "image": image_b64,
        }
    ]

    logger.info("Submitting video job to RunPod Serverless (endpoint: %s)...", config.RUNPOD_ENDPOINT_ID)
    job_id = await submit_job(workflow, images)

    if not job_id:
        logger.error("Failed to submit video job")
        return None

    # Poll for result (video takes longer, use 600s timeout)
    logger.info("Waiting for video result (job: %s)...", job_id)
    output = await poll_job(job_id, timeout=600)

    if not output:
        return None

    # Extract result video from output
    # Worker 5.0+ format: {"images": [{"filename": "...", "type": "base64", "data": "..."}]}
    logger.info("RunPod video output type: %s", type(output).__name__)
    if isinstance(output, dict):
        logger.info("RunPod video output keys: %s", list(output.keys()))

    try:
        video_bytes = _extract_file_from_output(output, prefer_video=True)
        if not video_bytes:
            logger.error("Could not extract video from RunPod output")
            return None

        # Verify it looks like an MP4 (starts with ftyp box or mdat)
        if len(video_bytes) > 8:
            logger.info(
                "Video output: %d bytes, magic: %s",
                len(video_bytes),
                video_bytes[:12].hex(),
            )

        return video_bytes
    except Exception as e:
        logger.error("Failed to parse video result: %s (type: %s)", e, type(e).__name__)
        logger.error("Output preview: %s", str(output)[:500])
        return None


def _extract_file_from_output(output, prefer_video: bool = False) -> bytes | None:
    """Extract file bytes from RunPod worker 5.0+ output.

    Worker output format:
        {"images": [{"filename": "...", "type": "base64", "data": "..."}]}

    When prefer_video=True, tries to find a video file (.mp4, .webm) first.
    Falls back to first available file.
    """
    if not isinstance(output, dict):
        # Raw base64 string fallback
        if isinstance(output, str):
            if "," in output and output.startswith("data:"):
                output = output.split(",", 1)[1]
            return base64.b64decode(output)
        logger.error("Unexpected output type: %s", type(output).__name__)
        return None

    items = output.get("images", [])
    if not items:
        # Legacy format: {"message": "base64..."}
        if "message" in output and isinstance(output["message"], str):
            return base64.b64decode(output["message"])
        logger.error("No 'images' in output. Keys: %s", list(output.keys()))
        return None

    VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

    target_item = None
    if prefer_video:
        # Find a video file first
        for item in items:
            fn = item.get("filename", "")
            ext = fn[fn.rfind("."):].lower() if "." in fn else ""
            if ext in VIDEO_EXTS:
                target_item = item
                logger.info("Found video file: %s", fn)
                break

    if target_item is None:
        target_item = items[0]
        logger.info("Using first output file: %s", target_item.get("filename", "unknown"))

    # Extract base64 data
    b64_str = target_item.get("data") or target_item.get("image") or target_item.get("base64")
    if not b64_str:
        logger.error("No data field in output item: %s", list(target_item.keys()))
        return None

    # Strip data URI prefix if present
    if isinstance(b64_str, str) and b64_str.startswith("data:") and "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    return base64.b64decode(b64_str)

