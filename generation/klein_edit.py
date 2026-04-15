"""Flux 2 Klein edit workflow — depth+canny editing + Dark Beast variant."""

import base64
import io
import logging
import uuid

from PIL import Image, ImageOps

import config
from generation.runpod import submit_job, poll_job
from generation.postprocess import (
    _image_to_base64,
    _image_to_png_b64,
    add_skin_enhance_and_grain as _add_skin_enhance_and_grain,
    extract_file_from_output as _extract_file_from_output,
)
from generation.loras import detect_character_loras

logger = logging.getLogger(__name__)


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
    model_name: str = "flux-2-klein-9b.safetensors",
    weight_dtype: str = "fp8_e4m3fn",
    character_loras: list[dict] | None = None,
    has_image2: bool = False,
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
                "unet_name": model_name,
                "weight_dtype": weight_dtype,
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
                "model": "depth_anything_v2_vits_fp16.safetensors",
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
    last_model_ref = ["106", 0]
    if lora_name:
        workflow["144"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": last_model_ref,
                "lora_name": lora_name,
                "strength_model": lora_strength,
            },
        }
        last_model_ref = ["144", 0]

    # Chain character LoRAs
    if character_loras:
        for i, char_lora in enumerate(character_loras):
            node_id = str(300 + i)
            workflow[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": last_model_ref,
                    "lora_name": char_lora["lora_name"],
                    "strength_model": char_lora["strength"],
                },
            }
            last_model_ref = [node_id, 0]

    # Redirect KSampler model input to last LoRA output
    workflow["138"]["inputs"]["model"] = last_model_ref

    # ---- Optional: Second reference image (separate ReferenceLatent chain) ----
    # This follows the official Flux 2 Klein 4B workflow approach:
    # Image2 → ScaleToPixels → VAEEncode → ReferenceLatent for pos & neg
    # Chains AFTER the image1 reference chain.
    if has_image2:
        # Load & resize image2
        workflow["400"] = {
            "class_type": "LoadImage",
            "inputs": {"image": "ref_image2.png"},
        }
        workflow["401"] = {
            "class_type": "ImageResize+",
            "inputs": {
                "image": ["400", 0],
                "width": 1024,
                "height": 1024,
                "interpolation": "lanczos",
                "method": "keep proportion",
                "condition": "always",
                "multiple_of": 0,
            },
        }
        # VAEEncode image2
        workflow["402"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["401", 0],
                "vae": ["110", 0],
            },
        }
        # ReferenceLatent for positive: chain after image1's last ref (canny=160)
        workflow["403"] = {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["160", 0],  # after image1's canny ref
                "latent": ["402", 0],
            },
        }
        # ReferenceLatent for negative: chain after image1's last neg ref (canny=159)
        workflow["404"] = {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["159", 0],  # after image1's neg canny ref
                "latent": ["402", 0],
            },
        }
        # Redirect KSampler to use image2's reference outputs
        workflow["138"]["inputs"]["positive"] = ["403", 0]
        workflow["138"]["inputs"]["negative"] = ["404", 0]

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
        # Skin Enhance + Film Grain post-processing
        _add_skin_enhance_and_grain(workflow, ["200", 0], "201")
    else:
        workflow["201"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["104", 0], "filename_prefix": "TGBot_Edit"},
        }
        # Skin Enhance + Film Grain post-processing
        _add_skin_enhance_and_grain(workflow, ["104", 0], "201")

    return workflow




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



# Dark Beast model config
# Fast  = base Klein 9B + Dark Beast LoRA (fp16, 1.23 GB) — fast cold-start
# Detailed = full Dark Beast bf16 checkpoint (31 GB) — better quality
DARK_BEAST_MODELS = {
    "fast": {
        "model_name": "flux-2-klein-9b.safetensors",
        "weight_dtype": "fp8_e4m3fn",
        "lora_name": "dark_beast_klein_v1.5_blitz_fp16.safetensors",
        "lora_strength": 1.0,
    },
    "detailed": {
        "model_name": "dark_beast_klein_blitz_bf16.safetensors",
        "weight_dtype": "default",
        "lora_name": "",
        "lora_strength": 0.0,
    },
}


async def run_image_edit_dark(
    image_bytes: bytes,
    prompt: str,
    negative: str = "blurry, ugly, deformed, watermark, text, low quality",
    denoise: float = 0.5,
    steps: int = 5,
    cfg: float = 1.0,
    quality: str = "fast",
    image2_bytes: bytes | None = None,
) -> bytes | None:
    """Full image editing pipeline via Dark Beast Klein on RunPod Serverless.

    Same pipeline as run_image_edit but uses the Dark Beast fine-tuned model.
    quality: 'fast' (Klein 9B + LoRA) or 'detailed' (full bf16 checkpoint)
    """
    model_cfg = DARK_BEAST_MODELS.get(quality, DARK_BEAST_MODELS["fast"])

    # Auto-detect character LoRAs from prompt
    char_loras = detect_character_loras(prompt)

    has_image2 = image2_bytes is not None

    edit_b64 = _image_to_png_b64(image_bytes)
    img2_b64 = _image_to_png_b64(image2_bytes) if has_image2 else None

    workflow = build_flux_klein_edit_workflow(
        prompt=prompt,
        negative=negative,
        denoise=denoise,
        steps=steps,
        cfg=cfg,
        has_reference=False,
        crop_x=0,
        crop_y=0,
        crop_w=0,
        crop_h=0,
        model_name=model_cfg["model_name"],
        weight_dtype=model_cfg["weight_dtype"],
        lora_name=model_cfg["lora_name"],
        lora_strength=model_cfg["lora_strength"],
        character_loras=char_loras,
        has_image2=has_image2,
    )

    images = [{"name": "edit_input.png", "image": edit_b64}]
    if has_image2 and img2_b64:
        images.append({"name": "ref_image2.png", "image": img2_b64})

    logger.info("Dark Beast edit job (quality=%s, denoise=%.2f, model=%s, lora=%s)",
                quality, denoise, model_cfg["model_name"], model_cfg["lora_name"])
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
        logger.error("Failed to parse dark edit result: %s", e)
        return None

