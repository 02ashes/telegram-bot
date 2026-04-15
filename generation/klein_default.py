"""Flux 2 Klein 'default' edit workflow — ReferenceLatent conditioning."""

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


def build_flux_klein_default_edit_workflow(
    prompt: str,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int | None = None,
    has_image2: bool = False,
    model_name: str = "flux-2-klein-9b.safetensors",
    weight_dtype: str = "fp8_e4m3fn",
    lora_name: str = "",
    lora_strength: float = 1.0,
) -> dict:
    """Build Flux 2 Klein 'default' image editing workflow for ComfyUI API.

    Unlike the depth-based workflow, this uses NO depth/canny extraction.
    Uses empty latent + ReferenceLatent conditioning (from the reference
    workflow image_flux2_klein_image_edit_4b_distilled).

    Pipeline:
        1. Load + scale image to 1 megapixel
        2. GetImageSize → EmptyFlux2LatentImage + Flux2Scheduler
        3. CLIPTextEncode (positive) → ConditioningZeroOut (negative)
        4. Reference Conditioning: VAEEncode(image) → ReferenceLatent for pos & neg
        5. Optional: second Reference Conditioning for image2
        6. RandomNoise + KSamplerSelect(euler) + CFGGuider + SamplerCustomAdvanced
        7. VAEDecode → SaveImage

    Supports 2 images for multi-reference editing.
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

        # ---- Load & Scale Image to 1MP ----
        "139": {
            "class_type": "LoadImage",
            "inputs": {"image": "edit_input.png"},
        },
        "146": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["139", 0],
                "upscale_method": "nearest-exact",
                "megapixels": 1.0,
                "resolution_steps": 1,
            },
        },

        # ---- GetImageSize → EmptyFlux2LatentImage + Flux2Scheduler ----
        "180": {
            "class_type": "GetImageSize",
            "inputs": {"image": ["146", 0]},
        },
        "181": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": ["180", 0],
                "height": ["180", 1],
                "batch_size": 1,
            },
        },
        "182": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": steps,
                "width": ["180", 0],
                "height": ["180", 1],
            },
        },

        # ---- CLIP Text Encode (positive) ----
        "108": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["107", 0]},
        },

        # ---- ConditioningZeroOut (negative = zeroed positive) ----
        "109": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["108", 0]},
        },

        # ---- Reference Conditioning for image1 ----
        # VAEEncode image → ReferenceLatent for pos + neg
        "150": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["146", 0],
                "vae": ["110", 0],
            },
        },
        "151": {  # ReferenceLatent positive
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["108", 0],
                "latent": ["150", 0],
            },
        },
        "152": {  # ReferenceLatent negative
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["109", 0],
                "latent": ["150", 0],
            },
        },

        # ---- Sampler ----
        "170": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "171": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "172": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["106", 0],  # updated below if LoRA
                "positive": ["151", 0],  # updated below if image2
                "negative": ["152", 0],  # updated below if image2
                "cfg": cfg,
            },
        },
        "173": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["170", 0],
                "guider": ["172", 0],
                "sampler": ["171", 0],
                "sigmas": ["182", 0],
                "latent_image": ["181", 0],
            },
        },

        # ---- Decode + Save ----
        "104": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["173", 0], "vae": ["110", 0]},
        },
        "201": {
            "class_type": "SaveImage",
            "inputs": {"images": ["104", 0], "filename_prefix": "TGBot_Edit_Default"},
        },
    }

    # Optional LoRA
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
    workflow["172"]["inputs"]["model"] = last_model_ref

    # ---- Optional: Second Reference Conditioning for image2 ----
    if has_image2:
        # Load & scale image2
        workflow["400"] = {
            "class_type": "LoadImage",
            "inputs": {"image": "ref_image2.png"},
        }
        workflow["401"] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["400", 0],
                "upscale_method": "nearest-exact",
                "megapixels": 1.0,
                "resolution_steps": 1,
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
        # ReferenceLatent for positive: chain after image1's ref
        workflow["403"] = {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["151", 0],  # after image1 pos ref
                "latent": ["402", 0],
            },
        }
        # ReferenceLatent for negative: chain after image1's neg ref
        workflow["404"] = {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["152", 0],  # after image1 neg ref
                "latent": ["402", 0],
            },
        }
        # Redirect CFGGuider to use image2's outputs
        workflow["172"]["inputs"]["positive"] = ["403", 0]
        workflow["172"]["inputs"]["negative"] = ["404", 0]

    # Skin Enhance + Film Grain post-processing
    _add_skin_enhance_and_grain(workflow, ["104", 0], "201")

    return workflow




async def run_image_edit_default(
    image_bytes: bytes,
    prompt: str,
    steps: int = 4,
    cfg: float = 1.0,
    image2_bytes: bytes | None = None,
    lora_name: str = "",
    lora_strength: float = 1.0,
) -> bytes | None:
    """Image editing via Flux 2 Klein 'default' mode on RunPod Serverless.

    No depth/canny. Uses empty latent + ReferenceLatent conditioning.
    Supports 2 separate images (not stitched — each as its own reference).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Save image1 as PNG
    edit_b64 = _image_to_png_b64(image_bytes)

    has_image2 = image2_bytes is not None

    # Save image2 as PNG if provided
    img2_b64 = _image_to_png_b64(image2_bytes) if has_image2 else None

    workflow = build_flux_klein_default_edit_workflow(
        prompt=prompt,
        steps=steps,
        cfg=cfg,
        has_image2=has_image2,
        lora_name=lora_name,
        lora_strength=lora_strength,
    )

    images = [{"name": "edit_input.png", "image": edit_b64}]
    if has_image2 and img2_b64:
        images.append({"name": "ref_image2.png", "image": img2_b64})

    logger.info("Default edit job (steps=%d, has_image2=%s)", steps, has_image2)
    job_id = await submit_job(workflow, images)
    if not job_id:
        return None

    output = await poll_job(job_id, timeout=600)
    if not output:
        return None

    try:
        result = _extract_file_from_output(output)
        if not result:
            logger.error("Could not extract image from RunPod output")
            return None
        return result
    except Exception as e:
        logger.error("Failed to parse default edit result: %s", e)
        return None

