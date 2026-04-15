"""Dark Beast Z6 text2img — DarkBeastZ6 (Z-Image Turbo) two-pass generation."""

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


def build_dark_img2img_workflow(
    prompt: str,
    negative: str = "",
    denoise: float = 0.85,
    steps: int = 10,
    cfg: float = 1.0,
    seed: int | None = None,
    model_name: str = "dark_beast_klein_blitz_bf16.safetensors",
    weight_dtype: str = "default",
    lora_name: str = "",
    lora_strength: float = 1.0,
    character_loras: list[dict] | None = None,
) -> dict:
    """Build a simple img2img workflow for Flux models (Dark Beast).

    NO ReferenceLatent, NO depth/canny — the model has full creative
    freedom to reshape the image based on the prompt.

    Pipeline:
        LoadImage → Resize(1024) → VAEEncode → KSampler(denoise) → VAEDecode → Save
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

        # ---- Encode image to latent ----
        "141": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["146", 0],
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

        # ---- KSampler (simple img2img — no ReferenceLatent!) ----
        "138": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["106", 0],
                "positive": ["108", 0],
                "negative": ["109", 0],
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
        "201": {
            "class_type": "SaveImage",
            "inputs": {"images": ["104", 0], "filename_prefix": "TGBot_DarkGen"},
        },
    }

    # Optional LoRA (e.g. Dark Beast)
    last_model_ref = ["106", 0]  # UNETLoader output
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

    # Chain character LoRAs (e.g. misu, anya, etc.)
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

    # Point KSampler to the last model in the chain
    workflow["138"]["inputs"]["model"] = last_model_ref

    # Skin Enhance + Film Grain post-processing
    _add_skin_enhance_and_grain(workflow, ["104", 0], "201")

    return workflow


# -------------------------------------------------------------------------
# Klein9b-BFS Face Swap Workflow
# Faithfully reproduces Klein9b-BFS-20260303.json
#
# 3-pass pipeline:
#   Pass 1: Text2img (352×640) → KSampler (5 steps, denoise=1)
#   Pass 2: LatentUpscale ×1.5 → KSampler (5 steps, denoise=0.5)
#   BFS:    Reference Conditioning (double ReferenceLatent) →
#           SamplerCustomAdvanced (Flux2Scheduler, CFGGuider)
#
# Model:  darkBeastMar0326Latest_dbkleinv2BFS.safetensors (fp8_e4m3fn)
# CLIP:   qwen_3_8b.safetensors (flux2)
# VAE:    flux2-vae.safetensors
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Dark Generate V2 — DarkBeastZ6 (Z-Image Turbo) two-pass text2img
# -------------------------------------------------------------------------
DARK_GENERATE_MODELS = {
    "fast": {
        "model_name": "DarkBeastZ6-BlitZ-F32-single-transformers.safetensors",
        "weight_dtype": "default",
        "clip_name": "zImageTurbo_textEncoder.safetensors",
        "clip_type": "lumina2",
        "vae_name": "ae.safetensors",
        "steps_pass1": 8,
        "steps_pass2": 5,
        "latent_upscale": 1.25,
    },
}


def build_dark_generate_v2_workflow(
    prompt: str,
    negative: str = "",
    width: int = 768,
    height: int = 1440,
    seed: int | None = None,
    steps_pass1: int = 8,
    steps_pass2: int = 8,
    cfg: float = 1.0,
    model_name: str = "DarkBeastZ6-BlitZ-F32-single-transformers.safetensors",
    weight_dtype: str = "default",
    clip_name: str = "zImageTurbo_textEncoder.safetensors",
    clip_type: str = "lumina2",
    vae_name: str = "ae.safetensors",
    character_loras: list[dict] | None = None,
    latent_upscale: float = 1.25,
    upscale_denoise: float = 0.5,
) -> dict:
    """Build two-pass text2img workflow — DarkBeastZ6 on Z-Image Turbo.

    PASS 1: Text2img
      UNETLoader → [Character LoRAs] → KSampler
      CLIPLoader(lumina2) → CLIPTextEncode(prompt) → positive
      ConditioningZeroOut(prompt_cond) → negative
      EmptySD3LatentImage(w×h) → KSampler (8 steps, cfg 1, euler/simple, denoise=1)

    PASS 2: Latent upscale + refine
      LatentUpscaleBy(×1.25, bicubic) → KSampler2 (8 steps, denoise=0.5)
      VAEDecode → SaveImage
    """
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))

    workflow = {
        # ---- Models ----
        "81": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model_name,
                "weight_dtype": weight_dtype,
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": clip_type,
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },

        # ---- Prompt Encoding ----
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["3", 0]},
        },

        # ---- Negative: ConditioningZeroOut ----
        "47": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["7", 0]},
        },

        # ---- Empty Latent (text2img) ----
        "10": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },

        # ---- PASS 1: KSampler (text2img, denoise=1.0) ----
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["81", 0],
                "positive": ["7", 0],
                "negative": ["47", 0],
                "latent_image": ["10", 0],
                "seed": seed,
                "control_after_generate": "randomize",
                "steps": steps_pass1,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },

        # NOTE: Pass 1 preview (nodes 25, 12) removed — was causing
        # RunPod handler to return the wrong (pre-upscale) image as items[0].

        # ---- PASS 2: Latent Upscale + Refine ----
        "78": {
            "class_type": "LatentUpscaleBy",
            "inputs": {
                "samples": ["9", 0],
                "upscale_method": "bicubic",
                "scale_by": latent_upscale,
            },
        },
        "76": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["81", 0],
                "positive": ["7", 0],
                "negative": ["47", 0],
                "latent_image": ["78", 0],
                "seed": seed,
                "control_after_generate": "randomize",
                "steps": steps_pass2,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": upscale_denoise,
            },
        },

        # ---- Decode PASS 2 (final output) ----
        "77": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["76", 0], "vae": ["4", 0]},
        },
        "79": {
            "class_type": "SaveImage",
            "inputs": {"images": ["77", 0], "filename_prefix": "TGBot_DBZ6_v2"},
        },
    }

    # ---- Character LoRA chain ----
    last_model_ref = ["81", 0]

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

    # Point both KSamplers to the last model in the chain
    workflow["9"]["inputs"]["model"] = last_model_ref
    workflow["76"]["inputs"]["model"] = last_model_ref

    # Skin Enhance + Film Grain post-processing
    _add_skin_enhance_and_grain(workflow, ["77", 0], "79")

    return workflow




async def run_dark_generate(
    prompt: str,
    negative: str = "",
    width: int = 768,
    height: int = 1440,
    quality: str = "fast",
    lora_strength_override: float | None = None,
) -> bytes | None:
    """Two-pass text2img via Dark Beast Klein V2.

    Uses the original Dark Beast workflow: text2img → upscale ×1.25 → refine.
    Character LoRAs are auto-detected from the prompt.
    lora_strength_override: if set, overrides the default strength for all character LoRAs.
    """
    gen_cfg = DARK_GENERATE_MODELS.get(quality, DARK_GENERATE_MODELS["fast"])

    # Auto-detect character LoRAs from prompt
    char_loras = detect_character_loras(prompt)

    # Apply strength override from UI if provided
    if lora_strength_override is not None and char_loras:
        for lora in char_loras:
            lora["strength"] = lora_strength_override
            logger.info("LoRA %s strength overridden to %.2f", lora["lora_name"], lora_strength_override)

    images = []

    workflow = build_dark_generate_v2_workflow(
        prompt=prompt,
        negative=negative,
        width=width,
        height=height,
        steps_pass1=gen_cfg["steps_pass1"],
        steps_pass2=gen_cfg["steps_pass2"],
        cfg=1.0,
        model_name=gen_cfg["model_name"],
        weight_dtype=gen_cfg["weight_dtype"],
        clip_name=gen_cfg["clip_name"],
        clip_type=gen_cfg["clip_type"],
        vae_name=gen_cfg["vae_name"],
        character_loras=char_loras,
        latent_upscale=gen_cfg["latent_upscale"],
    )

    logger.info(
        "DarkBeastZ6 Generate job (quality=%s, %dx%d, pass1=%d steps, chars=%s)",
        quality, width, height, gen_cfg["steps_pass1"],
        [c["trigger"] for c in char_loras],
    )
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
        logger.error("Failed to parse dark generate result: %s", e)
        return None


