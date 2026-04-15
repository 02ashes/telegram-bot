"""ComfyUI API client — RunPod Serverless version.

Workflow builders and runtime functions for image/video generation.
RunPod HTTP client, LoRA config, and post-processing live in generation/ subpackage.
"""

import base64
import io
import json
import logging
import os
import uuid

from PIL import Image, ImageOps

import config

# ── Re-export from generation submodules for backward compatibility ──
from generation.runpod import (
    submit_job,
    poll_job,
    cancel_job,
    get_job_status,
    close_session,
)
from generation.postprocess import (
    _image_to_base64,
    _image_to_png_b64,
    add_skin_enhance_and_grain as _add_skin_enhance_and_grain,
    strip_image_metadata as _strip_image_metadata,
    add_silent_audio as _add_silent_audio,
    extract_file_from_output as _extract_file_from_output,
)
from generation.loras import (
    CHARACTER_LORAS,
    detect_character_loras,
)

logger = logging.getLogger(__name__)




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

    # Skin Enhance + Film Grain post-processing
    _add_skin_enhance_and_grain(workflow, ["9", 0], "10")

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
def build_dark_bfs_workflow(
    prompt: str,
    bfs_prompt: str = (
        "Replace the woman's face and hair in Image1 with the woman's face "
        "and hair in Image2, without changing the clothing. Man roles and "
        "other places remain completely unchanged. Maintain consistent "
        "color tones for better integration of characters into the environment."
    ),
    base_width: int = 352,
    base_height: int = 640,
    bfs_width: int = 864,
    bfs_height: int = 1632,
    steps: int = 5,
    cfg: float = 1.0,
    seed: int | None = None,
    model_name: str = "darkBeastMar0326Latest_dbkleinv2BFS.safetensors",
    weight_dtype: str = "fp8_e4m3fn",
    clip_name: str = "qwen_3_8b.safetensors",
    vae_name: str = "flux2-vae.safetensors",
    character_loras: list[dict] | None = None,
) -> dict:
    """Build Klein9b-BFS face swap workflow — 1:1 with the workflow JSON.

    Pass 1: EmptyLatent → KSampler (text2img, denoise=1)
    Pass 2: LatentUpscale ×1.5 → KSampler (refine, denoise=0.5)
    BFS:    ReferenceConditioning(bfs_prompt, ref_face, source=pass2) →
            SamplerCustomAdvanced → VAEDecode → output
    """
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))

    workflow = {
        # ---- Models (nodes 18, 4, 6) ----
        "18": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": weight_dtype},
        },
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux2"},
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },

        # ---- Prompts (nodes 28, 5, 29, 10) ----
        "28": {  # Main generation prompt
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 0]},
        },
        "5": {  # BFS face swap prompt
            "class_type": "CLIPTextEncode",
            "inputs": {"text": bfs_prompt, "clip": ["4", 0]},
        },
        "29": {  # ZeroOut(main) → negative for Pass 1&2
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["28", 0]},
        },
        "10": {  # ZeroOut(BFS) → negative for BFS pass
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]},
        },

        # ---- Face reference image (node 20) ----
        "20": {
            "class_type": "LoadImage",
            "inputs": {"image": "face_ref.png"},
        },

        # ---- PASS 1: Text2img at base resolution ----
        "30": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": base_width, "height": base_height, "batch_size": 1},
        },
        "27": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["18", 0],
                "positive": ["28", 0],
                "negative": ["29", 0],
                "latent_image": ["30", 0],
                "seed": seed, "control_after_generate": "fixed",
                "steps": steps, "cfg": cfg,
                "sampler_name": "euler", "scheduler": "simple",
                "denoise": 1.0,
            },
        },

        # ---- PASS 2: Latent upscale ×1.5 + refine ----
        "36": {
            "class_type": "LatentUpscaleBy",
            "inputs": {
                "samples": ["27", 0],
                "upscale_method": "bicubic",
                "scale_by": 1.5,
            },
        },
        "35": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["18", 0],
                "positive": ["28", 0],
                "negative": ["29", 0],
                "latent_image": ["36", 0],
                "seed": seed, "control_after_generate": "fixed",
                "steps": steps, "cfg": cfg,
                "sampler_name": "euler", "scheduler": "simple",
                "denoise": 0.5,
            },
        },
        "37": {  # Decode Pass 2 → source image for BFS
            "class_type": "VAEDecode",
            "inputs": {"samples": ["35", 0], "vae": ["6", 0]},
        },

        # ---- BFS Reference Conditioning (expanded from subgraph 22) ----
        # Scale ref-image (face photo) to ~1 megapixel
        "bfs_scale_ref": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["20", 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0, "resolution_steps": 1,
            },
        },
        # Scale source image (Pass 2 output) to BFS target resolution
        "bfs_scale_src": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["37", 0],
                "upscale_method": "lanczos",
                "width": bfs_width, "height": bfs_height,
                "crop": "center",
            },
        },
        # VAEEncode source → latent_source
        "bfs_vae_src": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["bfs_scale_src", 0], "vae": ["6", 0]},
        },
        # ReferenceLatent — condition BFS positive on source
        "bfs_rl_pos_src": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["5", 0],
                "latent": ["bfs_vae_src", 0],
            },
        },
        # ReferenceLatent — condition BFS negative on source
        "bfs_rl_neg_src": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["10", 0],
                "latent": ["bfs_vae_src", 0],
            },
        },
        # VAEEncode face ref → latent_ref
        "bfs_vae_ref": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["bfs_scale_ref", 0], "vae": ["6", 0]},
        },
        # Double ReferenceLatent — chain positive on ref
        "bfs_rl_pos_ref": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["bfs_rl_pos_src", 0],
                "latent": ["bfs_vae_ref", 0],
            },
        },
        # Double ReferenceLatent — chain negative on ref
        "bfs_rl_neg_ref": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["bfs_rl_neg_src", 0],
                "latent": ["bfs_vae_ref", 0],
            },
        },
        # GetImageSize for empty latent dimensions
        "bfs_imgsize": {
            "class_type": "GetImageSize",
            "inputs": {"image": ["bfs_scale_src", 0]},
        },
        # Empty latent at BFS target resolution
        "bfs_empty": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": ["bfs_imgsize", 0],
                "height": ["bfs_imgsize", 1],
                "batch_size": 1,
            },
        },

        # ---- BFS Sampling (nodes 15, 2, 17, 1, 14) ----
        "15": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": steps,
                "width": ["bfs_imgsize", 0],
                "height": ["bfs_imgsize", 1],
            },
        },
        "2": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["18", 0],
                "positive": ["bfs_rl_pos_ref", 0],
                "negative": ["bfs_rl_neg_ref", 0],
                "cfg": cfg,
            },
        },
        "17": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "1": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["17", 0],
                "guider": ["2", 0],
                "sampler": ["1", 0],
                "sigmas": ["15", 0],
                "latent_image": ["bfs_empty", 0],
            },
        },

        # ---- Final decode + output (nodes 3, 40, 61) ----
        "3": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["14", 0], "vae": ["6", 0]},
        },
        "61": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "TGBot_DarkBFS"},
        },
    }

    # Optional character LoRAs
    last_model_ref = ["18", 0]
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
        # Point all 3 samplers to the LoRA chain
        workflow["27"]["inputs"]["model"] = last_model_ref
        workflow["35"]["inputs"]["model"] = last_model_ref
        workflow["2"]["inputs"]["model"] = last_model_ref

    # Skin Enhance + Film Grain post-processing
    _add_skin_enhance_and_grain(workflow, ["3", 0], "61")


    return workflow


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
    logger.debug("RunPod output: type=%s", type(output).__name__)
    
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


# -------------------------------------------------------------------------
# Dark Generate BFS — Klein9b face swap via RunPod
# -------------------------------------------------------------------------
async def run_dark_generate_bfs(
    face_image_bytes: bytes,
    prompt: str,
    bfs_prompt: str = "",
    base_width: int = 352,
    base_height: int = 640,
    bfs_width: int = 864,
    bfs_height: int = 1632,
    steps: int = 5,
) -> bytes | None:
    """Run Klein9b-BFS face swap pipeline on RunPod.

    1. Build 3-pass workflow (text2img → refine → BFS face swap)
    2. Upload face reference image
    3. Submit & poll for result
    """
    char_loras = detect_character_loras(prompt)

    # Prepare face image
    face_b64 = _image_to_png_b64(face_image_bytes)

    kwargs = {
        "prompt": prompt,
        "base_width": base_width,
        "base_height": base_height,
        "bfs_width": bfs_width,
        "bfs_height": bfs_height,
        "steps": steps,
        "character_loras": char_loras,
    }
    if bfs_prompt:
        kwargs["bfs_prompt"] = bfs_prompt

    workflow = build_dark_bfs_workflow(**kwargs)
    images = [{"name": "face_ref.png", "image": face_b64}]

    logger.info(
        "Dark BFS job: prompt=%s, base=%dx%d, bfs=%dx%d, steps=%d",
        prompt[:60], base_width, base_height, bfs_width, bfs_height, steps,
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
            logger.error("Could not extract image from BFS output")
            return None
        return result
    except Exception as e:
        logger.error("Failed to parse BFS result: %s", e)
        return None


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



async def check_video_status(job_id: str, endpoint_id: str | None = None) -> dict:
    """Check video generation job status and extract result if complete.

    Returns dict with:
        {"status": "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED",
         "video": "base64..." (only when COMPLETED),
         "error": "..." (only when FAILED)}
    """
    status_data = await get_job_status(job_id, endpoint_id=endpoint_id)
    status = status_data.get("status", "UNKNOWN")

    if status == "COMPLETED":
        output = status_data.get("output")
        if not output:
            return {"status": "FAILED", "error": "No output in completed job"}

        try:
            video_bytes = _extract_file_from_output(output, prefer_video=True)
            if not video_bytes:
                return {"status": "FAILED", "error": "Could not extract video from output"}

            if len(video_bytes) > 8:
                logger.info("Video output: %d bytes, magic: %s",
                            len(video_bytes), video_bytes[:12].hex())

            # Add silent audio track so Telegram treats it as video, not GIF
            video_bytes = _add_silent_audio(video_bytes)

            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
            return {"status": "COMPLETED", "video": video_b64}
        except Exception as e:
            logger.error("Failed to extract video: %s", e)
            return {"status": "FAILED", "error": str(e)}

    elif status == "FAILED":
        return {"status": "FAILED", "error": status_data.get("error", "Unknown error")}

    elif status == "CANCELLED":
        return {"status": "CANCELLED"}

    else:
        # IN_QUEUE, IN_PROGRESS, etc.
        return {"status": status}





# ============================================================
# Kenpechi SVI — 6-scene multi-pass video workflow
# ============================================================

# Node ID maps for template patching
_KENPECHI_PROMPT_IDS = ["975", "1024", "1025", "1030", "1139", "1313"]
_KENPECHI_NEGATIVE_ID = "973"
_KENPECHI_DURATION_IDS = ["851", "1330", "1331", "1332", "1333", "1334"]
_KENPECHI_SEED_IDS = ["1415", "1416", "1417", "1418", "1419", "1420"]
_KENPECHI_LORA_HIGH_IDS = ["1098", "1112", "1114", "1116", "1133", "1309"]
_KENPECHI_LORA_LOW_IDS = ["1111", "1113", "1115", "1117", "1135", "1310"]

# Load template once at import time
import copy
import pathlib as _pathlib

_KENPECHI_TEMPLATE_PATH = _pathlib.Path(__file__).parent / "kenpechi_template.json"
_KENPECHI_TEMPLATE: dict | None = None


def _load_kenpechi_template() -> dict:
    """Load and cache the Kenpechi SVI workflow template."""
    global _KENPECHI_TEMPLATE
    if _KENPECHI_TEMPLATE is None:
        with open(_KENPECHI_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            _KENPECHI_TEMPLATE = json.load(f)
        logger.info("Kenpechi template loaded (%d nodes)", len(_KENPECHI_TEMPLATE))
    return _KENPECHI_TEMPLATE


def build_kenpechi_svi_workflow(
    scenes: list[dict],
    negative: str = "",
    width: int = 720,
    height: int = 1072,
    steps: int = 7,
    split_steps: int = 3,
    fps: float = 16,
    rife_multiplier: int = 4,
    svi_motion_strength: float = 1.0,
    repulsion_boost: float = 1.0,
    shift: float = 5.0,
) -> dict:
    """Build a Kenpechi SVI 6-scene workflow by patching the template.

    Args:
        scenes: List of 6 scene dicts, each with:
            - prompt (str): Scene prompt text
            - duration (float): Duration in seconds (e.g. 2.5)
            - seed (int): Seed (-1 for random)
            - high_loras (list[dict]): Up to 4 HIGH LoRA configs
              Each: {"on": bool, "lora": str, "strength": float}
            - low_loras (list[dict]): Up to 4 LOW LoRA configs
              Each: {"on": bool, "lora": str, "strength": float}
    """
    template = _load_kenpechi_template()
    wf = copy.deepcopy(template)

    # --- Global parameters ---
    wf["860"]["inputs"]["value"] = width
    wf["861"]["inputs"]["value"] = height
    wf["853"]["inputs"]["value"] = steps
    wf["1194"]["inputs"]["value"] = split_steps
    wf["1363"]["inputs"]["value"] = fps
    wf["1347"]["inputs"]["value"] = rife_multiplier
    wf["1270"]["inputs"]["value"] = repulsion_boost
    wf["1271"]["inputs"]["value"] = svi_motion_strength

    # Patch ModelSamplingSD3 shift for both HIGH and LOW
    wf["833"]["inputs"]["shift"] = shift
    wf["834"]["inputs"]["shift"] = shift

    # Patch LoadImage node to use uploaded file name
    wf["864"]["inputs"]["image"] = "video_input.png"

    # Negative prompt
    if negative:
        wf[_KENPECHI_NEGATIVE_ID]["inputs"]["text"] = negative

    # --- Per-scene patching ---
    for i, scene in enumerate(scenes):
        if i >= 6:
            break

        # Prompt
        wf[_KENPECHI_PROMPT_IDS[i]]["inputs"]["text"] = scene.get("prompt", "")

        # Duration (seconds)
        wf[_KENPECHI_DURATION_IDS[i]]["inputs"]["value"] = scene.get("duration", 2.5)

        # Seed
        seed = scene.get("seed", -1)
        if seed == -1:
            seed = int(uuid.uuid4().int % (2**32))
        wf[_KENPECHI_SEED_IDS[i]]["inputs"]["seed"] = seed

        # HIGH LoRAs (up to 4 slots)
        high_loras = scene.get("high_loras", [])
        high_node = wf[_KENPECHI_LORA_HIGH_IDS[i]]
        for slot in range(4):
            key = f"lora_{slot + 1}"
            if slot < len(high_loras):
                lora_cfg = high_loras[slot]
                high_node["inputs"][key] = {
                    "on": lora_cfg.get("on", False),
                    "lora": lora_cfg.get("lora", "None"),
                    "strength": lora_cfg.get("strength", 1.0),
                }
            else:
                high_node["inputs"][key] = {
                    "on": False,
                    "lora": "None",
                    "strength": 1,
                }

        # LOW LoRAs (up to 4 slots)
        low_loras = scene.get("low_loras", [])
        low_node = wf[_KENPECHI_LORA_LOW_IDS[i]]
        for slot in range(4):
            key = f"lora_{slot + 1}"
            if slot < len(low_loras):
                lora_cfg = low_loras[slot]
                low_node["inputs"][key] = {
                    "on": lora_cfg.get("on", False),
                    "lora": lora_cfg.get("lora", "None"),
                    "strength": lora_cfg.get("strength", 1.0),
                }
            else:
                low_node["inputs"][key] = {
                    "on": False,
                    "lora": "None",
                    "strength": 1,
                }

    logger.info(
        "Kenpechi SVI workflow built: %d scenes, %dx%d, steps=%d, rife=%dx",
        len(scenes), width, height, steps, rife_multiplier,
    )
    return wf



async def submit_kenpechi_video(
    image_bytes: bytes,
    scenes: list[dict],
    negative: str = "",
    width: int = 720,
    height: int = 1072,
    steps: int = 7,
    split_steps: int = 3,
    fps: float = 16,
    rife_multiplier: int = 4,
    svi_motion_strength: float = 1.0,
    repulsion_boost: float = 1.0,
    shift: float = 5.0,
) -> str | None:
    """Submit Kenpechi SVI video generation job to RunPod.

    1. Resize input image to target resolution
    2. Build Kenpechi SVI workflow (6-scene template patching)
    3. Submit to Kenpechi RunPod Serverless endpoint
    4. Return job_id (does NOT wait for completion)
    """
    # Send image as-is — workflow handles resolution via its own nodes
    image_b64 = _image_to_base64(image_bytes)

    logger.info(
        "Building Kenpechi SVI workflow (%d scenes, %dx%d, steps=%d)...",
        len(scenes), width, height, steps,
    )
    workflow = build_kenpechi_svi_workflow(
        scenes=scenes,
        negative=negative,
        width=width,
        height=height,
        steps=steps,
        split_steps=split_steps,
        fps=fps,
        rife_multiplier=rife_multiplier,
        svi_motion_strength=svi_motion_strength,
        repulsion_boost=repulsion_boost,
        shift=shift,
    )

    images = [{"name": "video_input.png", "image": image_b64}]

    logger.info("Submitting Kenpechi video job to RunPod (endpoint: %s)...", config.RUNPOD_KENPECHI_ENDPOINT_ID)
    job_id = await submit_job(workflow, images, endpoint_id=config.RUNPOD_KENPECHI_ENDPOINT_ID)

    if not job_id:
        logger.error("Failed to submit Kenpechi video job")
        return None

    logger.info("Kenpechi video job submitted: %s", job_id)
    return job_id


# ============================================================
# ULTIMATE V9 — Test Mode (Z-Image Turbo + Detailers)
# ============================================================

_V9_TEMPLATE_PATH = _pathlib.Path(__file__).parent / "ULTIMATE_V9.json"
_V9_TEMPLATE: dict | None = None

# Nodes to remove from V9 workflow for API use (preview, cache cleanup, comparers)
_V9_REMOVE_NODES = [
    "9",     # SaveImage "Save RAW Image" — avoid returning 2 images
    "58",    # PreviewAny (resolution display)
    "68",    # Image Comparer (Detailer vs RAW)
    "98",    # cleanGpuUsed
    "99",    # clearCacheAll
    "101",   # Image Comparer (SeedVR2 vs RAW)
    "109",   # SeedVariance bypasser
    "179",   # cleanGpuUsed
    "180",   # clearCacheAll
    "321",   # cleanGpuUsed
    "322",   # clearCacheAll
    "364",   # PreviewImage (SeedVR2 output)
    "386",   # VAEDecode (Pass 1 preview)
    "387",   # PreviewImage (Pass 1)
]

# Detailer zone config: node IDs for LoRA loader, prompt, FastNodeBypasser, and prompt templates
_V9_DETAILER_ZONES = {
    "face": {
        "lora_node": "254",
        "prompt_node": "255",
        "bypass_node": "312",
        "bypass_key": "Enable 🧩  ZIT Face Lora Loader",
        "strength_model": 1.0,
        "prompt_template": "{trigger}, ultra-detailed portrait, photorealistic eyes, natural skin texture, realistic pores, smooth skin transitions, sharp focus, subtle lighting, perfectly clean white teeth, enhance original appearance without altering expression",
        "prompt_no_trigger": "ultra-detailed portrait, photorealistic eyes, natural skin texture, realistic pores, smooth skin transitions, sharp focus, subtle lighting, perfectly clean white teeth, enhance original appearance without altering expression",
        "always_on": True,  # LoRA always enabled even without character
    },
    "eyes": {
        "lora_node": "260",
        "prompt_node": "277",
        "bypass_node": "314",
        "bypass_key": "Enable 🧩  ZIT Eyes Lora Loader",
        "strength_model": 0.9,
        "prompt_template": "{trigger}, highly detailed eyes, glossy pupils, sharp eyelashes, smooth eyelids, realistic texture, fine iris details, preserve original eyes",
        "prompt_no_trigger": "highly detailed eyes, glossy pupils, sharp eyelashes, smooth eyelids, realistic texture, fine iris details, preserve original eyes",
        "always_on": True,
    },
    "hands": {
        "lora_node": "261",
        "prompt_node": "282",
        "bypass_node": "315",
        "bypass_key": "Enable 🧩  ZIT Hands Lora Loader",
        "strength_model": 0.8,
        "prompt_template": "{trigger}, realistic hands, natural finger proportions, detailed fingernails, soft skin texture, natural lighting, sharp focus, subtle veins, elegant posture",
        "prompt_no_trigger": "realistic hands, natural finger proportions, detailed fingernails, soft skin texture, natural lighting, sharp focus, subtle veins, elegant posture",
        "always_on": True,
    },
    "foot": {
        "lora_node": "262",
        "prompt_node": "295",
        "bypass_node": "316",
        "bypass_key": "Enable 🧩  ZIT Foot Lora Loader",
        "strength_model": 0.8,
        "prompt_template": "{trigger}, realistic feet, natural toe proportions, detailed toenails, soft skin texture, subtle veins, natural lighting, sharp focus, elegant posture",
        "prompt_no_trigger": "realistic feet, natural toe proportions, detailed toenails, soft skin texture, subtle veins, natural lighting, sharp focus, elegant posture",
        "always_on": False,  # Disabled when no character LoRA
    },
    "nipples": {
        "lora_node": "263",
        "prompt_node": "300",
        "bypass_node": "318",
        "bypass_key": "Enable 🧩  ZIT Nipples Lora Loader",
        "strength_model": 0.85,
        "prompt_template": "{trigger}, realistic small breasts, small nipples, natural skin texture, detailed nipples, subtle shading, soft transitions, sharp focus, photorealistic, elegant anatomy",
        "prompt_no_trigger": "realistic small breasts, small nipples, natural skin texture, detailed nipples, subtle shading, soft transitions, sharp focus, photorealistic, elegant anatomy",
        "always_on": False,
    },
    "pussy": {
        "lora_node": "264",
        "prompt_node": "302",
        "bypass_node": "319",
        "bypass_key": "Enable 🧩  ZIT Pussy Lora Loader",
        "strength_model": 0.9,
        "prompt_template": "{trigger}, shaved pussy, realistic small shaved vagina, subtle labia, natural clitoral hood, innie anatomy, soft skin texture, slight gloss, photorealistic, sharp focus, elegant details",
        "prompt_no_trigger": "shaved pussy, realistic small shaved vagina, subtle labia, natural clitoral hood, innie anatomy, soft skin texture, slight gloss, photorealistic, sharp focus, elegant details",
        "always_on": False,
    },
}

# Fallback LoRA when no character is detected
_V9_FALLBACK_LORA = "nicegirls_Zimage.safetensors"


def _load_v9_template() -> dict:
    """Load and cache the ULTIMATE_V9 workflow template."""
    global _V9_TEMPLATE
    if _V9_TEMPLATE is None:
        with open(_V9_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            _V9_TEMPLATE = json.load(f)
        logger.info("V9 template loaded (%d nodes)", len(_V9_TEMPLATE))
    return _V9_TEMPLATE


def build_v9_workflow(
    prompt: str,
    negative: str = "",
    aspect_ratio: str = "5:7 (Balanced Portrait)",
    character_loras: list[dict] | None = None,
) -> dict:
    """Build ULTIMATE_V9 workflow by deep-copying template and patching.

    Args:
        prompt: User positive prompt (natural language)
        negative: Negative prompt (optional)
        aspect_ratio: FluxResolutionNode aspect ratio string
        character_loras: List from detect_character_loras() or None
    """
    template = _load_v9_template()
    wf = copy.deepcopy(template)

    # ── Remove preview/cache/comparer nodes ──
    for nid in _V9_REMOVE_NODES:
        wf.pop(nid, None)

    # Fix dangling references after removing cache nodes.
    # Full original chain:
    #   8(VAEDecode) → 179(clean) → 180(clear) → 267(scale1.5x) → 321(clean) → 322(clear)
    #                 → FaceDetailer chain → "299:396" (detailer output)
    #                 → 98(clean) → 99(clear) → 377(AdvancedImageDenoiser) → post-processing → SeedVR2 → Save
    # Removed: 179, 180, 321, 322, 98, 99
    # Fixes needed:
    #   1) 267: image ← "8" (was 180)
    #   2) FaceDetailer chain input: ← "267" (was 322)
    #   3) 377: image ← "299:396" (was 99, detailer chain output)

    # Fix 1: 267 (ImageScaleBy 1.5x): was ← 180, now ← 8
    if "267" in wf:
        wf["267"]["inputs"]["image"] = ["8", 0]

    # Fix 2: Find any node that referenced removed 322 or 321 and redirect to 267
    removed_cache_ids = {"179", "180", "321", "322", "98", "99"}
    for nid, node in wf.items():
        inputs = node.get("inputs", {})
        for key, val in inputs.items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                if val[0] in removed_cache_ids:
                    # Determine correct redirect
                    if val[0] in ("321", "322"):
                        # Was between 267 and FaceDetailer → redirect to 267
                        wf[nid]["inputs"][key] = ["267", 0]
                        logger.info("V9 fix: node %s.%s: %s → 267", nid, key, val[0])
                    elif val[0] in ("179", "180"):
                        # Was between 8 and 267 → redirect to 8
                        wf[nid]["inputs"][key] = ["8", 0]
                        logger.info("V9 fix: node %s.%s: %s → 8", nid, key, val[0])
                    elif val[0] in ("98", "99"):
                        # Was between detailer output and 377 → redirect to "299:396"
                        wf[nid]["inputs"][key] = ["299:396", 0]
                        logger.info("V9 fix: node %s.%s: %s → 299:396", nid, key, val[0])

    # ── Core prompts ──
    wf["175"]["inputs"]["text"] = prompt
    if negative:
        wf["104"]["inputs"]["text"] = negative

    # ── Seed (random) — rgthree Seed max is 2^50 ──
    import random
    wf["388"]["inputs"]["seed"] = random.randint(0, 2**50)

    # ── Resolution ──
    wf["28"]["inputs"]["aspect_ratio"] = aspect_ratio

    # ── Model switch: always ST (Input=1) ──
    wf["344"]["inputs"]["Input"] = 1

    # ── Character LoRA logic ──
    has_character = bool(character_loras)
    if has_character:
        char = character_loras[0]  # Use first detected character
        trigger = char["trigger"]
        lora_name = char["lora_name"]
        # V9: Always force strength 1.0 for character LoRA in main merger
        lora_strength = 1.0
    else:
        trigger = ""
        lora_name = _V9_FALLBACK_LORA
        lora_strength = 1.0

    # ── Node 391 (NSFW LoRA Merger Pass 1): slot lora_2 = character/fallback ──
    wf["391"]["inputs"]["lora_2"] = lora_name
    wf["391"]["inputs"]["strength_2"] = lora_strength

    # ── Detailer zones ──
    for zone_name, zone_cfg in _V9_DETAILER_ZONES.items():
        lora_node = zone_cfg["lora_node"]
        prompt_node = zone_cfg["prompt_node"]
        bypass_node = zone_cfg["bypass_node"]
        bypass_key = zone_cfg["bypass_key"]

        if has_character and zone_cfg["always_on"]:
            # Character detected + Face/Eyes/Hands: enable with character LoRA
            wf[lora_node]["inputs"]["lora_name"] = lora_name
            wf[lora_node]["inputs"]["strength_model"] = zone_cfg["strength_model"]
            wf[prompt_node]["inputs"]["text"] = zone_cfg["prompt_template"].replace("{trigger}", trigger)
            wf[bypass_node]["inputs"][bypass_key] = True
        elif zone_cfg["always_on"]:
            # No character + Face/Eyes/Hands: use fallback LoRA
            wf[lora_node]["inputs"]["lora_name"] = _V9_FALLBACK_LORA
            wf[lora_node]["inputs"]["strength_model"] = zone_cfg["strength_model"]
            wf[prompt_node]["inputs"]["text"] = zone_cfg["prompt_no_trigger"]
            wf[bypass_node]["inputs"][bypass_key] = True
        else:
            # Foot/Nipples/Pussy: ALWAYS disable LoRA (character LoRA hurts anatomy)
            wf[bypass_node]["inputs"][bypass_key] = False
            wf[prompt_node]["inputs"]["text"] = zone_cfg["prompt_no_trigger"]

    # ── Penis detailer: always static (ZPenisv2, no character LoRA) ──
    # Node 350 (Penis Lora Loader) + 347 (Penis prompt) + 351 (bypass) stay as-is

    logger.info(
        "V9 workflow built: aspect=%s, char=%s, lora=%s",
        aspect_ratio,
        trigger if has_character else "(none)",
        lora_name,
    )
    return wf





async def run_v9_generate(
    prompt: str,
    negative: str = "",
    aspect_ratio: str = "5:7 (Balanced Portrait)",
    lora_strength_override: float | None = None,
) -> bytes | None:
    """Generate image via ULTIMATE V9 pipeline on dedicated endpoint.

    Auto-detects character LoRAs from prompt, patches detailers,
    submits to V9 endpoint, polls for result, strips metadata.
    """
    # Auto-detect character LoRAs
    char_loras = detect_character_loras(prompt)

    # Apply strength override if provided
    if lora_strength_override is not None and char_loras:
        for lora in char_loras:
            lora["strength"] = lora_strength_override
            logger.info("V9 LoRA %s strength overridden to %.2f", lora["lora_name"], lora_strength_override)

    # Build workflow
    workflow = build_v9_workflow(
        prompt=prompt,
        negative=negative,
        aspect_ratio=aspect_ratio,
        character_loras=char_loras,
    )

    logger.info(
        "V9 Generate job: aspect=%s, chars=%s",
        aspect_ratio,
        [c["trigger"] for c in char_loras] if char_loras else "(none)",
    )

    # Submit to V9 endpoint
    job_id = await submit_job(workflow, images=[], endpoint_id=config.RUNPOD_V9_ENDPOINT_ID)
    if not job_id:
        return None

    # Poll for result (V9 pipeline is slow: dual-pass + detailers + SeedVR2 + post-process)
    output = await poll_job(job_id, timeout=900, endpoint_id=config.RUNPOD_V9_ENDPOINT_ID)
    if not output:
        return None

    try:
        result = _extract_file_from_output(output)
        if not result:
            logger.error("Could not extract image from V9 output")
            return None
        # Strip metadata to prevent workflow leakage
        result = _strip_image_metadata(result)
        return result
    except Exception as e:
        logger.error("Failed to parse V9 result: %s", e)
        return None

