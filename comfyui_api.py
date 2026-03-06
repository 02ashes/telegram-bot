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
    else:
        workflow["201"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["104", 0], "filename_prefix": "TGBot_Edit"},
        }

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
        "40": {  # ImageScaleBy (scale=1 in workflow)
            "class_type": "ImageScaleBy",
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                "scale_by": 1.0,
            },
        },
        "61": {
            "class_type": "SaveImage",
            "inputs": {"images": ["40", 0], "filename_prefix": "TGBot_DarkBFS"},
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

    return workflow


# -------------------------------------------------------------------------
# WAN 2.2 NSFW Video Action LoRAs
# Source: tamin-akin/wan2.2-nsfw-lora
# Actions with high/low = separate LoRA per sampler (dual noise architecture)
# Actions with single = same LoRA for both samplers
# -------------------------------------------------------------------------
VIDEO_ACTIONS = {
    "none": {"label": "None", "high": None, "low": None},
    "blowjob": {
        "label": "🍆 POV Blowjob",
        "high": "pov-blowjob-i2v-v1.2.safetensors",
        "low": "pov-blowjob-i2v-v1.2.safetensors",
    },
    "deepthroat": {
        "label": "🫦 Deepthroat",
        "high": "jfj-deepthroat-W22-I2V-HN.safetensors",
        "low": "jfj-deepthroat-W22-I2V-LN.safetensors",
    },
    "doggy": {
        "label": "🐕 Front Doggy",
        "high": "front_doggy_plow_v1_1_wan.safetensors",
        "low": "front_doggy_plow_v1_1_wan.safetensors",
    },
    "missionary": {
        "label": "🛏️ POV Missionary",
        "high": "pov-missionary-i2v-high-v1.0.safetensors",
        "low": "pov-missionary-i2v-low-v1.0.safetensors",
    },
    "side_sex": {
        "label": "🔄 Side Sex",
        "high": "side-sex-i2v-v10.safetensors",
        "low": "side-sex-i2v-v10.safetensors",
    },
    "cowgirl": {
        "label": "🤠 Reverse Cowgirl",
        "high": "wan22.r3v3rs3_c0wg1rl-14b-High-i2v_e70.safetensors",
        "low": "wan22.r3v3rs3_c0wg1rl-14b-Low-i2v_e70.safetensors",
    },
    "fingering": {
        "label": "👆 Fingering",
        "high": "fingering-high-v1.0.safetensors",
        "low": "fingering-low-v1.0.safetensors",
    },
    "nipple": {
        "label": "💋 Nipple Stroke",
        "high": "nipple_stroke_WAN22_I2V_v1_high_noise.safetensors",
        "low": "nipple_stroke_WAN22_I2V_v1_low_noise.safetensors",
    },
    "allinone": {
        "label": "🔥 All-In-One",
        "high": "DR34ML4Y_I2V_14B_HIGH_v2.safetensors",
        "low": "DR34ML4Y_I2V_14B_LOW_v2.safetensors",
    },
}


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
    action: str = "none",
    shift: float = 5.0,
    cfg_high: float = 5.0,
    cfg_low: float = 1.0,
    lora_strength: float = 1.3,
    scheduler: str = "beta",
    steps: int = 20,
) -> dict:
    """Build a WAN 2.2 Remix NSFW I2V workflow with optional MMAudio.

    Uses dual-sampler architecture with multi-stage CFG:
    - High noise sampler: CFG=5 (stamps the action) with beta scheduler
    - Low noise sampler: CFG=1 (smooths details) with beta scheduler
    Optimal shift=5 for dynamic NSFW actions.
    Optional action LoRA at strength 1.3 to compensate for missing I2V keys.
    """
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))

    # Dual-sampler step split: 2/3 high noise, 1/3 low noise
    split_step = max(1, int(steps * 2 / 3))

    if not negative:
        negative = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

    # Get action LoRA config
    action_cfg = VIDEO_ACTIONS.get(action, VIDEO_ACTIONS["none"])
    lora_high = action_cfg.get("high")
    lora_low = action_cfg.get("low")

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
        # --- TeaCache (2-3x speedup, requires ComfyUI-TeaCache custom node) ---
        # Set to True after Docker image is rebuilt with TeaCache installed
        "54": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["77", 0],  # default: direct from UNET (overridden below if TeaCache)
                "shift": shift,
            },
        },
        "55": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["103", 0],  # default: direct from UNET
                "shift": shift,
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
        # --- Dual Sampler (split_step high + rest low) ---
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
                "steps": steps,
                "cfg": cfg_high,
                "sampler_name": "uni_pc",
                "scheduler": scheduler,
                "start_at_step": 0,
                "end_at_step": split_step,
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
                "steps": steps,
                "cfg": cfg_low,
                "sampler_name": "uni_pc",
                "scheduler": scheduler,
                "start_at_step": split_step,
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

    # --- EasyCache (built-in ComfyUI 0.14+, similar to TeaCache for speedup) ---
    EASYCACHE_ENABLED = True

    if EASYCACHE_ENABLED:
        # Insert EasyCache between UNET and ModelSamplingSD3
        workflow["200"] = {
            "class_type": "EasyCache",
            "inputs": {
                "model": ["77", 0],
                "reuse_threshold": 0.2,
                "start_percent": 0.15,
                "end_percent": 0.95,
                "verbose": False,
            },
        }
        workflow["201"] = {
            "class_type": "EasyCache",
            "inputs": {
                "model": ["103", 0],
                "reuse_threshold": 0.2,
                "start_percent": 0.15,
                "end_percent": 0.95,
                "verbose": False,
            },
        }
        # Wire: UNET → EasyCache → ModelSamplingSD3
        workflow["54"]["inputs"]["model"] = ["200", 0]
        workflow["55"]["inputs"]["model"] = ["201", 0]

    # --- Optional Action LoRA ---
    if lora_high:
        workflow["112"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["77", 0],
                "lora_name": lora_high,
                "strength_model": lora_strength,
            },
        }
        if EASYCACHE_ENABLED:
            # Chain: UNET → LoRA → EasyCache → ModelSamplingSD3
            workflow["200"]["inputs"]["model"] = ["112", 0]
        else:
            # Chain: UNET → LoRA → ModelSamplingSD3
            workflow["54"]["inputs"]["model"] = ["112", 0]

    if lora_low:
        workflow["113"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["103", 0],
                "lora_name": lora_low,
                "strength_model": lora_strength,
            },
        }
        if EASYCACHE_ENABLED:
            workflow["201"]["inputs"]["model"] = ["113", 0]
        else:
            workflow["55"]["inputs"]["model"] = ["113", 0]

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


async def poll_job(job_id: str, timeout: int = 1200) -> dict | None:
    """Poll RunPod Serverless for job completion.
    
    Returns the output dict or None on timeout/failure.
    Default timeout: 1200s (20 minutes) to handle long NSFW video generation.
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
                    elif status == "CANCELLED":
                        logger.info("Job %s was cancelled", job_id)
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


async def cancel_job(job_id: str) -> bool:
    """Cancel a running or queued RunPod job.
    
    Returns True if cancel request was sent successfully.
    """
    url = f"{RUNPOD_API_BASE}/{config.RUNPOD_ENDPOINT_ID}/cancel/{job_id}"
    headers = {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers) as resp:
                if resp.status == 200:
                    logger.info("Job %s cancel requested", job_id)
                    return True
                else:
                    logger.warning("Failed to cancel job %s: %s", job_id, resp.status)
                    return False
    except Exception as e:
        logger.error("Cancel error: %s", e)
        return False


async def get_job_status(job_id: str) -> dict:
    """Get current status of a RunPod job.
    
    Returns dict with {status, output?}.
    """
    url = f"{RUNPOD_API_BASE}/{config.RUNPOD_ENDPOINT_ID}/status/{job_id}"
    headers = {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                data = await resp.json()
                return {
                    "status": data.get("status", "UNKNOWN"),
                    "output": data.get("output"),
                    "error": data.get("error"),
                }
    except Exception as e:
        logger.error("Status check error: %s", e)
        return {"status": "ERROR", "error": str(e)}


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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    edit_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    has_image2 = image2_bytes is not None

    # Save image2 as PNG if provided
    img2_b64 = None
    if has_image2:
        img2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB")
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        img2_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

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


# -------------------------------------------------------------------------
# Character LoRAs — trigger word → LoRA filename + strength
# Add new characters here after training. Trigger word must be lowercase.
# -------------------------------------------------------------------------
CHARACTER_LORAS = {
    "misu": {"lora_name": "misu_z6_step1000.safetensors", "strength": 0.95},
    "anya":  {"lora_name": "anya_lora.safetensors", "strength": 0.9},
    "jane":  {"lora_name": "janelora.safetensors", "strength": 0.9},
    "lera":  {"lora_name": "leralora.safetensors", "strength": 0.9},
    "mirana": {"lora_name": "miranalora.safetensors", "strength": 0.9},
    "moondina": {"lora_name": "moonlora.safetensors", "strength": 0.9},
}


def detect_character_loras(prompt: str) -> list[dict]:
    """Detect character trigger words in prompt, return list of LoRA configs."""
    prompt_lower = prompt.lower()
    found = []
    for trigger, cfg in CHARACTER_LORAS.items():
        if trigger in prompt_lower:
            found.append({"trigger": trigger, **cfg})
            logger.info("Detected character LoRA: %s → %s", trigger, cfg["lora_name"])
    return found


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

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    has_image2 = image2_bytes is not None

    # Save image1 as PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    edit_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Save image2 as PNG if provided
    img2_b64 = None
    if has_image2:
        img2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB")
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        img2_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

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
    face_img = Image.open(io.BytesIO(face_image_bytes)).convert("RGB")
    buf = io.BytesIO()
    face_img.save(buf, format="PNG")
    face_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

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

    return workflow


async def run_dark_generate(
    prompt: str,
    negative: str = "",
    width: int = 768,
    height: int = 1440,
    quality: str = "fast",
    reference_bytes: bytes | None = None,
    lora_strength_override: float | None = None,
) -> bytes | None:
    """Two-pass text2img via Dark Beast Klein V2.

    Uses the original Dark Beast workflow: text2img → upscale ×1.5 → refine.
    Optionally accepts a reference image for ReferenceLatent conditioning.
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


async def submit_video(
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
    action: str = "none",
    shift: float = 5.0,
    cfg_high: float = 5.0,
    cfg_low: float = 1.0,
    lora_strength: float = 1.3,
    scheduler: str = "beta",
    steps: int = 20,
) -> str | None:
    """Submit video generation job to RunPod (returns immediately).

    1. Resize input image to target resolution
    2. Build WAN I2V workflow
    3. Submit to RunPod Serverless
    4. Return job_id (does NOT wait for completion)
    """

    # Open image and auto-detect resolution if not explicitly set
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    if width == 0 or height == 0:
        max_side = 1280
        if max(orig_w, orig_h) > max_side:
            scale = max_side / max(orig_w, orig_h)
            width = int(orig_w * scale)
            height = int(orig_h * scale)
        else:
            width = orig_w
            height = orig_h
        width = max(16, (width // 16) * 16)
        height = max(16, (height // 16) * 16)
        logger.info("Auto-detected resolution: %dx%d → %dx%d", orig_w, orig_h, width, height)

    img = img.resize((width, height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resized_bytes = buf.getvalue()

    image_b64 = _image_to_base64(resized_bytes)

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
        action=action,
        shift=shift,
        cfg_high=cfg_high,
        cfg_low=cfg_low,
        lora_strength=lora_strength,
        scheduler=scheduler,
        steps=steps,
    )

    images = [{"name": "video_input.png", "image": image_b64}]

    logger.info("Submitting video job to RunPod Serverless (endpoint: %s)...", config.RUNPOD_ENDPOINT_ID)
    job_id = await submit_job(workflow, images)

    if not job_id:
        logger.error("Failed to submit video job")
        return None

    logger.info("Video job submitted: %s", job_id)
    return job_id


async def check_video_status(job_id: str) -> dict:
    """Check video generation job status and extract result if complete.

    Returns dict with:
        {"status": "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED",
         "video": "base64..." (only when COMPLETED),
         "error": "..." (only when FAILED)}
    """
    status_data = await get_job_status(job_id)
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

