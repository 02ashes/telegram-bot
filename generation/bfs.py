"""Dark Beast BFS face swap workflows.

- Klein9b 3-pass pipeline (legacy): build_dark_bfs_workflow / run_dark_generate_bfs
- Final T2I + BFS v5 (new):        build_t2i_bfs_workflow / run_t2i_bfs
"""

import base64
import copy
import io
import json
import logging
import uuid
from pathlib import Path

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


# ============================================================
# Final T2I + BFS v5 — ZIT quality body + Klein BFS face swap
# ============================================================

_T2I_BFS_TEMPLATE: dict | None = None
_T2I_BFS_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "Final t2i + BFS v5.json"


def _load_t2i_bfs_template() -> dict:
    """Load and cache the T2I+BFS workflow template."""
    global _T2I_BFS_TEMPLATE
    if _T2I_BFS_TEMPLATE is None:
        with open(_T2I_BFS_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            _T2I_BFS_TEMPLATE = json.load(f)
        logger.info("Loaded T2I+BFS template: %s (%d nodes)", _T2I_BFS_TEMPLATE_PATH.name, len(_T2I_BFS_TEMPLATE))
    return _T2I_BFS_TEMPLATE


def build_t2i_bfs_workflow(
    prompt: str,
    negative: str = "blurry image",
    aspect_ratio: str = "5:7 (Balanced Portrait)",
    seed: int | None = None,
    shift: float = 3.0,
) -> dict:
    """Build T2I+BFS workflow from template JSON.

    Patches:
        Node 175 — positive prompt
        Node 104 — negative prompt
        Node 508 — reference face image (LoadImage → face_ref.png)
        Node 28  — aspect ratio
        Node 391 — sampler seed
        Node 394 — refine seed
        Node 524 — BFS noise seed
    """
    if seed is None:
        seed = int(uuid.uuid4().int % (2**63))

    template = _load_t2i_bfs_template()
    wf = copy.deepcopy(template)

    # ── Prompt ──
    wf["175"]["inputs"]["text"] = prompt
    wf["104"]["inputs"]["text"] = negative

    # ── Aspect ratio ──
    wf["28"]["inputs"]["aspect_ratio"] = aspect_ratio

    # ── Face reference (will be uploaded as face_ref.png) ──
    wf["508"]["inputs"]["image"] = "face_ref.png"

    # ── Seeds ──
    wf["391"]["inputs"]["seed"] = seed
    wf["394"]["inputs"]["seed"] = seed + 1
    wf["524"]["inputs"]["noise_seed"] = seed + 2

    # ── Remove preview/comparer nodes (not needed on server) ──
    for nid in ["51", "58", "509", "332"]:
        wf.pop(nid, None)

    # ── Remove old SaveImage nodes (keep only BFS final) ──
    # Node 9 = save raw T2I, Node 396 = save raw pass1 — not needed
    wf.pop("9", None)
    wf.pop("396", None)

    # ── Remove VRAM/Cache cleanup nodes & reconnect ──
    # Chain was: 8→179→180→370→98→99→373
    # Becomes:   8→370→373
    for nid in ["98", "99", "179", "180"]:
        wf.pop(nid, None)
    wf["370"]["inputs"]["image"] = ["8", 0]     # Denoiser ← VAEDecode
    wf["373"]["inputs"]["image"] = ["370", 0]   # CameraForensic ← Denoiser

    # ── Aura Flow Shift ──
    if "225" in wf:
        wf["225"]["inputs"]["shift"] = shift
    if "401" in wf:
        wf["401"]["inputs"]["shift"] = shift

    return wf


async def run_t2i_bfs(
    face_image_bytes: bytes,
    prompt: str,
    negative: str = "blurry image",
    aspect_ratio: str = "5:7 (Balanced Portrait)",
    shift: float = 3.0,
) -> bytes | None:
    """Run Final T2I + BFS v5 pipeline on RunPod V9 endpoint.

    1. ZIT T2I generates high-quality body/scene
    2. CameraForensic + CRT post-processing
    3. Klein BFS swaps face from reference photo

    Returns PNG bytes or None on failure.
    """
    # Prepare face image
    face_b64 = _image_to_png_b64(face_image_bytes)

    workflow = build_t2i_bfs_workflow(
        prompt=prompt,
        negative=negative,
        aspect_ratio=aspect_ratio,
        shift=shift,
    )
    images = [{"name": "face_ref.png", "image": face_b64}]

    logger.info(
        "T2I+BFS job: prompt=%s, aspect=%s",
        prompt[:60], aspect_ratio,
    )

    job_id = await submit_job(
        workflow, images,
        endpoint_id=config.RUNPOD_V9_ENDPOINT_ID,
    )
    if not job_id:
        return None

    output = await poll_job(
        job_id, timeout=600,
        endpoint_id=config.RUNPOD_V9_ENDPOINT_ID,
    )
    if not output:
        return None

    try:
        result = _extract_file_from_output(output)
        if not result:
            logger.error("Could not extract image from T2I+BFS output")
            return None
        return result
    except Exception as e:
        logger.error("Failed to parse T2I+BFS result: %s", e)
        return None
