"""Kenpechi SVI — 6-scene multi-pass video workflow."""

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

import copy
import json
import pathlib as _pathlib
from generation.runpod import get_job_status
from generation.postprocess import add_silent_audio as _add_silent_audio

logger = logging.getLogger(__name__)


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



_KENPECHI_PROMPT_IDS = ["975", "1024", "1025", "1030", "1139", "1313"]
_KENPECHI_NEGATIVE_ID = "973"
_KENPECHI_DURATION_IDS = ["851", "1330", "1331", "1332", "1333", "1334"]
_KENPECHI_SEED_IDS = ["1415", "1416", "1417", "1418", "1419", "1420"]
_KENPECHI_LORA_HIGH_IDS = ["1098", "1112", "1114", "1116", "1133", "1309"]
_KENPECHI_LORA_LOW_IDS = ["1111", "1113", "1115", "1117", "1135", "1310"]

# Load template once at import time
import copy
import pathlib as _pathlib

_KENPECHI_TEMPLATE_PATH = _pathlib.Path(__file__).parent.parent / "kenpechi_template.json"
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
