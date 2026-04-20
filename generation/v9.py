"""ULTIMATE V9 — high-fidelity dual-pass generation pipeline."""

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
from generation.postprocess import strip_image_metadata as _strip_image_metadata

logger = logging.getLogger(__name__)


_V9_TEMPLATE_PATH = _pathlib.Path(__file__).parent.parent / "ULTIMATE_V9.json"
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
        lora_name = _V9_FALLBACK_LORA  # still need a valid filename for the node
        lora_strength = 0.0  # effectively disabled — pure base model

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
            # No character + Face/Eyes/Hands: disable LoRA — pure base model
            wf[lora_node]["inputs"]["lora_name"] = _V9_FALLBACK_LORA
            wf[lora_node]["inputs"]["strength_model"] = 0.0
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

