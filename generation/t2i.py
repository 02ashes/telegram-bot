"""Final T2I — text-to-image generation pipeline.

Two-pass generation + optional SDXL Pre-Definer + up to 7 detailers +
optional SeedVR2 upscale + CameraForensic + CRT post-processing.
"""

import copy
import json
import logging
import pathlib
import random

from generation.loras import detect_character_loras
from generation.postprocess import (
    extract_file_from_output as _extract_file_from_output,
    strip_image_metadata as _strip_image_metadata,
)
from generation.runpod import submit_job, poll_job
import config

logger = logging.getLogger(__name__)

# ── Template ──
_T2I_TEMPLATE_PATH = pathlib.Path(__file__).parent.parent / "Final t2i API 2.json"
_T2I_TEMPLATE: dict | None = None

# ── Nodes to always remove (preview, comparers, extra saves) ──
_REMOVE_NODES = [
    "9",     # SaveImage "Save RAW Image"
    "51",    # Image Comparer (Color Graded vs RAW)
    "58",    # PreviewAny (resolution display)
    "68",    # Image Comparer (Detailer vs RAW)
    "101",   # Image Comparer (SeedVR2 vs RAW)
    "179",   # cleanGpuUsed
    "180",   # clearCacheAll
    "286",   # Image Comparer (PreDefined vs RAW)
    "332",   # PreviewImage (Final)
    "396",   # SaveImage (Pass 1 RAW)
    "98",    # cleanGpuUsed (before SeedVR2)
    "99",    # clearCacheAll (before SeedVR2)
]

# ── SeedVR2 upscale nodes (removed when upscale=False) ──
_UPSCALE_NODES = [
    "91",    # ImageScaleBy (downscale 0.5x)
    "92",    # SeedVR2LoadDiTModel
    "93",    # SeedVR2LoadVAEModel
    "94",    # SeedVR2VideoUpscaler
]

# ── SDXL Pre-Definer nodes (removed when redefine=False) ──
_PREDEFINER_NODES = [
    "309:308",  # UltimateSDUpscaleNoUpscale (Pre-Definer)
    "323",      # easy imageSwitch (Face Preservation Switch)
    "324:358",  # UltralyticsDetectorProvider (Face — preservation)
    "324:359",  # FaceDetailer (Face Preservation)
    "324:360",  # ImageCompositeMasked
    "324:361",  # GrowMaskWithBlur
]

# ── SDXL model nodes (removed when neither redefine nor SDXL detailers are used) ──
_SDXL_NODES = [
    "176",   # CheckpointLoaderSimple (lustifySDXL)
    "183",   # CLIPSetLastLayer
    "185",   # Power Lora Loader (SDXL)
    "177",   # Negative Prompt (SDXL)
    "178",   # Positive Prompt (SDXL)
]

# ── Detailer zone config ──
_DETAILER_ZONES = {
    "face": {
        "lora_node": "254",
        "prompt_node": "255",
        "bypass_node": "314",
        "bypass_key": "Enable \U0001f9e9  SDXL Face Lora Loader",
        "strength_model": 1.0,
        "detailer_node": "240:239",
        "detector_node": "240:238",
        "prompt_template": "{trigger}, ultra-detailed portrait, photorealistic eyes, natural skin texture, realistic pores, smooth skin transitions, sharp focus, subtle lighting, perfectly clean white teeth, enhance original appearance without altering expression",
        "prompt_no_trigger": "ultra-detailed portrait, photorealistic eyes, natural skin texture, realistic pores, smooth skin transitions, sharp focus, subtle lighting, perfectly clean white teeth, enhance original appearance without altering expression",
    },
    "eyes": {
        "lora_node": "260",
        "prompt_node": "277",
        "bypass_node": "316",
        "bypass_key": "Enable \U0001f9e9  SDXL Eyes Lora Loader",
        "strength_model": 1.0,
        "detailer_node": "237:236",
        "detector_node": "237:235",
        "prompt_template": "{trigger}, highly detailed eyes, glossy pupils, sharp eyelashes, smooth eyelids, realistic texture, fine iris details, preserve original eyes",
        "prompt_no_trigger": "highly detailed eyes, glossy pupils, sharp eyelashes, smooth eyelids, realistic texture, fine iris details, preserve original eyes",
    },
    "hands": {
        "lora_node": "261",
        "prompt_node": "282",
        "bypass_node": "317",
        "bypass_key": "Enable \U0001f9e9  SDXL Hands Lora Loader",
        "strength_model": 0.7,
        "detailer_node": "243:241",
        "detector_node": "243:242",
        "prompt_template": "{trigger}, realistic hands, natural finger proportions, detailed fingernails, soft skin texture, natural lighting, sharp focus, subtle veins, elegant posture",
        "prompt_no_trigger": "realistic hands, natural finger proportions, detailed fingernails, soft skin texture, natural lighting, sharp focus, subtle veins, elegant posture",
    },
    "foot": {
        "lora_node": None,  # No LoRA for foot — uses base model
        "prompt_node": "295",
        "bypass_node": "318",
        "bypass_key": "Enable \U0001f9e9  SDXL Foot Lora Loader",
        "strength_model": None,
        "detailer_node": "297:375",
        "detector_node": "297:376",
        "prompt_template": "realistic feet, natural toe proportions, detailed toenails, soft skin texture, subtle veins, natural lighting, sharp focus, elegant posture",
        "prompt_no_trigger": "realistic feet, natural toe proportions, detailed toenails, soft skin texture, subtle veins, natural lighting, sharp focus, elegant posture",
    },
    "nipples": {
        "lora_node": None,  # Uses SDXL model (lustify)
        "prompt_node": "300",
        "bypass_node": "319",
        "bypass_key": "Enable \U0001f9e9  SDXL Nipples Lora Loader",
        "strength_model": None,
        "detailer_node": "298:377",
        "detector_node": "298:378",
        "prompt_template": "realistic small breasts, small nipples, natural skin texture, detailed nipples, subtle shading, soft transitions, sharp focus, photorealistic, elegant anatomy",
        "prompt_no_trigger": "realistic small breasts, small nipples, natural skin texture, detailed nipples, subtle shading, soft transitions, sharp focus, photorealistic, elegant anatomy",
    },
    "pussy": {
        "lora_node": None,  # Uses SDXL model (lustify)
        "prompt_node": "302",
        "bypass_node": "320",
        "bypass_key": "Enable \U0001f9e9  SDXL Pussy Lora Loader",
        "strength_model": None,
        "detailer_node": "299:379",
        "detector_node": "299:380",
        "prompt_template": "shaved pussy, realistic small shaved vagina, subtle labia, natural clitoral hood, innie anatomy, soft skin texture, slight gloss, photorealistic, sharp focus, elegant details",
        "prompt_no_trigger": "shaved pussy, realistic small shaved vagina, subtle labia, natural clitoral hood, innie anatomy, soft skin texture, slight gloss, photorealistic, sharp focus, elegant details",
    },
    "penis": {
        "lora_node": None,
        "prompt_node": "327",
        "bypass_node": "325",
        "bypass_key": "Enable \U0001f9e9  SDXL Penis Lora Loader",
        "strength_model": None,
        "detailer_node": "329:381",
        "detector_node": "329:382",
        "prompt_template": "realistic penis, ultra-detailed anatomy, photorealistic skin texture, natural veins, realistic skin folds, subtle moisture, sharp focus on details, elegant proportions, soft shadows, realistic glans texture",
        "prompt_no_trigger": "realistic penis, ultra-detailed anatomy, photorealistic skin texture, natural veins, realistic skin folds, subtle moisture, sharp focus on details, elegant proportions, soft shadows, realistic glans texture",
    },
}

# Fallback LoRA when no character is detected
_FALLBACK_LORA = "nicegirls_Zimage.safetensors"

# Checkpoint mapping
_CHECKPOINTS = {
    "sfw": "zImageTurbo_turbo.safetensors",
    "nsfw": "pornmasterZImage_turboV2.safetensors",
}


def _load_template() -> dict:
    """Load and cache the Final T2I workflow template."""
    global _T2I_TEMPLATE
    if _T2I_TEMPLATE is None:
        with open(_T2I_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            _T2I_TEMPLATE = json.load(f)
        logger.info("T2I template loaded (%d nodes)", len(_T2I_TEMPLATE))
    return _T2I_TEMPLATE


def build_t2i_workflow(
    prompt: str,
    negative: str = "",
    aspect_ratio: str = "5:7 (Balanced Portrait)",
    nsfw: bool = False,
    redefine: bool = False,
    upscale: bool = False,
    detailers: dict | None = None,
    character_loras: list[dict] | None = None,
) -> dict:
    """Build Final T2I workflow by deep-copying template and patching.

    Args:
        prompt: User positive prompt
        negative: Negative prompt (optional)
        aspect_ratio: FluxResolutionNode aspect ratio string
        nsfw: True = pornmaster checkpoint, False = zImageTurbo
        redefine: Enable SDXL Pre-Definer refinement pass
        upscale: Enable SeedVR2 upscale
        detailers: Dict of zone_name -> bool (e.g. {"face": True, "eyes": True, ...})
        character_loras: List from detect_character_loras() or None
    """
    if detailers is None:
        detailers = {
            "face": True, "eyes": True, "hands": True,
            "foot": False, "nipples": False, "pussy": False, "penis": False,
        }

    template = _load_template()
    wf = copy.deepcopy(template)

    # ── Remove preview/cache/comparer nodes ──
    for nid in _REMOVE_NODES:
        wf.pop(nid, None)

    # ── Fix dangling references after removing cache nodes ──
    # 267 (Upscale 2x): was ← 180 (clearCache), now ← 8 (VAEDecode)
    if "267" in wf:
        wf["267"]["inputs"]["image"] = ["8", 0]

    # Post-processing chain: Denoiser → CameraForensic → CRT → (then upscale or save)
    # CameraForensic (373) reads from denoiser output (370)
    if "373" in wf:
        wf["373"]["inputs"]["image"] = ["370", 0]
    # CRT (374) reads from CameraForensic (373) — this is the original connection

    # ── Checkpoint (SFW / NSFW) ──
    checkpoint_name = _CHECKPOINTS["nsfw"] if nsfw else _CHECKPOINTS["sfw"]
    wf["16"]["inputs"]["unet_name"] = checkpoint_name

    # ── Core prompts ──
    wf["175"]["inputs"]["text"] = prompt
    if "178" in wf:
        wf["178"]["inputs"]["text"] = prompt
    if negative:
        wf["104"]["inputs"]["text"] = negative

    # ── Seeds (random) ──
    wf["391"]["inputs"]["seed"] = random.randint(0, 2**50)
    wf["394"]["inputs"]["seed"] = random.randint(0, 2**50)

    # ── Resolution ──
    wf["28"]["inputs"]["aspect_ratio"] = aspect_ratio

    # ── Character LoRA logic ──
    has_character = bool(character_loras)
    if has_character:
        char = character_loras[0]
        trigger = char["trigger"]
        lora_name = char["lora_name"]
        lora_strength_p1 = 0.8
        lora_strength_p2 = 1.0
    else:
        trigger = ""
        lora_name = _FALLBACK_LORA  # still need a valid filename for the node
        lora_strength_p1 = 0.0  # effectively disabled — pure base model
        lora_strength_p2 = 0.0

    # Pass 1 LoRAs: node 387 (character) + 388 (RealisticSnapshot)
    wf["387"]["inputs"]["lora_name"] = lora_name
    wf["387"]["inputs"]["strength_model"] = lora_strength_p1

    # RealisticSnapshot LoRA: stronger when no character LoRA for max realism
    wf["388"]["inputs"]["strength_model"] = 0.4 if has_character else 0.7

    # Pass 2 LoRA: node 400 (character only)
    wf["400"]["inputs"]["lora_name"] = lora_name
    wf["400"]["inputs"]["strength_model"] = lora_strength_p2

    # ── SDXL Pre-Definer toggle ──
    if not redefine:
        # Remove Pre-Definer nodes AND the 2x upscale (only needed for Pre-Definer)
        for nid in _PREDEFINER_NODES:
            wf.pop(nid, None)
        wf.pop("267", None)  # Upscale 2x — not needed without Pre-Definer

    # ── Detailer toggles ──
    # Build the sequential detailer chain, skipping disabled ones
    # Full order: face → eyes → hands → foot → nipples → pussy → penis
    zone_order = ["face", "eyes", "hands", "foot", "nipples", "pussy", "penis"]

    # Determine chain source:
    #   redefine=True  → 323 (Face Preservation Switch, after Pre-Definer)
    #   redefine=False → 8   (VAEDecode, base resolution — no 2x upscale)
    if redefine:
        chain_source = "323"
    else:
        chain_source = "8"

    prev_output = chain_source
    active_zones = []

    for zone_name in zone_order:
        zone_cfg = _DETAILER_ZONES[zone_name]
        enabled = detailers.get(zone_name, False)
        detailer_node = zone_cfg["detailer_node"]
        detector_node = zone_cfg["detector_node"]
        bypass_node = zone_cfg["bypass_node"]
        bypass_key = zone_cfg["bypass_key"]
        lora_node = zone_cfg.get("lora_node")
        prompt_node = zone_cfg["prompt_node"]

        if not enabled:
            # Disable via bypass and remove detailer nodes
            if bypass_node in wf:
                wf[bypass_node]["inputs"][bypass_key] = False
            # Remove the detailer + detector nodes to save VRAM
            wf.pop(detailer_node, None)
            wf.pop(detector_node, None)
            continue

        # Enable via bypass
        if bypass_node in wf:
            wf[bypass_node]["inputs"][bypass_key] = True

        # Rewire image input to previous output in chain
        if detailer_node in wf:
            wf[detailer_node]["inputs"]["image"] = [prev_output, 0]
            prev_output = detailer_node

        # Set LoRA for zones that have one (face, eyes, hands)
        if lora_node and lora_node in wf:
            wf[lora_node]["inputs"]["lora_name"] = lora_name
            wf[lora_node]["inputs"]["strength_model"] = zone_cfg["strength_model"]

        # Set prompt
        if prompt_node in wf:
            if has_character and trigger:
                wf[prompt_node]["inputs"]["text"] = zone_cfg["prompt_template"].replace("{trigger}", trigger)
            else:
                wf[prompt_node]["inputs"]["text"] = zone_cfg["prompt_no_trigger"]

        active_zones.append(zone_name)

    # ── Rewire denoiser input to last active detailer ──
    if "370" in wf:
        wf["370"]["inputs"]["image"] = [prev_output, 0]

    # ── Upscale toggle ──
    # Post-processing (CameraForensic → CRT) always runs on base resolution.
    # Upscale (SeedVR2) runs AFTER post-processing.
    if upscale:
        # Chain: Denoiser(370) → CameraForensic(373) → CRT(374) → Downscale(91) → SeedVR2(94) → Save(100)
        if "91" in wf:
            wf["91"]["inputs"]["image"] = ["374", 0]  # Downscale reads from CRT
        if "100" in wf:
            wf["100"]["inputs"]["images"] = ["94", 0]  # Save reads from SeedVR2
    else:
        # No upscale: Chain: Denoiser(370) → CameraForensic(373) → CRT(374) → Save(100)
        for nid in _UPSCALE_NODES:
            wf.pop(nid, None)
        # CRT(374) already reads from CameraForensic(373) which reads from Denoiser(370)
        # Save(100) already reads from CRT(374) — no rewire needed

    # ── Check if SDXL model is needed at all ──
    # Only Pre-Definer + nipples/pussy/penis use SDXL (176).
    # Face/eyes/hands/foot use ZIT model (16) with LoRAs.
    sdxl_needed = redefine or detailers.get("nipples", False) or detailers.get("pussy", False) or detailers.get("penis", False)
    if not sdxl_needed:
        # Remove SDXL model nodes to save VRAM
        for nid in _SDXL_NODES:
            wf.pop(nid, None)
        # Also remove SDXL conditioning combines
        for nid in ["290", "291", "292", "293"]:
            wf.pop(nid, None)

    logger.info(
        "T2I workflow built: checkpoint=%s, aspect=%s, char=%s, lora=%s, "
        "redefine=%s, upscale=%s, detailers=%s",
        "NSFW" if nsfw else "SFW", aspect_ratio,
        trigger if has_character else "(none)", lora_name,
        redefine, upscale, active_zones,
    )
    return wf


async def run_t2i_generate(
    prompt: str,
    negative: str = "",
    aspect_ratio: str = "5:7 (Balanced Portrait)",
    nsfw: bool = False,
    redefine: bool = False,
    upscale: bool = False,
    detailers: dict | None = None,
) -> bytes | None:
    """Generate image via Final T2I pipeline on dedicated endpoint.

    Auto-detects character LoRAs from prompt, patches workflow,
    submits to RunPod, polls for result.
    """
    # Auto-detect character LoRAs
    char_loras = detect_character_loras(prompt)

    # Build workflow
    workflow = build_t2i_workflow(
        prompt=prompt,
        negative=negative,
        aspect_ratio=aspect_ratio,
        nsfw=nsfw,
        redefine=redefine,
        upscale=upscale,
        detailers=detailers,
        character_loras=char_loras,
    )

    logger.info(
        "T2I job: aspect=%s, nsfw=%s, redefine=%s, upscale=%s, chars=%s",
        aspect_ratio, nsfw, redefine, upscale,
        [c["trigger"] for c in char_loras] if char_loras else "(none)",
    )

    # Submit to V9 endpoint (same RunPod endpoint, different workflow)
    job_id = await submit_job(workflow, images=[], endpoint_id=config.RUNPOD_V9_ENDPOINT_ID)
    if not job_id:
        return None

    # Poll — timeout depends on upscale
    timeout = 900 if upscale else 600
    output = await poll_job(job_id, timeout=timeout, endpoint_id=config.RUNPOD_V9_ENDPOINT_ID)
    if not output:
        return None

    try:
        result = _extract_file_from_output(output)
        if not result:
            logger.error("Could not extract image from T2I output")
            return None
        result = _strip_image_metadata(result)
        return result
    except Exception as e:
        logger.error("Failed to parse T2I result: %s", e)
        return None
