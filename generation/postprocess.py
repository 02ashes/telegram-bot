"""Post-processing utilities — skin enhance, film grain, metadata stripping, silent audio."""

import base64
import io
import logging
import os

from PIL import Image

logger = logging.getLogger(__name__)


def _image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _image_to_png_b64(image_bytes: bytes) -> str:
    """Open image bytes, convert to RGB PNG, return base64 string."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def add_skin_enhance_and_grain(
    workflow: dict,
    image_source_ref: list,
    save_node_id: str,
    skin_model: str = "1xSkinContrast-High-SuperUltraCompact.pth",
    blend_factor: float = 0.45,
    grain_intensity: float = 0.015,
    grain_softness: float = 0.4,
    grain_shadow: float = 0.12,
) -> None:
    """Inject Skin Enhance + Film Grain post-processing into a workflow.

    Adds 4 nodes between the image source and SaveImage:
      UpscaleModelLoader(1x skin model)
      → ImageUpscaleWithModel (enhances skin detail)
      → ImageBlend (blends enhanced with original)
      → FilmGrain (adds film grain for realism)
      → SaveImage

    Args:
        workflow: The workflow dict to modify in-place.
        image_source_ref: Reference to the image source node, e.g. ["77", 0].
        save_node_id: The SaveImage node ID to re-target.
        skin_model: Filename of the 1x upscale model for skin detail.
        blend_factor: ImageBlend factor (0=original, 1=fully enhanced).
        grain_intensity: Film grain intensity (0.01-0.05 range).
        grain_softness: Film grain softness.
        grain_shadow: Film grain shadow amount.
    """
    # Node IDs 500-503 reserved for skin enhance + grain pipeline
    workflow["500"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": skin_model},
    }
    workflow["501"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": ["500", 0],
            "image": image_source_ref,
        },
    }
    workflow["502"] = {
        "class_type": "ImageBlend",
        "inputs": {
            "image1": image_source_ref,
            "image2": ["501", 0],
            "blend_factor": blend_factor,
            "blend_mode": "normal",
        },
    }
    workflow["503"] = {
        "class_type": "FilmGrain",
        "inputs": {
            "image": ["502", 0],
            "intensity": grain_intensity,
            "softness": grain_softness,
            "shadow": grain_shadow,
            "mid": 0.5,
            "scale": 1,
            "grain_type": "Color",
            "mode": "Color",
            "grain_saturation": 1.0,
            "brightness_impact": 0.0,
            "image_saturation": 1.0,
            "grain_size": 1.0,
        },
    }
    # Re-target SaveImage to use FilmGrain output
    workflow[save_node_id]["inputs"]["images"] = ["503", 0]


def strip_image_metadata(image_bytes: bytes) -> bytes:
    """Strip EXIF/metadata from image to prevent workflow leakage.

    Returns clean PNG bytes with no embedded metadata.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Create new image without metadata
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        buf = io.BytesIO()
        clean.save(buf, format="PNG", optimize=True)
        logger.info("Stripped metadata: %d → %d bytes", len(image_bytes), buf.tell())
        return buf.getvalue()
    except Exception as e:
        logger.warning("Failed to strip metadata: %s — returning original", e)
        return image_bytes


def add_silent_audio(video_bytes: bytes) -> bytes:
    """Add a silent audio track to MP4 so Telegram treats it as video, not GIF.
    
    Uses ffmpeg subprocess. Returns original bytes if ffmpeg is unavailable.
    """
    import subprocess
    import tempfile
    import shutil

    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found — skipping silent audio injection")
        return video_bytes

    tmp_in_path = None
    tmp_out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
            tmp_in.write(video_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path.replace(".mp4", "_audio.mp4")

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                tmp_out_path,
            ],
            capture_output=True,
            timeout=30,
        )

        if result.returncode == 0 and os.path.exists(tmp_out_path):
            with open(tmp_out_path, "rb") as f:
                out_bytes = f.read()
            logger.info("Silent audio added: %d -> %d bytes", len(video_bytes), len(out_bytes))
            return out_bytes
        else:
            logger.warning("ffmpeg silent audio failed: %s", result.stderr[:200])
            return video_bytes

    except Exception as e:
        logger.warning("Silent audio injection error: %s", e)
        return video_bytes
    finally:
        for p in [tmp_in_path, tmp_out_path]:
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass


def extract_file_from_output(output, prefer_video: bool = False) -> bytes | None:
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
