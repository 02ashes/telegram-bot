"""
Auto-Prompt Enhancement via SiliconFlow API (Qwen2.5-VL-72B-Instruct).

Analyzes uploaded images and rewrites short user prompts into detailed,
photorealistic descriptions optimized for Z-Image Turbo / Dark Beast Klein.

API: OpenAI-compatible (SiliconFlow), supports Vision input.
"""

import asyncio
import base64
import logging
import time

import aiohttp

import config

logger = logging.getLogger(__name__)

# ── Models ────────────────────────────────────────────────────
VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
TEXT_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"  # same model handles both

# ── System Prompt (based on proven Z-Turbo methodology) ───────
SYSTEM_PROMPT = """You are an expert AI image prompt engineer specialized in photorealistic amateur-style photography, optimized for Flux-based AI image generation. Your mission is to take a simple idea from the user and translate it into a single perfect dense prompt paragraph.

═══════════════════════════════════════════════════════════════
USER IDEA — ABSOLUTE PRIORITY
═══════════════════════════════════════════════════════════════
The user's idea is the foundation of everything.
- NEVER ignore, simplify, or water down what the user describes.
- NEVER replace the user's idea with something generically "better" or "safer".
- If the user specifies a location, pose, expression, object, or action — it MUST appear in the final prompt, developed and detailed.
- SFW or NSFW: handle whatever the user requests without restrictions.

═══════════════════════════════════════════════════════════════
CRITICAL OUTPUT RULES
═══════════════════════════════════════════════════════════════
- Output ONLY the final prompt text. Nothing else.
- NO explanations, NO preamble, NO "Here is the prompt:", NO markdown, NO code blocks.
- ONE single dense paragraph of comma-separated descriptors and short natural phrases.
- Length: 80-150 words. Dense and specific, never padded or vague.
- NEVER use structured label blocks like "Clothing:", "Face:", "Body:" — these waste tokens.
- NEVER use generic quality tags: "masterpiece", "best quality", "8k", "ultra detailed", "cinematic", "stunning", "perfect".
- NEVER describe professional camera gear (Canon EOS, 85mm lens, etc.).
- NEVER include model names, checkpoint filenames, file extensions (.safetensors, .ckpt), or technical AI metadata.

═══════════════════════════════════════════════════════════════
SCENE THINKING — THINK BEFORE YOU WRITE
═══════════════════════════════════════════════════════════════
Think through ALL of this before writing:

LOCATION: Where is she? Home, bar, gym, car, mountains, bathroom, bedroom? What time of day? What is she doing?

FRAMING: Face only → describe only face. Face and bust → shoulders up. Full body → full outfit top to bottom. What angle? High, low, turned slightly, from behind, lying down?

CLOTHING: Does the outfit make sense for this situation? Describe how fabric BEHAVES, not brand names: "stretched grey cotton tee creating tension lines across chest", "loose linen shirt draping off one shoulder", "denim creasing at the hips".

EXPRESSION: ALWAYS specific. "Half-lidded eyes, lips barely parted", "genuine crooked smile", "bored flat expression" — NEVER "beautiful expression" or "pretty smile".

LIGHTING: ALWAYS name the actual source. "Window light casting a hard diagonal shadow", "warm bedside lamp from the right", "harsh cold bathroom fluorescent overhead". NEVER "cinematic lighting" or "perfect lighting".

═══════════════════════════════════════════════════════════════
CHARACTER LoRA TRIGGERS
═══════════════════════════════════════════════════════════════
If the user mentions a character name, place the trigger word at the START:
- misu, jane, lera, anya, mirana, moondina
Do NOT add physical descriptions that conflict with the LoRA — keep character description minimal, let the LoRA handle appearance.

═══════════════════════════════════════════════════════════════
PROMPT STRUCTURE — follow this order
═══════════════════════════════════════════════════════════════
[LoRA trigger if any] → [scene / activity / location] → [physical description or "let LoRA handle"] → [clothing with fabric behavior] → [pose and body position] → [specific expression and gaze] → [hand detail if visible] → [lighting source described concretely] → [camera: smartphone, amateur grain, slight front-camera distortion, no flash]

═══════════════════════════════════════════════════════════════
ABSOLUTE PROHIBITIONS
═══════════════════════════════════════════════════════════════
- NO quality booster tags (masterpiece, best quality, 8k, ultra detailed, cinematic, perfect, stunning)
- NO professional camera specs (Canon, 85mm, studio lighting)
- NO label blocks (Clothing:, Face:, Body:, Background:)
- NO model names or checkpoint filenames (flux, klein, darkbeast, sdxl, safetensors)
- NO contradictory or nonsensical scene descriptions
- NO arrows (→) in the final output — use commas only"""


# ── Timeout & Fallback ────────────────────────────────────────
REQUEST_TIMEOUT = 15  # seconds — if API doesn't respond, return original prompt


async def enhance_prompt(
    user_prompt: str,
    image_b64: str | None = None,
    mode: str = "edit",
    lora_trigger: str = "",
) -> dict:
    """Enhance a user prompt via SiliconFlow Vision LLM.

    Args:
        user_prompt: The raw prompt from the user (any language).
        image_b64: Optional base64-encoded image (JPEG/PNG) for vision analysis.
        mode: "edit", "generate", or "dark" — adjusts enhancement style.
        lora_trigger: Character LoRA trigger word (e.g. "misu", "jane").

    Returns:
        {
            "enhanced": str,   # The improved prompt (or original on failure)
            "original": str,   # The original user prompt
            "model": str,      # Model used
            "time_ms": int,    # Processing time in ms
            "used_vision": bool,
        }
    """
    api_key = config.SILICONFLOW_API_KEY
    api_url = config.SILICONFLOW_API_URL

    if not api_key:
        logger.warning("SILICONFLOW_API_KEY not set — skipping prompt enhancement")
        return _fallback(user_prompt)

    # Build context instruction
    context_parts = []
    if mode == "generate":
        context_parts.append("Mode: TEXT-TO-IMAGE generation (no source image, create complete scene)")
    elif mode == "dark":
        context_parts.append("Mode: DARK BEAST edit (NSFW edits, be explicit)")
    else:
        context_parts.append("Mode: IMAGE EDITING (describe changes to the uploaded photo)")

    if lora_trigger:
        context_parts.append(f"Character LoRA active: '{lora_trigger}' — MUST start prompt with this trigger word")

    context = "\n".join(context_parts)
    user_message = f"{context}\n\nUser's request: {user_prompt}"

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if image_b64 and api_key:
        # Vision request: send image + text
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": user_message,
                },
            ],
        })
        used_vision = True
    else:
        # Text-only request
        messages.append({
            "role": "user",
            "content": user_message,
        })
        used_vision = False

    # Call API
    start = time.monotonic()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": VISION_MODEL if used_vision else TEXT_MODEL,
                    "messages": messages,
                    "max_tokens": 400,
                    "temperature": 0.7,
                    "stream": False,
                },
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                elapsed_ms = int((time.monotonic() - start) * 1000)

                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "SiliconFlow API error %d: %s", resp.status, error_text[:300]
                    )
                    return _fallback(user_prompt, elapsed_ms)

                data = await resp.json()
                enhanced = data["choices"][0]["message"]["content"].strip()

                # Clean up: remove markdown quotes, thinking tags, etc.
                enhanced = _clean_response(enhanced)

                logger.info(
                    "Prompt enhanced in %dms (vision=%s): '%s' → '%s'",
                    elapsed_ms, used_vision, user_prompt[:50], enhanced[:80],
                )

                return {
                    "enhanced": enhanced,
                    "original": user_prompt,
                    "model": VISION_MODEL if used_vision else TEXT_MODEL,
                    "time_ms": elapsed_ms,
                    "used_vision": used_vision,
                }

    except asyncio.TimeoutError:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.warning("SiliconFlow API timeout after %dms — using original prompt", elapsed_ms)
        return _fallback(user_prompt, elapsed_ms)
    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.exception("Prompt enhance error: %s", e)
        return _fallback(user_prompt, elapsed_ms)


def _fallback(user_prompt: str, time_ms: int = 0) -> dict:
    """Return original prompt when enhancement fails."""
    return {
        "enhanced": user_prompt,
        "original": user_prompt,
        "model": "fallback",
        "time_ms": time_ms,
        "used_vision": False,
    }


def _clean_response(text: str) -> str:
    """Remove common LLM artifacts from the response."""
    # Remove <think>...</think> blocks (Qwen thinking mode)
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove markdown code blocks
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = text.replace("```", "")

    # Remove leading/trailing quotes
    text = text.strip('"\'""''')

    # Remove "Enhanced prompt:" prefix
    for prefix in ["Enhanced prompt:", "Enhanced Prompt:", "Prompt:", "Output:", "Result:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]

    return text.strip()
