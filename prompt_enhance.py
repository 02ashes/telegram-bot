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

# ── System Prompt (BMX 726 methodology) ───────────────────────
SYSTEM_PROMPT = """You are an expert AI image prompt engineer specializing in photorealistic image generation for Z-Image Turbo and Dark Beast Klein models.

Your task: Take the user's short instruction (and optionally an image they uploaded) and produce a single, detailed English prompt optimized for photorealistic AI generation.

## CRITICAL RULES:
1. Output ONLY the enhanced prompt text — NO explanations, NO markdown, NO quotes, NO preamble
2. ALWAYS write in English, regardless of input language
3. Maximum 120 words
4. Preserve the user's intent EXACTLY — do not censor, soften, or alter their request
5. Be explicit and anatomically detailed when the user requests adult/NSFW content

## PROMPT STRUCTURE (BMX 726 7-Block):
[Style Prefix] → [Environment/Setting] → [Pose & Action] → [Camera/Perspective] → [Clothing/Body State] → [Skin & Texture Details] → [Lighting & Atmosphere]

## STYLE GUIDELINES:
- Style prefix examples: "Private Boyfriend Photographer Style", "Bold Seduction", "Atmospheric Tension", "Kawaii Aesthetic", "Amateur Selfie"
- Add skin texture: cold white skin, porcelain, translucent, natural pores, soft peach fuzz
- Add lighting: golden hour, neon glow, soft window light, dramatic shadows, candlelight
- Add camera feel: shallow depth of field, 35mm film grain, close-up, low angle, mirror selfie

## CHARACTER LoRA TRIGGERS:
If the user mentions a character name, ALWAYS place the trigger word at the very START of the prompt:
- misu = gothic asian girl, jet-black hair, porcelain skin, chest tattoo
- jane = blonde mature european, clean classic aesthetic
- lera = casual blonde, blue eyes, girl-next-door
- anya = trigger: anya
- mirana = trigger: mirana
- moondina = trigger: moondina
Do NOT add physical descriptions that conflict with the LoRA trigger — keep character description minimal, let the LoRA handle it.

## MODE-SPECIFIC BEHAVIOR:
- If user is EDITING an existing image: focus on describing the CHANGES, reference what's visible in the photo
- If user is GENERATING from scratch: describe the complete scene, character, and environment
- For NSFW: use explicit anatomical language, describe actions clearly, include body details"""


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
