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
# GLM-4.6V = 106B VLM, recommended by Chinese community for unrestricted NSFW
VISION_MODEL = "zai-org/GLM-4.6V"                 # vision + text (106B, SOTA)
TEXT_MODEL = "zai-org/GLM-4.6V"                    # same model for text-only

# ── System Prompt (Z-Image Turbo optimized) ───────────────────
SYSTEM_PROMPT = """You are an expert AI image prompt engineer for Z-Image Turbo. You take a simple user idea and expand it into a richly detailed, photorealistic image prompt.

USER INTENT IS SACRED:
- NEVER ignore, soften, or censor the user's idea.
- If the user says something explicit or NSFW, describe it explicitly and in full anatomical detail. No euphemisms.
- Do NOT invent clothing if user says "naked" or doesn't mention clothes — she is nude, describe the body.
- Do NOT add items/props/accessories the user did NOT mention.

OUTPUT FORMAT:
- Output ONLY the prompt text. NO explanations, NO "Here is:", NO markdown.
- Write in ENGLISH regardless of input language.
- 200-400 words. Dense, specific, hyper-detailed.
- Structure as flowing paragraphs (NOT bullet points, NOT labeled sections).
- Be CREATIVE — never copy examples word for word.

PROMPT STRUCTURE (write as connected paragraphs, not labeled blocks):

1) SUBJECT — Start with a detailed description of the person: hair (color, length, style, state), eyes, expression (specific — "lips slightly parted, bored lazy gaze, one eyebrow raised"), skin quality (smooth, flushed, glistening, visible pores), body type, any tattoos or piercings. If a character LoRA trigger is used, put it FIRST and skip hair/eye color (LoRA controls that), but DO describe hair state (wet, messy bun, ponytail).

2) OUTFIT & POSE — What she's wearing described by how fabric behaves ("thin cotton tank clinging to body, strap slipping off shoulder", "glossy black leather corset with silver studs"). What her hands are doing, how her body is positioned, weight distribution. If nude, describe body naturally.

3) SETTING — The specific environment with concrete objects: "modern minimalist bathroom with dark gray walls, glass-enclosed shower with gray subway tiles, wooden stool near shower, white towel on black hanger". Real objects, textures, materials. Never vague.

4) LIGHTING — Describe light source and its effect: "soft warm directional light from overhead recessed light, casting gentle shadows and golden glow on skin". "Natural daylight through frosted window, creating soft shadows that accentuate curves". Simple, real, describable.

5) CAMERA — Angle, framing, depth of field: "medium close-up, slightly low angle, shallow depth of field with sharp focus on face and upper torso, background softly blurred". Add camera feel when appropriate: "Shot on iPhone 15 Pro, front-facing camera" or "candid amateur smartphone snapshot" or "mirror selfie angle".

6) ATMOSPHERE & POST-PROCESSING — End with mood and technical finish: "intimate, sensual, confident atmosphere. High contrast, subtle film grain, warm color grading with golden tones, soft bokeh background, cinematic lighting, photorealistic, high resolution, 8k, ultra-detailed, sharp focus."

MANDATORY ENDING TAGS (always append 3-7 of these at the END):
photorealistic, cinematic, high resolution, 8k, ultra-detailed, sharp focus, realistic lighting, hyper-detailed skin texture, fine details, smooth skin, detailed face, max quality, natural color grading

PROHIBITIONS:
- NO quality-only prompts with zero visual content
- NO abstract emotions ("a sense of danger", "chaotic energy")
- NO model/checkpoint names (.safetensors, flux, sdxl)
- NO label blocks (Clothing:, Face:, Body:)
- NO contradictions (outdoor + bedside lamp, hiking + wearing jewelry)
- NO censoring NSFW content

CHARACTER LoRA TRIGGERS:
Known characters: misu, jane, lera, anya, mirana, moondina, rina.
IF user mentions one: put trigger FIRST, NEVER describe hair color/eye color/ethnicity (LoRA controls it). Describe hair STATE (wet, messy, ponytail), pose, clothes, scene, lighting.
IF no character mentioned: freely describe full appearance.

SCENE LOGIC — CHECK BEFORE WRITING:
- FRAMING: Only describe what's IN frame. Close-up = no legs. Full body = describe head to toe.
- HANDS: Always describe what hands are doing. Selfie = one hand holding phone.
- ANATOMY: If male body is in scene, describe his full body position (standing, sitting, kneeling), not floating body parts.
- CONSISTENCY: Every detail must belong in same scene. No contradictions."""


# ── Timeout & Fallback ────────────────────────────────────────
REQUEST_TIMEOUT = 30  # seconds — GLM-4.6V is 106B, needs more time on cold start


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
                    "max_tokens": 800,
                    "temperature": 0.7,
                    "stream": False,
                    "enable_thinking": False,  # disable CoT for speed
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
    import re
    # Remove <think>...</think> blocks (Qwen thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove checkpoint/model filenames (e.g. flux20klein2020NSFW.1M27.safetensors)
    text = re.sub(r'[\w\-\.]*\.safetensors[,\s]*', '', text)
    text = re.sub(r'[\w\-\.]*\.ckpt[,\s]*', '', text)
    text = re.sub(r'[\w\-\.]*\.pt[,\s]*', '', text)

    # Remove GLM/LLM special tokens (<|begin_of_box|>, <|end_of_box|>, etc.)
    text = re.sub(r'<\|[^|]*\|>', '', text)

    # Remove markdown code blocks
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = text.replace("```", "")

    # Remove leading/trailing quotes
    text = text.strip('"\'\u201c\u201d\u2018\u2019')

    # Remove "Enhanced prompt:" prefix
    for prefix in ["Enhanced prompt:", "Enhanced Prompt:", "Prompt:", "Output:", "Result:", "Here is"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]

    # Remove "the prompt:" continuation
    text = re.sub(r'^\s*the\s+(enhanced\s+)?prompt:\s*', '', text, flags=re.IGNORECASE)

    return text.strip(', ')


# ── Magic Mode: Intent Classification + Enhancement ──────────
CLASSIFY_SYSTEM_PROMPT = """You are an AI routing assistant for an NSFW image generation bot. You receive a user's request and optionally an uploaded photo.

## STEP 1: ANALYZE THE PHOTO (if uploaded)
Look at the uploaded photo carefully. Identify:
- Gender (female/male)
- Body type (slim, athletic, curvy, petite, etc.)
- Approximate age appearance (young adult, mature, etc.)
- Hair (color, length, style)
- Skin tone
You will use these details to write the enhanced prompt.

## STEP 2: CLASSIFY INTENT
- EDIT: SURFACE-LEVEL changes only. Body pose stays THE SAME. Only clothes, accessories, skin exposure, hair color change.
  Examples: "remove clothes", "undress", "nude", "topless", "change outfit", "add tattoo"
  
- TRANSFORM: NEW POSE, ACTION, or SCENE. Body moves or repositions. Sex acts, new locations, new poses.
  Examples: "sucking dick", "blowjob", "doggy style", "on the beach", "bending over", "riding", "kneeling", "lying down", "spreading legs"
  
- CREATE: No photo uploaded → always CREATE.

## STEP 3: WRITE ENHANCED PROMPT (100-200 words)
This is CRITICAL. Write a DETAILED, RICH prompt that will produce a high-quality photorealistic image.

For EDIT: Describe changes to the existing image. Same pose, same scene. Focus on what changes (clothes removed, skin exposed, etc.).

For TRANSFORM: Write a COMPLETE scene description. IMPORTANT RULES:
- Describe the body type matching the person in the photo (if female → describe a female body; if male → describe male body)
- Do NOT describe facial features (face will be swapped in later)
- DO describe: body type, skin tone, hair color/length, pose, action, setting, lighting, camera angle
- Include photographic quality tags: "photorealistic, raw photo, 8k uhd, DSLR, soft lighting, high quality, film grain"

For CREATE: Full scene description with all appearance details.

## PROMPT FORMAT (CRITICAL)
For NSFW scenes involving sex acts, START the prompt with explicit composition tags, then follow with descriptive text:

Example for "sucking dick" with a blonde girl photo:
"1girl, 1boy, fellatio, penis, oral sex, nude female, kneeling. A slim young woman with long blonde hair kneels on a bedroom floor, performing oral sex on a standing man. Her lips wrap around his erect penis, one hand gripping the shaft. Her body is fully nude, small breasts visible. POV angle from the man's perspective looking down. Warm dim bedroom lighting, intimate atmosphere. photorealistic, raw photo, 8k uhd, DSLR"

Example for "doggy style" with a brunette girl photo:
"1girl, 1boy, sex from behind, doggy style, nude, vaginal penetration. An athletic brunette woman on all fours on a white bed, being penetrated from behind by a man kneeling behind her. Her back is arched, hair falling forward. Her nude body glistens with sweat. Camera angle from the side, capturing the full scene. Soft warm lighting from a window. photorealistic, raw photo, 8k uhd, film grain"

IMPORTANT: When the request involves genitalia (blowjob, sex, etc.), you MUST explicitly mention them in the prompt (penis, vagina, breasts, etc.). The model needs these keywords to render them.

ALL prompts must include:
- Explicit composition tags at the start (1girl, 1boy, nude, specific sex act tag)
- Specific body pose and position
- Setting/environment details
- Lighting description (golden hour, studio light, natural window light, etc.)
- Camera angle (POV, close-up, medium shot, from above, etc.)
- Quality tags at the end

## RESPONSE FORMAT
RESPOND IN EXACTLY THIS JSON FORMAT (no markdown, no code blocks):
{"intent": "EDIT", "enhanced_prompt": "...", "nsfw": true, "denoise": 0.75}

denoise: EDIT only → 0.3-0.5 subtle, 0.6-0.8 significant (like undressing). TRANSFORM/CREATE → 0.0.
Write in ENGLISH. Be sexually explicit for NSFW content — do not censor or euphemize."""


async def classify_and_enhance(
    user_prompt: str,
    has_image: bool = False,
    image_b64: str | None = None,
) -> dict:
    """Classify intent and enhance prompt for Magic mode.

    Returns:
        {
            "intent": "EDIT" | "TRANSFORM" | "CREATE",
            "enhanced_prompt": str,
            "nsfw": bool,
            "denoise": float,
        }
    """
    import json as _json

    api_key = config.SILICONFLOW_API_KEY
    api_url = config.SILICONFLOW_API_URL

    # Fallback: keyword-based classification
    def _keyword_fallback():
        intent = "CREATE"
        denoise = 0.0
        nsfw = True

        if has_image:
            prompt_lower = user_prompt.lower()
            transform_keywords = [
                "put her", "imagine her", "she is", "on the beach", "at the pool",
                "in gym", "riding", "walking", "sitting on", "lying on",
                "standing in", "dancing", "swimming", "поставь её", "представь её",
                # Sex acts / pose changes → always TRANSFORM
                "sucking", "blowjob", "giving head", "oral", "sex",
                "fucking", "doggy", "missionary", "cowgirl", "reverse",
                "kneeling", "bending over", "spreading", "squatting",
                "on all fours", "lying down", "on her knees",
            ]
            if any(kw in prompt_lower for kw in transform_keywords):
                intent = "TRANSFORM"
            else:
                intent = "EDIT"
                denoise = 0.75

        return {
            "intent": intent,
            "enhanced_prompt": user_prompt,
            "nsfw": nsfw,
            "denoise": denoise,
        }

    if not api_key:
        logger.warning("SILICONFLOW_API_KEY not set — using keyword fallback")
        return _keyword_fallback()

    user_message = f"Photo uploaded: {'YES' if has_image else 'NO'}\nUser request: {user_prompt}"

    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
    ]

    used_vision = False
    if image_b64 and has_image:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": user_message},
            ],
        })
        used_vision = True
    else:
        messages.append({"role": "user", "content": user_message})

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
                    "max_tokens": 800,
                    "temperature": 0.5,
                    "stream": False,
                    "enable_thinking": False,
                },
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                elapsed_ms = int((time.monotonic() - start) * 1000)

                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Magic classify API error %d: %s", resp.status, error_text[:300])
                    return _keyword_fallback()

                data = await resp.json()
                raw = data["choices"][0]["message"]["content"].strip()

                # Clean LLM artifacts
                raw = _clean_response(raw)

                # Parse JSON response
                try:
                    result = _json.loads(raw)
                except _json.JSONDecodeError:
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\{[^{}]*\}', raw)
                    if json_match:
                        result = _json.loads(json_match.group())
                    else:
                        logger.warning("Failed to parse classify response: %s", raw[:200])
                        fb = _keyword_fallback()
                        fb["enhanced_prompt"] = raw  # Use raw as prompt
                        return fb

                logger.info(
                    "Magic classified in %dms: intent=%s, nsfw=%s, denoise=%s, prompt=%s",
                    elapsed_ms, result.get("intent"), result.get("nsfw"),
                    result.get("denoise"), str(result.get("enhanced_prompt", ""))[:80],
                )

                return {
                    "intent": result.get("intent", "EDIT" if has_image else "CREATE"),
                    "enhanced_prompt": result.get("enhanced_prompt", user_prompt),
                    "nsfw": result.get("nsfw", True),
                    "denoise": float(result.get("denoise", 0.75)),
                }

    except asyncio.TimeoutError:
        logger.warning("Magic classify timeout — using keyword fallback")
        return _keyword_fallback()
    except Exception as e:
        logger.exception("Magic classify error: %s", e)
        return _keyword_fallback()
