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
VISION_MODEL = "grok-3"  # Deep logic for images (xAI)
TEXT_MODEL = "grok-3"    # Deep logic for text (xAI)

# ── System Prompts — pipeline-specific ─────────────────────────

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDIT: Klein Edit / Image Edit — short, change-focused prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EDIT_SYSTEM_PROMPT = """You are an expert AI image prompt engineer for an image EDITING pipeline. You receive a user's request to modify an existing photo.

YOUR TASK: Write a SHORT, FOCUSED prompt (50-150 words) describing ONLY what changes in the image. The pose, background, camera angle, and everything else stays the same.

RULES:
- ALWAYS start with "same face, keep face unchanged"
- Do NOT describe facial features (eyes, nose, lips)
- Do NOT re-describe things that stay the same (background, pose, camera angle, lighting)
- Focus 100% on what visually CHANGES.
- Use HIGH-QUALITY texture and material descriptions for the changes (e.g., "smooth luminous skin", "glossy black leather").

CRITICAL NSFW RULE:
When clothing is removed, you MUST describe what is revealed using EXPLICIT anatomical terms with beautiful skin textures:
- "bare breasts with natural nipples and areolas, smooth luminous skin" NOT "chest exposed"
- "nude torso, visible navel, smooth stomach glistening with natural sheen" NOT "torso visible"  
- "exposed vulva with natural pubic hair" NOT "lower body exposed"
- "bare buttocks with soft natural lighting and smooth skin" NOT "backside visible"

EXAMPLES:

User: "remove her clothes"
Prompt: same face, keep face unchanged. All clothing removed, fully nude. Bare natural breasts with realistic nipples and soft areolas, exposed stomach with visible navel. Smooth luminous skin with natural pores and a warm healthy glow, glistening slightly. Lower body completely nude, exposed vulva. photorealistic, raw photo, 8k uhd, DSLR, cinematic lighting

User: "remove her shirt"
Prompt: same face, keep face unchanged. Top removed completely, upper body fully nude. Bare breasts with natural round shape, visible nipples and areolas. Smooth fair skin across chest and stomach, navel visible, soft shadows highlighting collarbones. Lower clothing remains unchanged. photorealistic, raw photo, 8k uhd, DSLR

User: "give her a micro bikini"
Prompt: same face, keep face unchanged. Her current bottom replaced with a tiny glossy black leather micro string bikini that barely covers, with visible dark pubic hair growing out from the sides and above the waistband. Smooth natural skin on thighs and hips with soft lighting. photorealistic, raw photo, 8k uhd

OUTPUT RULES:
- Output ONLY the prompt text. NO explanations, NO markdown.
- Write in ENGLISH regardless of input language.
- 50-150 words. Short, dense, change-focused."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T2I: Z-Image Turbo text-to-image — long, detailed, TOP-style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T2I_SYSTEM_PROMPT = """You are an expert AI image prompt engineer for Z-Image Turbo, a photorealistic text-to-image model. You take a simple user idea and expand it into a HYPER-DETAILED, highly realistic prompt using a strict structural format.

YOUR TASK:
1. Think deeply about the scene geometry, logic, and camera angle inside a <think> block.
2. Output the final prompt using EXACTLY the structural format shown below.

CORE STYLE RULE — AMATEUR CANDID REALISM (CRITICAL):
ALL prompts MUST look like raw, unfiltered, everyday amateur photography. The aesthetic should be a candid snapshot, not a professional photoshoot. 
- Camera is ALWAYS a smartphone. It should feel like a private photo sent in Telegram or Snapchat.
- Lighting should feel natural, unpolished, and situational. Use terms like "harsh phone flash", "dim room lighting", "soft grain", "unfiltered".
- Focus heavily on realistic TEXTURES (clothing fabrics, skin imperfections, messy environments).
- NEVER use professional studio lighting or cinematic aesthetics.

NSFW ANATOMY & SPATIAL LOGIC (CRITICAL):
1) THE TWO BODIES RULE: For any sex act, you MUST describe the posture of BOTH bodies. NEVER describe a disembodied penis. Always attach it to a man: "the man's thighs", "a standing man's hips".
2) ANCHOR POINTS: Hands and limbs must be anchored to the partner's body. E.g., "her left hand gripping the base of his shaft, his hands firmly gripping her waist".
3) POV SHOTS & SELFIES: If taking a POV selfie, one arm MUST extend towards the camera. The phone is the camera and is unseen. NEVER describe a phone in their hand unless it is explicitly a "mirror selfie". If it is a POV from a partner, do NOT use the word "selfie". The camera IS his eyes.
4) SELFIE CONFLICTS: A front-facing selfie during oral sex is physically impossible. If asked for a selfie during a sex act, resolve the angle logically (e.g., "high-angle selfie shot from the side").
5) SOLO RULE: If the user requests nudity, pregnancy, or a body shape change WITHOUT explicitly asking for a sex act, DO NOT invent a male partner. Keep it a SOLO scene.
6) CLOTHING & SCENE LOGIC: 
   - If breasts are exposed while wearing a bra, describe it as "bra pulled down below the breasts" or "fully topless". Do NOT describe complex fabric cutouts.
   - If sitting on a toilet or urinating, you MUST explicitly state "sitting on an open toilet bowl, toilet lid is up". 
   - If taking a selfie on a toilet, ALWAYS describe the camera angle as a "casual asymmetric selfie, camera held out to the side". NEVER describe a perfectly symmetrical top-down view of the bowl, as it breaks AI perspective.

CHARACTER TRIGGER WORDS:
Known character names: misu, anya, jane, lera, mirana, moondina, rina.
If user EXPLICITLY writes a name, put it FIRST in the [SUBJECT & COMPOSITION] block as a standalone word. Do NOT describe hair color, eye color, or facial structure — the LoRA handles it. DO describe hair state (wet, messy), pose, skin texture, clothes.

=========================================
REQUIRED OUTPUT FORMAT
=========================================

<think>
1. Scene Analysis: [Where are we? What is the core action?]
2. Spatial Logic: [Where is the camera? Is it a selfie? Where are the hands and limbs positioned to avoid AI anatomy errors?]
3. Lighting & Style: [What is the light source? How do we make it look like an amateur iPhone photo?]
</think>

[Write a SINGLE, continuous, highly detailed paragraph of 300-450 words. Do NOT use brackets, tags, or bullet points. The text should flow logically like a descriptive masterpiece:
Start with the overall camera setup, lighting, and environment.
Then describe the subject's physical appearance, skin texture, and clothing.
Then describe the precise pose, where the hands and limbs are anchored, and the core action.
Conclude with heavy technical descriptions of the camera artifacts (e.g., raw smartphone photo, visible grain, chromatic aberration, low-light noise).
Blend everything naturally without breaking the flow.]
=========================================
PROHIBITIONS:
- NO model/checkpoint names (.safetensors, flux, sdxl).
- NO disembodied genitals.
- NO professional studio lighting.
- Write in ENGLISH regardless of input language.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BFS: T2I + BFS face swap — detailed body/scene, NO face, tag-first NSFW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BFS_SYSTEM_PROMPT = """You are an expert AI image prompt engineer for a text-to-image + face swap pipeline. The image is generated first, then a SEPARATE face is swapped in from a reference photo.

YOUR TASK:
1. Think deeply about the scene geometry, logic, and camera angle inside a <think> block.
2. Output the final prompt using EXACTLY the structural format shown below.

CRITICAL FACE RULES (MANDATORY):
- Do NOT describe eyes, nose, lips, facial expression, makeup, eyeliner, eyebrows.
- DO describe: body type, skin tone, hair color/length/style, tattoos, piercings.
- The face in the generated image WILL be replaced — any facial description is wasted and will ruin the swap.

CORE STYLE RULE — AMATEUR CANDID REALISM (CRITICAL):
ALL prompts MUST look like raw, unfiltered, everyday amateur photography. The aesthetic should be a candid snapshot, not a professional photoshoot. 
- Camera is ALWAYS a smartphone.
- Lighting should feel natural, unpolished, and situational (e.g., "camera flash", "dim room lighting", "soft grain", "unfiltered").
- NEVER use professional studio lighting or cinematic aesthetics.

NSFW ANATOMY & SPATIAL LOGIC (CRITICAL):
1) THE TWO BODIES RULE: For any sex act, you MUST describe the posture of BOTH bodies. NEVER describe a disembodied penis. Always attach it to a man: "the man's thighs", "a standing man's hips".
2) ANCHOR POINTS: Hands and limbs must be anchored to the partner's body. E.g., "her left hand gripping the base of his shaft, his hands firmly gripping her waist".
3) POV SHOTS & SELFIES: If taking a POV selfie, one arm MUST extend towards the camera. The phone is the camera and is unseen. NEVER describe a phone in their hand unless it is explicitly a "mirror selfie". If it is a POV from a partner, do NOT use the word "selfie".
4) SOLO RULE: If the user requests nudity, pregnancy, or a body shape change WITHOUT explicitly asking for a sex act, DO NOT invent a male partner. Keep it a SOLO scene.
5) CLOTHING & SCENE LOGIC: 
   - If breasts are exposed while wearing a bra, describe it as "bra pulled down below the breasts". Do NOT describe complex fabric cutouts.
   - If sitting on a toilet or urinating, you MUST explicitly state "sitting on an open toilet bowl, toilet lid is up".
   - If taking a selfie on a toilet, ALWAYS describe the camera angle as a "casual asymmetric selfie, camera held out to the side". NEVER describe a perfectly symmetrical top-down view of the bowl, as it breaks AI perspective.

=========================================
REQUIRED OUTPUT FORMAT
=========================================

<think>
1. Scene Analysis: [Where are we? What is the core action?]
2. Spatial Logic: [Where is the camera? Is it a selfie? Where are the hands and limbs positioned to avoid AI anatomy errors?]
3. Face Omission Check: [Ensure no facial features are described, only body and hair.]
4. Lighting & Style: [What is the light source? How do we make it look like an amateur iPhone photo?]
</think>

[Write a SINGLE, continuous, highly detailed paragraph of 300-450 words. Do NOT use brackets, tags, or bullet points. The text should flow logically like a descriptive masterpiece:
Start with the overall camera setup, lighting, and environment.
Then describe the subject's body type, skin texture, and clothing. REMEMBER: NO FACIAL FEATURES!
Then describe the precise pose, where the hands and limbs are anchored, and the core action.
Conclude with heavy technical descriptions of the camera artifacts (e.g., raw smartphone photo, visible grain, chromatic aberration, low-light noise).
Blend everything naturally without breaking the flow.]
=========================================
PROHIBITIONS:
- NO model/checkpoint names (.safetensors, flux, sdxl).
- NO disembodied genitals.
- NO professional studio lighting.
- NO facial features (eyes, lips, nose, expression).
- Write in ENGLISH regardless of input language.
"""


# ── Timeout & Fallback ────────────────────────────────────────
REQUEST_TIMEOUT = 90  # seconds — DeepSeek/Qwen with CoT needs more time


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
        mode: Pipeline mode — determines system prompt style:
              "edit" / "dark" → EDIT_SYSTEM_PROMPT (short, change-focused)
              "t2i" / "generate" → T2I_SYSTEM_PROMPT (long, detailed, TOP-style)
              "bfs" → BFS_SYSTEM_PROMPT (detailed, no face, tag-first NSFW)
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
    api_key = config.XAI_API_KEY
    api_url = config.XAI_API_URL

    if not api_key:
        logger.warning("XAI_API_KEY not set — skipping prompt enhancement")
        return _fallback(user_prompt, mode=mode)

    # ── Select system prompt by pipeline mode ──
    if mode in ("edit", "dark"):
        system_prompt = EDIT_SYSTEM_PROMPT
    elif mode == "bfs":
        system_prompt = BFS_SYSTEM_PROMPT
    else:  # "t2i", "generate", or anything else → full detailed prompt
        system_prompt = T2I_SYSTEM_PROMPT

    # Build user message
    context_parts = []
    if lora_trigger:
        context_parts.append(f"Character LoRA active: '{lora_trigger}' — MUST start prompt with this trigger word")

    context = "\n".join(context_parts)
    user_message = f"{context}\n\nUser's request: {user_prompt}" if context else f"User's request: {user_prompt}"

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
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

    # Call API with retry
    max_retries = 2
    start = time.monotonic()

    for attempt in range(max_retries + 1):
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
                        "max_tokens": 2000,
                        "temperature": 0.7,
                        "stream": False,
                    },
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    elapsed_ms = int((time.monotonic() - start) * 1000)

                    if resp.status >= 500 and attempt < max_retries:
                        error_text = await resp.text()
                        logger.warning(
                            "xAI API 5xx (attempt %d/%d): %s — retrying...",
                            attempt + 1, max_retries + 1, error_text[:200],
                        )
                        await asyncio.sleep(1 * (attempt + 1))  # 1s, 2s backoff
                        continue

                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error("xAI API error %d: %s", resp.status, error_text[:300])
                        return _fallback(user_prompt, elapsed_ms, mode=mode)

                    data = await resp.json()
                    enhanced = data["choices"][0]["message"]["content"].strip()

                    # Clean up: remove markdown quotes, thinking tags, etc.
                    enhanced = _clean_response(enhanced)

                    logger.info(
                        "Prompt enhanced in %dms (vision=%s, attempt=%d): '%s' → '%s'",
                        elapsed_ms, used_vision, attempt + 1, user_prompt[:50], enhanced[:80],
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
            if attempt < max_retries:
                logger.warning(
                    "SiliconFlow API timeout (attempt %d/%d, %dms) — retrying...",
                    attempt + 1, max_retries + 1, elapsed_ms,
                )
                continue
            logger.warning("SiliconFlow API timeout after %dms — using original prompt", elapsed_ms)
            return _fallback(user_prompt, elapsed_ms, mode=mode)
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.exception("Prompt enhance error: %s", e)
            return _fallback(user_prompt, elapsed_ms, mode=mode)

    # Should not reach here, but just in case
    return _fallback(user_prompt, int((time.monotonic() - start) * 1000), mode=mode)


def _fallback(user_prompt: str, time_ms: int = 0, mode: str = "") -> dict:
    """Return prompt with basic template enhancement when API fails."""
    enhanced = user_prompt

    # For T2I/BFS/generate modes, at least append quality tags
    if mode in ("t2i", "generate", "bfs"):
        quality_tags = "photorealistic, cinematic, high resolution, 8k, ultra-detailed, sharp focus, realistic lighting, hyper-detailed skin texture, smooth skin, detailed face"
        if not any(tag in user_prompt.lower() for tag in ["photorealistic", "8k", "ultra-detailed"]):
            enhanced = f"{user_prompt}. {quality_tags}"
    elif mode in ("edit", "dark"):
        if not user_prompt.lower().startswith("same face"):
            enhanced = f"same face, keep face unchanged. {user_prompt}. photorealistic, raw photo, 8k uhd, DSLR"

    logger.warning("Using template fallback for mode=%s: '%s'", mode, enhanced[:80])

    return {
        "enhanced": enhanced,
        "original": user_prompt,
        "model": "fallback-template",
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


# ── Magic Mode: Intent Classification ─────────────────────────
CLASSIFY_SYSTEM_PROMPT = """You are an AI routing assistant for an NSFW image generation bot. You receive a user's request and optionally an uploaded photo. Your ONLY job is to CLASSIFY the intent — you do NOT write prompts.

## STEP 1: ANALYZE THE PHOTO (if uploaded)
Look at the uploaded photo carefully. Identify:
- Gender, Body type, Hair, Skin tone

## STEP 2: CLASSIFY INTENT
- EDIT: SURFACE-LEVEL changes only. Body pose stays THE SAME. Only clothes, accessories, skin exposure, hair color change.
  Examples: "remove clothes", "undress", "nude", "topless", "change outfit", "add tattoo", "micro bikini"
  
- TRANSFORM: NEW POSE, ACTION, CAMERA ANGLE or SCENE. Body moves or repositions.
  Examples: "sucking dick", "blowjob", "doggy style", "on the beach", "bending over", "riding", "kneeling", "selfie in mirror", "селфи у зеркала", "selfie", any scene description that differs from the original photo.
  CRITICAL: If the user says "selfie" or "mirror" and a photo is uploaded, it is ALWAYS TRANSFORM because a selfie requires a specific camera angle and pose change.
  
- CREATE: ONLY when NO photo is uploaded. Never use CREATE if a photo is present.

CRITICAL RULE: If "Photo uploaded: YES" → intent is ALWAYS either EDIT or TRANSFORM. NEVER CREATE.

## STEP 3: DETERMINE DENOISE
For EDIT only: 0.3-0.5 = subtle changes, 0.6-0.8 = significant changes.
For TRANSFORM/CREATE: always 0.0.

## STEP 4: DESCRIBE PERSON (for TRANSFORM/CREATE only)
If intent is TRANSFORM or CREATE and a photo is uploaded, write a SHORT body description (30-50 words) of the person in the photo: body type, skin tone, hair color/length. Do NOT describe face.

## RESPONSE FORMAT
RESPOND IN EXACTLY THIS JSON FORMAT:
{"intent": "EDIT", "nsfw": true, "denoise": 0.65, "body_desc": ""}

- intent: "EDIT", "TRANSFORM", or "CREATE"
- nsfw: true ONLY for explicit nudity or sex acts.
- denoise: float
- body_desc: string"""


async def classify_and_enhance(
    user_prompt: str,
    has_image: bool = False,
    image_b64: str | None = None,
) -> dict:
    """Classify intent and enhance prompt for Magic mode (2-step pipeline).

    Step 1: Classify intent (EDIT/TRANSFORM/CREATE) + extract body description
    Step 2: Enhance prompt using the correct pipeline-specific system prompt

    Returns:
        {
            "intent": "EDIT" | "TRANSFORM" | "CREATE",
            "enhanced_prompt": str,
            "nsfw": bool,
            "denoise": float,
        }
    """
    import json as _json

    api_key = config.XAI_API_KEY
    api_url = config.XAI_API_URL

    # Fallback: keyword-based classification (still runs step 2 enhancement!)
    def _keyword_classify():
        """Classify intent using keywords when LLM is unavailable."""
        _intent = "CREATE"
        _denoise = 0.0
        _nsfw = True

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
                # New scene descriptions (RU) → TRANSFORM
                "селфи", "у зеркала", "в комнате", "в спальне", "в ванной",
                "на кровати", "на полу", "ночное", "в машине", "на кухне",
                "в подъезде", "курит", "сидит", "стоит", "лежит",
                "фотка", "фотография", "делает фото", "снимает",
                # New scene descriptions (EN) → TRANSFORM
                "selfie", "mirror", "bedroom", "bathroom", "kitchen",
                "in a room", "at night", "smoking", "taking a photo",
            ]
            # Check for EDIT keywords first (simple surface changes)
            edit_keywords = [
                "remove", "undress", "nude", "naked", "topless",
                "change outfit", "change clothes", "add tattoo",
                "убери", "раздень", "голая", "сними", "переодень",
                "micro bikini", "бикини",
            ]
            if any(kw in prompt_lower for kw in edit_keywords) and not any(kw in prompt_lower for kw in transform_keywords):
                _intent = "EDIT"
                _denoise = 0.75
            elif any(kw in prompt_lower for kw in transform_keywords):
                _intent = "TRANSFORM"
            else:
                # Default for photo + unrecognized prompt → TRANSFORM (safer than EDIT)
                _intent = "TRANSFORM"

        return _intent, _nsfw, _denoise, ""  # intent, nsfw, denoise, body_desc

    if not api_key:
        logger.warning("XAI_API_KEY not set — using keyword classify + enhance")
        intent, nsfw, denoise, body_desc = _keyword_classify()
        classify_ms = 0

    else:
        # ─── STEP 1: CLASSIFY (LLM) ──────────────────────────────
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
        max_retries = 2
        classify_ok = False

        for attempt in range(max_retries + 1):
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
                            "max_tokens": 300,
                            "temperature": 0.3,
                            "stream": False,
                            "enable_thinking": False,
                        },
                        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                    ) as resp:
                        classify_ms = int((time.monotonic() - start) * 1000)

                        if resp.status >= 500 and attempt < max_retries:
                            error_text = await resp.text()
                            logger.warning(
                                "Magic classify 5xx (attempt %d/%d): %s — retrying...",
                                attempt + 1, max_retries + 1, error_text[:200],
                            )
                            await asyncio.sleep(1 * (attempt + 1))
                            continue

                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error("Magic classify API error %d: %s", resp.status, error_text[:300])
                            intent, nsfw, denoise, body_desc = _keyword_classify()
                            classify_ok = True
                            break

                        data = await resp.json()
                        raw = data["choices"][0]["message"]["content"].strip()

                        # Clean LLM artifacts
                        raw = _clean_response(raw)

                        # Parse JSON response
                        try:
                            result = _json.loads(raw)
                        except _json.JSONDecodeError:
                            import re
                            json_match = re.search(r'\{[^{}]*\}', raw)
                            if json_match:
                                result = _json.loads(json_match.group())
                            else:
                                logger.warning("Failed to parse classify response: %s", raw[:200])
                                intent, nsfw, denoise, body_desc = _keyword_classify()
                                classify_ok = True
                                break

                        intent = result.get("intent", "EDIT" if has_image else "CREATE")
                        nsfw = result.get("nsfw", True)
                        denoise = float(result.get("denoise", 0.75 if intent == "EDIT" else 0.0))
                        body_desc = result.get("body_desc", "")

                        logger.info(
                            "Magic classified in %dms (attempt=%d): intent=%s, nsfw=%s, denoise=%s, body=%s",
                            classify_ms, attempt + 1, intent, nsfw, denoise, body_desc[:60],
                        )
                        classify_ok = True
                        break

            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(
                        "Magic classify timeout (attempt %d/%d) — retrying...",
                        attempt + 1, max_retries + 1,
                    )
                    continue
                logger.warning("Magic classify timeout — using keyword classify")
                intent, nsfw, denoise, body_desc = _keyword_classify()
                classify_ms = int((time.monotonic() - start) * 1000)
                classify_ok = True
            except Exception as e:
                logger.exception("Magic classify error: %s", e)
                intent, nsfw, denoise, body_desc = _keyword_classify()
                classify_ms = int((time.monotonic() - start) * 1000)
                classify_ok = True
                break

        if not classify_ok:
            # All retries exhausted without break
            intent, nsfw, denoise, body_desc = _keyword_classify()
            classify_ms = int((time.monotonic() - start) * 1000)

    # ─── STEP 2: ENHANCE with pipeline-specific prompt ────────
    # Map intent to pipeline mode
    if intent == "EDIT":
        enhance_mode = "edit"
    elif intent == "TRANSFORM":
        enhance_mode = "bfs"
    else:  # CREATE
        enhance_mode = "t2i"

    # For TRANSFORM/CREATE, prepend body description from classifier
    # so the enhancer knows the person's appearance from the photo
    enhance_input = user_prompt
    if body_desc and intent in ("TRANSFORM", "CREATE"):
        enhance_input = f"[Person in photo: {body_desc}]\n{user_prompt}"

    # Pass image for EDIT mode (enhancer uses vision to see what to change)
    enhance_image = image_b64 if intent == "EDIT" and image_b64 else None

    enhanced_info = await enhance_prompt(
        user_prompt=enhance_input,
        image_b64=enhance_image,
        mode=enhance_mode,
    )

    total_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "Magic enhance done in %dms (classify=%dms): '%s' → '%s'",
        total_ms, classify_ms, user_prompt[:50], enhanced_info["enhanced"][:80],
    )

    return {
        "intent": intent,
        "enhanced_prompt": enhanced_info["enhanced"],
        "nsfw": nsfw,
        "denoise": denoise,
    }

