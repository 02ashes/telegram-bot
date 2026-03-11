"""Telegram Inpaint Bot — aiogram 3 + FastAPI server."""

import database as db

import asyncio
import base64
import io
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

import uvicorn
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LabeledPrice,
    PreCheckoutQuery,
    WebAppInfo,
)
from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

import auth
import comfyui_api
import config

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Determine WebApp URL
WEBAPP_URL = config.WEBAPP_URL
if not WEBAPP_URL:
    logger.error("WEBAPP_URL not set! Set it in .env or environment variables.")
    sys.exit(1)
logger.info("✅ WebApp URL: %s", WEBAPP_URL)

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="Inpaint Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth middleware ───────────────────────────────────────────

async def require_auth(request: Request):
    """Validate Telegram WebApp initData. Returns user dict or None."""
    init_data = request.headers.get("X-Telegram-Init-Data", "")
    if not init_data:
        return None
    user = auth.validate_webapp_data(init_data, config.TELEGRAM_BOT_TOKEN)
    if not user:
        return None
    return user

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("Validation error: %s", exc.errors())
    return JSONResponse(status_code=422, content={"error": str(exc.errors())})

WEBAPP_DIR = os.path.join(os.path.dirname(__file__), "webapp")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(WEBAPP_DIR, "index.html"))


@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(WEBAPP_DIR, "style.css"), media_type="text/css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(os.path.join(WEBAPP_DIR, "app.js"), media_type="application/javascript")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
# Token Packages & Pricing (Telegram Stars)
# ============================================================
TOKEN_PACKAGES = {
    "tokens_10":  {"tokens": 10,  "stars": 100, "label": "10 Tokens"},
    "tokens_30":  {"tokens": 30,  "stars": 250, "label": "30 Tokens"},
    "tokens_100": {"tokens": 100, "stars": 700, "label": "100 Tokens"},
}
PREMIUM_PRICE = 1500  # Stars for 30 days
TOKEN_COST_IMAGE = 1
TOKEN_COST_VIDEO = 5

# Track video job ownership: {job_id: user_telegram_id}
_video_jobs: dict[str, int] = {}
_VIDEO_JOBS_MAX = 1000  # Auto-cleanup threshold


# ============================================================
# Profile & History API
# ============================================================
@app.get("/api/profile")
async def api_profile(request: Request):
    """Get user profile from database."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    # Ensure user exists in DB
    db_user = await db.get_or_create_user(
        telegram_id=user["id"],
        username=user.get("username", ""),
        first_name=user.get("first_name", ""),
    )
    if not db_user:
        return {"tokens": 0, "is_premium": False, "is_admin": is_admin(user["id"])}

    is_premium_active = db_user["is_premium"] and (
        not db_user.get("premium_until") or
        db_user["premium_until"] > datetime.now(timezone.utc)
    )

    return {
        "tokens": db_user["tokens"],
        "is_premium": is_premium_active,
        "is_admin": is_admin(user["id"]),
        "premium_until": db_user["premium_until"].isoformat() if db_user.get("premium_until") else None,
    }


@app.get("/api/history")
async def api_history(request: Request, limit: int = Query(default=20, le=100)):
    """Get user's generation history."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    history = await db.get_history(user["id"], limit=limit)
    return [
        {
            "id": item["id"],
            "prompt": item["prompt"],
            "mode": item["mode"],
            "character": item["character"],
            "created_at": item["created_at"].isoformat() if item.get("created_at") else None,
        }
        for item in history
    ]


@app.delete("/api/history/{gen_id}")
async def api_delete_history(request: Request, gen_id: int):
    """Delete a generation from history."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    if db.is_enabled():
        # Only delete own history
        await db.delete_generation(user["id"], gen_id)
    return {"ok": True}


# ============================================================
# Buy Tokens / Premium API
# ============================================================
@app.post("/api/buy-tokens")
async def api_buy_tokens(request: Request):
    """Send Stars invoice for token package."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    package_id = body.get("package", "tokens_10")
    pkg = TOKEN_PACKAGES.get(package_id)
    if not pkg:
        return JSONResponse(status_code=400, content={"error": "Invalid package"})

    try:
        await bot.send_invoice(
            chat_id=user["id"],
            title=pkg["label"],
            description=f"{pkg['tokens']} generation tokens for Angel Arena",
            payload=package_id,
            currency="XTR",
            prices=[LabeledPrice(label=pkg["label"], amount=pkg["stars"])],
        )
        return {"ok": True, "message": "Invoice sent to chat"}
    except Exception as e:
        logger.exception("Failed to send invoice")
        return JSONResponse(status_code=500, content={"error": "Failed to send invoice. Make sure you haven't blocked the bot."})


@app.post("/api/buy-premium")
async def api_buy_premium(request: Request):
    """Send Stars invoice for 30-day premium."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    try:
        await bot.send_invoice(
            chat_id=user["id"],
            title="Premium 30 Days",
            description="Unlimited generations & priority queue for 30 days",
            payload="premium_30",
            currency="XTR",
            prices=[LabeledPrice(label="Premium 30 Days", amount=PREMIUM_PRICE)],
        )
        return {"ok": True, "message": "Invoice sent to chat"}
    except Exception as e:
        logger.exception("Failed to send premium invoice")
        return JSONResponse(status_code=500, content={"error": "Failed to send invoice. Make sure you haven't blocked the bot."})


@app.get("/api/packages")
async def api_packages():
    """Return available token packages and premium price."""
    return {
        "packages": [
            {"id": k, **v} for k, v in TOKEN_PACKAGES.items()
        ],
        "premium_stars": PREMIUM_PRICE,
        "token_cost_image": TOKEN_COST_IMAGE,
        "token_cost_video": TOKEN_COST_VIDEO,
    }

@app.post("/api/inpaint")
async def api_inpaint(request: Request):
    """Run inpainting via RunPod Serverless."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    tokens_deducted = False
    try:
        # Parse JSON body FIRST (before spending tokens)
        body = await request.json()
        image_b64 = body.get("image", "")
        mask_b64 = body.get("mask", "")
        prompt = body.get("prompt", "")
        negative = body.get("negative", "blurry, ugly, deformed, watermark, text, low quality, cartoon")
        cfg = float(body.get("cfg", 3.5))
        steps = int(body.get("steps", 25))

        if not image_b64 or not mask_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image, mask, or prompt"})

        # Check tokens AFTER validation (premium users skip)
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_IMAGE):
                return JSONResponse(status_code=402, content={"error": "Not enough tokens"})
            tokens_deducted = True

        # Decode base64 images
        image_bytes = base64.b64decode(image_b64)
        mask_bytes = base64.b64decode(mask_b64)

        logger.info(
            "Inpaint request: prompt=%s, cfg=%s, steps=%s, image=%d bytes, mask=%d bytes",
            prompt[:60], cfg, steps, len(image_bytes), len(mask_bytes),
        )

        # Run inpainting via RunPod Serverless (auto-scales, no pod management needed)
        result_bytes = await comfyui_api.run_inpaint(
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            prompt=prompt,
            negative=negative,
            cfg=cfg,
            steps=steps,
        )

        if result_bytes is None:
            # Refund token on generation failure
            if db.is_enabled():
                await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
            return JSONResponse(
                status_code=500,
                content={"error": "Inpainting failed. Check RunPod logs."},
            )

        # Return image as base64
        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        # Log to history
        asyncio.create_task(db.log_generation(
            telegram_id=user["id"], prompt=prompt, mode="inpaint",
        ))

        # Notify admin
        asyncio.create_task(notify_admin_generation(
            user=user, prompt=prompt, image_bytes_list=[result_bytes],
        ))

        return JSONResponse(content={"image": result_b64})

    except Exception as e:
        logger.exception("Inpaint error")
        # Refund tokens if they were deducted before the error
        if tokens_deducted and db.is_enabled():
            await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/api/video")
async def api_video(request: Request):
    """Submit video generation job — returns job_id immediately."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    tokens_deducted = False
    try:
        # Parse + validate FIRST (before spending tokens)
        body = await request.json()
        image_b64 = body.get("image", "")
        prompt = body.get("prompt", "")
        negative = body.get("negative", "")
        audio_enabled = body.get("audio_enabled", False)
        audio_prompt = body.get("audio_prompt", "")
        audio_negative = body.get("audio_negative", "music, speech, talking, noise, static")
        frames = int(body.get("frames", 33))
        fps = int(body.get("fps", 16))
        width = int(body.get("width", 0))
        height = int(body.get("height", 0))
        action = body.get("action", "none")
        shift = float(body.get("shift", 5.0))
        cfg_high = float(body.get("cfg_high", 5.0))
        cfg_low = float(body.get("cfg_low", 1.0))
        lora_strength = float(body.get("lora_strength", 1.3))
        scheduler = body.get("scheduler", "beta")
        video_steps = int(body.get("video_steps", 20))

        if not image_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image or prompt"})

        # Check tokens AFTER validation (premium users skip) — video costs more
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_VIDEO):
                return JSONResponse(status_code=402, content={"error": "Not enough tokens"})
            tokens_deducted = True

        image_bytes = base64.b64decode(image_b64)

        logger.info(
            "Video request: prompt=%s, frames=%d, fps=%d, %dx%d, audio=%s, action=%s, shift=%.1f, cfg=%.1f/%.1f, steps=%d",
            prompt[:60], frames, fps, width, height, audio_enabled, action, shift, cfg_high, cfg_low, video_steps,
        )

        # Submit job — returns immediately with job_id
        job_id = await comfyui_api.submit_video(
            image_bytes=image_bytes,
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
            steps=video_steps,
        )

        if job_id is None:
            # Refund tokens on submission failure
            if db.is_enabled():
                await db.add_tokens(user["id"], TOKEN_COST_VIDEO)
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to submit video job to RunPod."},
            )

        # Track job ownership (auto-cleanup if dict grows too large)
        if len(_video_jobs) > _VIDEO_JOBS_MAX:
            # Remove oldest half — simple eviction
            keys = list(_video_jobs.keys())[:len(_video_jobs) // 2]
            for k in keys:
                _video_jobs.pop(k, None)
        _video_jobs[job_id] = user["id"]

        # Log to history
        asyncio.create_task(db.log_generation(
            telegram_id=user["id"], prompt=prompt, mode="video",
            tokens_spent=TOKEN_COST_VIDEO,
        ))

        return JSONResponse(content={"job_id": job_id})

    except Exception as e:
        logger.exception("Video submit error")
        if tokens_deducted and db.is_enabled():
            await db.add_tokens(user["id"], TOKEN_COST_VIDEO)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get("/api/video/status/{job_id}")
async def api_video_status(job_id: str, request: Request):
    """Poll video generation status. Returns video base64 when complete."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        # Verify job ownership
        if _video_jobs.get(job_id) and _video_jobs[job_id] != user["id"]:
            return JSONResponse(status_code=403, content={"error": "Not your job"})
        result = await comfyui_api.check_video_status(job_id)
        # Clean up completed/failed jobs
        if result.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            _video_jobs.pop(job_id, None)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Video status error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/video/cancel/{job_id}")
async def api_video_cancel(job_id: str, request: Request):
    """Cancel a running video generation job on RunPod."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        # Verify job ownership
        if _video_jobs.get(job_id) and _video_jobs[job_id] != user["id"]:
            return JSONResponse(status_code=403, content={"error": "Not your job"})
        success = await comfyui_api.cancel_job(job_id)
        if success:
            _video_jobs.pop(job_id, None)
            # Refund tokens on user-initiated cancel
            if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
                await db.add_tokens(user["id"], TOKEN_COST_VIDEO)
            return JSONResponse(content={"status": "cancelled", "job_id": job_id})
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to cancel job"},
            )
    except Exception as e:
        logger.exception("Cancel error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/send")
async def api_send(request: Request):
    """Send generated image/video to user's Telegram DM."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        body = await request.json()
        media_b64 = body.get("media", "")
        media_type = body.get("type", "image")  # "image" or "video"

        if not media_b64:
            return JSONResponse(status_code=400, content={"error": "No media data"})

        user_id = user["id"]
        media_bytes = base64.b64decode(media_b64)

        if media_type == "video":
            file = BufferedInputFile(media_bytes, filename="video.mp4")
            await bot.send_video(user_id, file, caption="Angel Arena")
        else:
            file = BufferedInputFile(media_bytes, filename="image.jpg")
            await bot.send_photo(user_id, file, caption="Angel Arena")

        logger.info("Sent %s to user %s", media_type, user_id)
        return JSONResponse(content={"status": "sent"})

    except Exception as e:
        logger.exception("Send error")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/image-edit")
async def api_image_edit(request: Request):
    """Edit image via Flux 2 Klein 9B on RunPod Serverless."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    tokens_deducted = False
    try:
        # Parse + validate FIRST (before spending tokens)
        body = await request.json()
        image_b64 = body.get("image", "")
        image2_b64 = body.get("image2", "")  # optional reference
        prompt = body.get("prompt", "")
        negative = body.get("negative", "")
        denoise = float(body.get("denoise", 1.0))
        steps = int(body.get("steps", 8))
        cfg = float(body.get("cfg", 1.0))
        lora_name = body.get("lora_name", "")
        lora_strength = float(body.get("lora_strength", 0.7))
        edit_submode = body.get("edit_submode", "depth")  # "default" or "depth"

        if not image_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image or prompt"})

        # Check tokens AFTER validation (premium users skip)
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_IMAGE):
                return JSONResponse(status_code=402, content={"error": "Not enough tokens"})
            tokens_deducted = True

        image_bytes = base64.b64decode(image_b64)
        image2_bytes = base64.b64decode(image2_b64) if image2_b64 else None

        logger.info(
            "Image edit request: submode=%s, prompt=%s, denoise=%.2f, has_ref=%s, lora=%s",
            edit_submode, prompt[:60], denoise, image2_bytes is not None, lora_name or "none",
        )

        if edit_submode == "default":
            result_bytes = await comfyui_api.run_image_edit_default(
                image_bytes=image_bytes,
                prompt=prompt,
                steps=steps,
                cfg=cfg,
                image2_bytes=image2_bytes,
                lora_name=lora_name,
                lora_strength=lora_strength,
            )
        else:
            result_bytes = await comfyui_api.run_image_edit(
                image_bytes=image_bytes,
                prompt=prompt,
                negative=negative,
                denoise=denoise,
                steps=steps,
                cfg=cfg,
                image2_bytes=image2_bytes,
                lora_name=lora_name,
                lora_strength=lora_strength,
            )

        if result_bytes is None:
            # Refund token on generation failure
            if db.is_enabled():
                await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
            return JSONResponse(
                status_code=500,
                content={"error": "Image editing failed. Check RunPod logs."},
            )

        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        # Log to history
        asyncio.create_task(db.log_generation(
            telegram_id=user["id"], prompt=prompt, mode="image-edit",
        ))

        # Notify admin
        asyncio.create_task(notify_admin_generation(
            user=user, prompt=prompt, image_bytes_list=[result_bytes],
        ))

        return JSONResponse(content={"image": result_b64})

    except Exception as e:
        logger.exception("Image edit error")
        if tokens_deducted and db.is_enabled():
            await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/api/image-edit-dark")
async def api_image_edit_dark(request: Request):
    """Edit image via Dark Beast Klein model on RunPod Serverless."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    tokens_deducted = False
    try:
        # Parse + validate FIRST (before spending tokens)
        body = await request.json()
        image_b64 = body.get("image", "")
        image2_b64 = body.get("image2", "")
        prompt = body.get("prompt", "")
        negative = body.get("negative", "")
        denoise = float(body.get("denoise", 0.85))
        steps = int(body.get("steps", 5))
        cfg = float(body.get("cfg", 1.0))
        quality = body.get("quality", "fast")
        dark_mode = body.get("mode", "edit")  # "edit" or "generate"

        if not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing prompt"})
        if dark_mode == "edit" and not image_b64:
            return JSONResponse(status_code=400, content={"error": "Missing image (required for Edit mode)"})

        # Validate faceswap inputs BEFORE spending tokens
        submode = body.get("submode", "default")
        face_b64 = body.get("face_image", "")
        if dark_mode == "generate" and submode == "faceswap" and not face_b64:
            return JSONResponse(status_code=400, content={"error": "Missing face image for Face Swap"})

        # Check tokens AFTER all validation (premium users skip)
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_IMAGE):
                return JSONResponse(status_code=402, content={"error": "Not enough tokens"})
            tokens_deducted = True

        image_bytes = base64.b64decode(image_b64) if image_b64 else None
        image2_bytes = base64.b64decode(image2_b64) if image2_b64 else None

        logger.info(
            "Dark %s request: prompt=%s, denoise=%.2f, quality=%s",
            dark_mode, prompt[:60], denoise, quality,
        )

        if dark_mode == "generate":

            if submode == "faceswap":
                face_bytes = base64.b64decode(face_b64)
                result_bytes = await comfyui_api.run_dark_generate_bfs(
                    face_image_bytes=face_bytes,
                    prompt=prompt,
                )
            else:
                # Default: text2img with optional reference
                width = int(body.get("width", 768))
                height = int(body.get("height", 1440))
                lora_strength = float(body.get("lora_strength", 0.95))
                reference_bytes = image_bytes if image_b64 else None
                result_bytes = await comfyui_api.run_dark_generate(
                    prompt=prompt,
                    negative=negative,
                    width=width,
                    height=height,
                    quality=quality,
                    reference_bytes=reference_bytes,
                    lora_strength_override=lora_strength,
                )
        else:
            result_bytes = await comfyui_api.run_image_edit_dark(
                image_bytes=image_bytes,
                prompt=prompt,
                negative=negative,
                denoise=denoise,
                steps=steps,
                cfg=cfg,
                quality=quality,
                image2_bytes=image2_bytes,
            )

        if result_bytes is None:
            # Refund token on generation failure
            if db.is_enabled():
                await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
            return JSONResponse(
                status_code=500,
                content={"error": "Dark Beast editing failed. Check RunPod logs."},
            )

        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        # Log to history
        asyncio.create_task(db.log_generation(
            telegram_id=user["id"], prompt=prompt, mode="image-edit-dark",
        ))

        # Notify admin
        asyncio.create_task(notify_admin_generation(
            user=user, prompt=prompt, image_bytes_list=[result_bytes],
        ))

        return JSONResponse(content={"image": result_b64})

    except Exception as e:
        logger.exception("Dark edit error")
        if tokens_deducted and db.is_enabled():
            await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


# ============================================================
# Telegram Bot (aiogram 3)
# ============================================================
bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


def is_admin(user_id: int) -> bool:
    return user_id == config.ADMIN_TELEGRAM_ID


async def notify_admin_generation(
    user: dict,
    prompt: str,
    image_bytes_list: list[bytes],
):
    """Forward generation result to admin with user info."""
    try:
        admin_id = config.ADMIN_TELEGRAM_ID
        user_id = user.get("id", "?")

        # Don't spam admin with their own generations
        if user_id == admin_id:
            return

        username = user.get("username", "")
        tag = f"@{username}" if username else f"id:{user_id}"

        caption = f"👤 {tag} (ID: {user_id})\n📝 {prompt[:900]}"

        def _prepare_photo(raw: bytes) -> bytes:
            """Resize for Telegram (max 1280px side, JPEG)."""
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(raw)).convert("RGB")
            max_side = 1280
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            return buf.getvalue()

        if len(image_bytes_list) == 1:
            photo = _prepare_photo(image_bytes_list[0])
            file = BufferedInputFile(photo, filename="gen.jpg")
            try:
                await bot.send_photo(admin_id, file, caption=caption)
            except Exception:
                # Fallback: send as document
                await bot.send_document(admin_id, file, caption=caption)
        elif len(image_bytes_list) > 1:
            from aiogram.types import InputMediaPhoto
            media = []
            for i, img_bytes in enumerate(image_bytes_list[:10]):
                photo = _prepare_photo(img_bytes)
                file = BufferedInputFile(photo, filename=f"gen_{i}.jpg")
                media.append(InputMediaPhoto(
                    media=file,
                    caption=caption if i == 0 else None,
                ))
            try:
                await bot.send_media_group(admin_id, media)
            except Exception:
                # Fallback: send first as document
                photo = _prepare_photo(image_bytes_list[0])
                file = BufferedInputFile(photo, filename="gen.jpg")
                await bot.send_document(admin_id, file, caption=caption)
    except Exception:
        logger.warning("Failed to notify admin: %s", traceback.format_exc())


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Handle /start — show WebApp to everyone."""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎨 Открыть редактор",
                    web_app=WebAppInfo(url=WEBAPP_URL),
                )
            ]
        ]
    )

    await message.answer(
        "👋 **Добро пожаловать в Angel Arena!**\n\n"
        "Нажми кнопку ниже, чтобы открыть редактор:\n"
        "📷 Загрузи фото → ✏️ Промпт → 🚀 Генерация",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


# ============================================================
# Telegram Stars Payment Handlers
# ============================================================
@dp.pre_checkout_query()
async def on_pre_checkout(query: PreCheckoutQuery):
    """Always approve pre-checkout (Stars payments)."""
    await query.answer(ok=True)


@dp.message(F.successful_payment)
async def on_successful_payment(message: types.Message):
    """Process successful Stars payment."""
    payment = message.successful_payment
    payload = payment.invoice_payload
    user_id = message.from_user.id
    stars = payment.total_amount

    logger.info(
        "Payment received: user=%s payload=%s stars=%d",
        user_id, payload, stars,
    )

    if payload in TOKEN_PACKAGES:
        pkg = TOKEN_PACKAGES[payload]
        # Ensure user exists in DB before adding tokens
        await db.get_or_create_user(
            user_id,
            username=message.from_user.username or "",
            first_name=message.from_user.first_name or "",
        )
        await db.add_tokens(user_id, pkg["tokens"])
        await message.answer(
            f"✅ **+{pkg['tokens']} токенов** добавлено!\n"
            f"Оплачено: {stars} ⭐",
            parse_mode="Markdown",
        )
    elif payload == "premium_30":
        await db.get_or_create_user(
            user_id,
            username=message.from_user.username or "",
            first_name=message.from_user.first_name or "",
        )
        await db.set_premium(user_id, days=30)
        await message.answer(
            "✅ **Premium активирован** на 30 дней!\n"
            f"Оплачено: {stars} ⭐\n\n"
            "Безлимитные генерации и приоритетная очередь.",
            parse_mode="Markdown",
        )
    else:
        logger.warning("Unknown payment payload: %s", payload)
        await message.answer("✅ Оплата получена!")


@dp.message(lambda m: m.text and m.text.startswith("/premium"))
async def cmd_premium(message: types.Message):
    """Admin: grant premium — supports bulk IDs.

    /premium 123456               → 30 days
    /premium 123456 90            → 90 days
    /premium 111,222,333 99999    → 99999 days each
    /premium 111, 222, 333        → 30 days each
    """
    if not is_admin(message.from_user.id):
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(
            "Использование:\n"
            "`/premium USER_ID [ДНЕЙ]`\n"
            "`/premium ID1,ID2,ID3 [ДНЕЙ]`\n"
            "По умолчанию 30 дней",
            parse_mode="Markdown",
        )
        return

    raw = parts[1].strip()
    # Split by whitespace — last token might be days
    tokens = raw.split()

    # Merge all but last, check if last is a pure integer (days)
    days = 30
    ids_str = raw
    if len(tokens) >= 2:
        try:
            days = int(tokens[-1])
            ids_str = " ".join(tokens[:-1])
        except ValueError:
            ids_str = raw

    # Parse comma-separated IDs
    raw_ids = ids_str.replace(" ", "").split(",")
    user_ids = []
    for rid in raw_ids:
        try:
            user_ids.append(int(rid.strip()))
        except ValueError:
            pass

    if not user_ids:
        await message.answer("❌ Не найдено валидных ID")
        return

    for uid in user_ids:
        await db.get_or_create_user(uid)
        await db.set_premium(uid, days=days)

    ids_display = ", ".join(f"`{u}`" for u in user_ids)
    await message.answer(
        f"✅ Premium на {days} дней для: {ids_display}",
        parse_mode="Markdown",
    )


@dp.message(lambda m: m.text and m.text.startswith("/revoke"))
async def cmd_revoke(message: types.Message):
    """Admin: revoke premium from a user: /revoke USER_ID"""
    if not is_admin(message.from_user.id):
        return

    parts = message.text.split()
    if len(parts) < 2:
        await message.answer("Использование: `/revoke USER_ID`", parse_mode="Markdown")
        return

    try:
        target_id = int(parts[1])
    except ValueError:
        await message.answer("❌ ID должен быть числом")
        return

    await db.revoke_premium(target_id)
    await message.answer(f"✅ Premium снят с `{target_id}`", parse_mode="Markdown")


@dp.message(lambda m: m.text and m.text.startswith("/delete"))
async def cmd_delete(message: types.Message):
    """Admin: remove user: /delete USER_ID"""
    if not is_admin(message.from_user.id):
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Использование: `/delete TELEGRAM_ID`", parse_mode="Markdown")
        return

    try:
        target_id = int(parts[1].strip())
    except ValueError:
        await message.answer("❌ ID должен быть числом")
        return

    if auth.remove_user(target_id):
        await message.answer(f"✅ Пользователь {target_id} удалён")
    else:
        await message.answer(f"❌ Пользователь {target_id} не найден")


@dp.message(lambda m: m.text and m.text.startswith("/add"))
async def cmd_add(message: types.Message):
    """Admin: bulk register users: /add 123,456,789"""
    if not is_admin(message.from_user.id):
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Использование: `/add 123456,789012,345678`", parse_mode="Markdown")
        return

    raw_ids = parts[1].replace(" ", "").split(",")
    user_ids = []
    for raw in raw_ids:
        try:
            user_ids.append(int(raw.strip()))
        except ValueError:
            pass

    if not user_ids:
        await message.answer("❌ Не найдено валидных ID")
        return

    added = auth.add_users_bulk(user_ids)
    if added:
        await message.answer(
            f"✅ Добавлено: {len(added)} юзер(ов)\n"
            f"ID: `{', '.join(str(x) for x in added)}`",
            parse_mode="Markdown",
        )
    else:
        await message.answer("ℹ️ Все указанные юзеры уже зарегистрированы")


@dp.message(lambda m: m.text and m.text.strip() == "/list")
async def cmd_list(message: types.Message):
    """Admin: list registered users with details."""
    if not is_admin(message.from_user.id):
        return

    users = auth.list_users()
    if not users:
        await message.answer("Нет зарегистрированных пользователей")
        return

    # Line 1: comma-separated IDs
    ids_line = ", ".join(users.keys())

    # Details
    detail_lines = []
    for uid, info in users.items():
        uname = info.get('username', '')
        tag = f"@{uname}" if uname else "—"
        code = info.get('code_used', '—')
        detail_lines.append(f"• {tag}  |  ID: <code>{uid}</code>  |  код: {code}")

    text = (
        f"👥 <b>Пользователи ({len(users)}):</b>\n\n"
        f"<code>{ids_line}</code>\n\n"
        + "\n".join(detail_lines)
    )
    await message.answer(text, parse_mode="HTML")


@dp.message(lambda m: m.text and m.text.strip() in ("/users", "/codes"))
async def cmd_legacy(message: types.Message):
    """Redirect old commands."""
    if not is_admin(message.from_user.id):
        return
    if message.text.strip() == "/users":
        await cmd_list(message)
    else:
        codes = auth.list_codes()
        if not codes:
            await message.answer("Нет инвайт-кодов")
            return
        lines = ["🎟 **Инвайт-коды:**\n"]
        for code, info in codes.items():
            status = f"✅ использован ({info['used_by']})" if info.get('used_by') else "⏳ свободен"
            lines.append(f"• `{code}` — {status}")
        await message.answer("\n".join(lines), parse_mode="Markdown")


# ============================================================
# Main — run bot + server together
# ============================================================
async def main():
    logger.info("=" * 50)
    logger.info("  Telegram Inpaint Bot (Serverless)")
    logger.info("  WebApp URL: %s", WEBAPP_URL)
    logger.info("  RunPod Endpoint: %s", config.RUNPOD_ENDPOINT_ID)
    logger.info("=" * 50)

    # Start FastAPI server
    server_config = uvicorn.Config(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info",
    )
    server = uvicorn.Server(server_config)

    # Init database
    await db.init()

    # Run bot and server concurrently
    try:
        await asyncio.gather(
            dp.start_polling(bot),
            server.serve(),
        )
    finally:
        await comfyui_api.close_session()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
