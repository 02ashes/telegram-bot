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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

import auth
import comfyui_api
import config
import prompt_enhance

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
# Telegram Bot (aiogram 3) — initialized early so is_admin()
# is available for FastAPI routes below
# ============================================================
bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


def is_admin(user_id: int) -> bool:
    return user_id == config.ADMIN_TELEGRAM_ID

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
TOKEN_COST_KENPECHI = 37  # ~$0.47 cost + 60% margin

# Quick-buy: single generation prices (Stars)
QUICK_BUY_IMAGE_STARS = 15
QUICK_BUY_VIDEO_STARS = 399

# Track video job ownership: {job_id: (user_telegram_id, endpoint_id)}
_video_jobs: dict[str, tuple] = {}
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
        "total_gens": db_user.get("total_gens", 0),
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
    """Create invoice link for token package (for WebApp openInvoice)."""
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
        invoice_link = await bot.create_invoice_link(
            title=pkg["label"],
            description=f"{pkg['tokens']} generation tokens for Angel Arena",
            payload=package_id,
            currency="XTR",
            prices=[LabeledPrice(label=pkg["label"], amount=pkg["stars"])],
        )
        return {"ok": True, "invoice_url": invoice_link}
    except Exception as e:
        logger.exception("Failed to create invoice link")
        return JSONResponse(status_code=500, content={"error": "Failed to create invoice"})


@app.post("/api/buy-premium")
async def api_buy_premium(request: Request):
    """Create invoice link for 30-day premium (for WebApp openInvoice)."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    try:
        invoice_link = await bot.create_invoice_link(
            title="Premium 30 Days",
            description="Unlimited generations & priority queue for 30 days",
            payload="premium_30",
            currency="XTR",
            prices=[LabeledPrice(label="Premium 30 Days", amount=PREMIUM_PRICE)],
        )
        return {"ok": True, "invoice_url": invoice_link}
    except Exception as e:
        logger.exception("Failed to create premium invoice link")
        return JSONResponse(status_code=500, content={"error": "Failed to create invoice"})


@app.post("/api/quick-buy")
async def api_quick_buy(request: Request):
    """Create invoice link for single image/video generation (for WebApp openInvoice)."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    buy_type = body.get("type", "image")
    if buy_type == "video":
        title = "1 Video Generation"
        description = "37 tokens for one video generation"
        payload = "quick_video"
        stars = QUICK_BUY_VIDEO_STARS
    else:
        title = "1 Image Generation"
        description = "1 token for one image generation"
        payload = "quick_image"
        stars = QUICK_BUY_IMAGE_STARS

    try:
        invoice_link = await bot.create_invoice_link(
            title=title,
            description=description,
            payload=payload,
            currency="XTR",
            prices=[LabeledPrice(label=title, amount=stars)],
        )
        return {"ok": True, "invoice_url": invoice_link}
    except Exception as e:
        logger.exception("Failed to create quick-buy invoice link")
        return JSONResponse(status_code=500, content={"error": "Failed to create invoice"})



@app.get("/api/packages")
async def api_packages():
    """Return available token packages and premium price."""
    return {
        "packages": [
            {"id": k, **v} for k, v in TOKEN_PACKAGES.items()
        ],
        "premium_stars": PREMIUM_PRICE,
        "token_cost_image": TOKEN_COST_IMAGE,
        "token_cost_kenpechi": TOKEN_COST_KENPECHI,
        "quick_buy_image": QUICK_BUY_IMAGE_STARS,
        "quick_buy_video": QUICK_BUY_VIDEO_STARS,
    }


@app.post("/api/enhance-prompt")
async def api_enhance_prompt(request: Request):
    """Enhance user prompt via SiliconFlow Vision LLM (standalone call)."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    try:
        body = await request.json()
        user_prompt = body.get("prompt", "").strip()
        image_b64 = body.get("image", "")  # optional: uploaded photo for vision analysis
        mode = body.get("mode", "edit")  # "edit", "generate", "dark"
        lora_trigger = body.get("lora_trigger", "")

        if not user_prompt:
            return JSONResponse(status_code=400, content={"error": "Missing prompt"})

        if not config.AUTOPROMPT_ENABLED:
            return JSONResponse(content={
                "enhanced": user_prompt,
                "original": user_prompt,
                "time_ms": 0,
            })

        result = await prompt_enhance.enhance_prompt(
            user_prompt=user_prompt,
            image_b64=image_b64 or None,
            mode=mode,
            lora_trigger=lora_trigger,
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Enhance prompt error")
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "enhanced": body.get("prompt", ""),
            "original": body.get("prompt", ""),
        })


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
        auto_prompt = body.get("auto_prompt", False)

        if not image_b64 or not mask_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image, mask, or prompt"})

        # Auto-prompt enhancement
        original_prompt = prompt
        enhanced_info = None
        if auto_prompt and config.AUTOPROMPT_ENABLED:
            enhanced_info = await prompt_enhance.enhance_prompt(
                user_prompt=prompt,
                image_b64=image_b64,
                mode="edit",
            )
            prompt = enhanced_info["enhanced"]

        # Check tokens AFTER validation (premium/admin skip)
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_IMAGE):
                # Try free generation
                if not await db.use_free_gen(user["id"]):
                    return JSONResponse(status_code=402, content={
                        "error": "Not enough tokens",
                        "quick_buy": {"image": QUICK_BUY_IMAGE_STARS, "video": QUICK_BUY_VIDEO_STARS},
                    })
                # free_gen consumed — no token refund needed on failure
            else:
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
            if tokens_deducted and db.is_enabled():
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

        # Build admin caption with both prompts
        admin_prompt = prompt
        if enhanced_info and enhanced_info["model"] != "fallback":
            admin_prompt = f"✨ AUTO-PROMPT ({enhanced_info['time_ms']}ms)\n📝 Original: {original_prompt}\n🔮 Enhanced: {prompt}"

        # Notify admin
        asyncio.create_task(notify_admin_generation(
            user=user, prompt=admin_prompt, image_bytes_list=[result_bytes],
        ))

        # Send result to user's DM
        asyncio.create_task(send_result_to_user(
            user=user, prompt=prompt, image_bytes_list=[result_bytes],
        ))

        response_data = {"image": result_b64}
        if enhanced_info and enhanced_info["model"] != "fallback":
            response_data["enhanced_prompt"] = prompt
            response_data["original_prompt"] = original_prompt
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.exception("Inpaint error")
        # Refund tokens if they were deducted before the error
        if tokens_deducted and db.is_enabled():
            await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )





@app.post("/api/video/kenpechi")
async def api_video_kenpechi(request: Request):
    """Submit Kenpechi SVI 6-scene video generation job."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    tokens_deducted = False
    try:
        body = await request.json()
        image_b64 = body.get("image", "")
        scenes = body.get("scenes", [])
        negative = body.get("negative", "")
        width = int(body.get("width", 720))
        height = int(body.get("height", 1072))
        steps = int(body.get("steps", 7))
        split_steps = int(body.get("split_steps", 3))
        fps = float(body.get("fps", 16))
        rife_multiplier = int(body.get("rife_multiplier", 4))
        svi_motion_strength = float(body.get("svi_motion_strength", 1.0))
        repulsion_boost = float(body.get("repulsion_boost", 1.0))
        shift = float(body.get("shift", 5.0))

        if not image_b64:
            return JSONResponse(status_code=400, content={"error": "Missing image"})
        if not scenes or len(scenes) < 1:
            return JSONResponse(status_code=400, content={"error": "At least 1 scene required"})
        if len(scenes) > 6:
            scenes = scenes[:6]

        for i, sc in enumerate(scenes):
            if not sc.get("prompt", "").strip():
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Scene {i + 1} is missing a prompt"},
                )

        # Token check (no free gens for video)
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_KENPECHI):
                return JSONResponse(status_code=402, content={
                    "error": "Not enough tokens",
                    "quick_buy": {"image": QUICK_BUY_IMAGE_STARS, "video": QUICK_BUY_VIDEO_STARS},
                })
            tokens_deducted = True

        image_bytes = base64.b64decode(image_b64)

        logger.info(
            "Kenpechi video request: %d scenes, %dx%d, steps=%d, user=%s",
            len(scenes), width, height, steps, user["id"],
        )

        job_id = await comfyui_api.submit_kenpechi_video(
            image_bytes=image_bytes,
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

        if job_id is None:
            if tokens_deducted and db.is_enabled():
                await db.add_tokens(user["id"], TOKEN_COST_KENPECHI)
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to submit Kenpechi video job."},
            )

        if len(_video_jobs) > _VIDEO_JOBS_MAX:
            keys = list(_video_jobs.keys())[:len(_video_jobs) // 2]
            for k in keys:
                _video_jobs.pop(k, None)
        _video_jobs[job_id] = (user["id"], config.RUNPOD_KENPECHI_ENDPOINT_ID)

        asyncio.create_task(db.log_generation(
            telegram_id=user["id"],
            prompt=scenes[0].get("prompt", "")[:100],
            mode="kenpechi_video",
            tokens_spent=TOKEN_COST_KENPECHI,
        ))

        return JSONResponse(content={"job_id": job_id})

    except Exception as e:
        logger.exception("Kenpechi video submit error")
        if tokens_deducted and db.is_enabled():
            await db.add_tokens(user["id"], TOKEN_COST_KENPECHI)
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
        job_info = _video_jobs.get(job_id)
        if job_info:
            job_user_id, job_endpoint_id = job_info
            if job_user_id != user["id"]:
                return JSONResponse(status_code=403, content={"error": "Not your job"})
        else:
            job_endpoint_id = None
        result = await comfyui_api.check_video_status(job_id, endpoint_id=job_endpoint_id)
        # Clean up completed/failed jobs
        if result.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            _video_jobs.pop(job_id, None)

        # Auto-send video to user's DM when completed
        if result.get("status") == "COMPLETED" and result.get("video"):
            asyncio.create_task(_send_video_to_user(user, result["video"]))

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
        job_info = _video_jobs.get(job_id)
        if job_info and job_info[0] != user["id"]:
            return JSONResponse(status_code=403, content={"error": "Not your job"})
        job_endpoint_id = job_info[1] if job_info else None
        success = await comfyui_api.cancel_job(job_id, endpoint_id=job_endpoint_id)
        if success:
            _video_jobs.pop(job_id, None)
            # Refund tokens on user-initiated cancel
            if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
                await db.add_tokens(user["id"], TOKEN_COST_KENPECHI)
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
        auto_prompt = body.get("auto_prompt", False)

        if not image_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image or prompt"})

        # Auto-prompt enhancement
        original_prompt = prompt
        enhanced_info = None
        if auto_prompt and config.AUTOPROMPT_ENABLED:
            enhanced_info = await prompt_enhance.enhance_prompt(
                user_prompt=prompt,
                image_b64=image_b64,
                mode="edit",
                lora_trigger=body.get("lora_trigger", ""),
            )
            prompt = enhanced_info["enhanced"]

        # Check tokens AFTER validation (premium/admin skip)
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_IMAGE):
                # Try free generation
                if not await db.use_free_gen(user["id"]):
                    return JSONResponse(status_code=402, content={
                        "error": "Not enough tokens",
                        "quick_buy": {"image": QUICK_BUY_IMAGE_STARS, "video": QUICK_BUY_VIDEO_STARS},
                    })
                # free_gen consumed — no token refund needed on failure
            else:
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
            if tokens_deducted and db.is_enabled():
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

        # Build admin caption with both prompts
        admin_prompt = prompt
        if enhanced_info and enhanced_info["model"] != "fallback":
            admin_prompt = f"✨ AUTO-PROMPT ({enhanced_info['time_ms']}ms)\n📝 Original: {original_prompt}\n🔮 Enhanced: {prompt}"

        # Notify admin
        asyncio.create_task(notify_admin_generation(
            user=user, prompt=admin_prompt, image_bytes_list=[result_bytes],
        ))

        # Send result to user's DM
        asyncio.create_task(send_result_to_user(
            user=user, prompt=prompt, image_bytes_list=[result_bytes],
        ))

        response_data = {"image": result_b64}
        if enhanced_info and enhanced_info["model"] != "fallback":
            response_data["enhanced_prompt"] = prompt
            response_data["original_prompt"] = original_prompt
        return JSONResponse(content=response_data)

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
        auto_prompt = body.get("auto_prompt", False)

        if not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing prompt"})
        if dark_mode == "edit" and not image_b64:
            return JSONResponse(status_code=400, content={"error": "Missing image (required for Edit mode)"})

        # Auto-prompt enhancement
        original_prompt = prompt
        enhanced_info = None
        if auto_prompt and config.AUTOPROMPT_ENABLED:
            enhanced_info = await prompt_enhance.enhance_prompt(
                user_prompt=prompt,
                image_b64=image_b64 if image_b64 else None,
                mode="dark" if dark_mode == "edit" else "generate",
                lora_trigger=body.get("lora_trigger", ""),
            )
            prompt = enhanced_info["enhanced"]

        # Validate faceswap inputs BEFORE spending tokens
        submode = body.get("submode", "default")
        face_b64 = body.get("face_image", "")
        if dark_mode == "generate" and submode == "faceswap" and not face_b64:
            return JSONResponse(status_code=400, content={"error": "Missing face image for Face Swap"})

        # Check tokens AFTER all validation (premium/admin skip)
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_IMAGE):
                # Try free generation
                if not await db.use_free_gen(user["id"]):
                    return JSONResponse(status_code=402, content={
                        "error": "Not enough tokens",
                        "quick_buy": {"image": QUICK_BUY_IMAGE_STARS, "video": QUICK_BUY_VIDEO_STARS},
                    })
                # free_gen consumed — no token refund needed on failure
            else:
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
                # Default: text2img
                width = int(body.get("width", 768))
                height = int(body.get("height", 1440))
                lora_strength = float(body.get("lora_strength", 0.95))
                result_bytes = await comfyui_api.run_dark_generate(
                    prompt=prompt,
                    negative=negative,
                    width=width,
                    height=height,
                    quality=quality,
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
            if tokens_deducted and db.is_enabled():
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

        # Build admin caption with both prompts
        admin_prompt = prompt
        if enhanced_info and enhanced_info["model"] != "fallback":
            admin_prompt = f"✨ AUTO-PROMPT ({enhanced_info['time_ms']}ms)\n📝 Original: {original_prompt}\n🔮 Enhanced: {prompt}"

        # Notify admin
        asyncio.create_task(notify_admin_generation(
            user=user, prompt=admin_prompt, image_bytes_list=[result_bytes],
        ))

        # Send result to user's DM
        asyncio.create_task(send_result_to_user(
            user=user, prompt=prompt, image_bytes_list=[result_bytes],
        ))

        response_data = {"image": result_b64}
        if enhanced_info and enhanced_info["model"] != "fallback":
            response_data["enhanced_prompt"] = prompt
            response_data["original_prompt"] = original_prompt
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.exception("Dark edit error")
        if tokens_deducted and db.is_enabled():
            await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

# ============================================================
# Text to Image — Final T2I pipeline (default mode)
# ============================================================
@app.post("/api/image-test")
async def api_image_test(request: Request):
    """Generate image via Final T2I pipeline on dedicated endpoint."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    tokens_deducted = False
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        negative = body.get("negative", "")
        aspect_ratio = body.get("aspect_ratio", "5:7 (Balanced Portrait)")
        nsfw = body.get("nsfw", False)
        redefine = body.get("redefine", False)
        upscale = body.get("upscale", False)
        auto_prompt = body.get("auto_prompt", False)
        detailers = body.get("detailers", {
            "face": True, "eyes": True, "hands": True,
            "foot": False, "nipples": False, "pussy": False, "penis": False,
        })

        if not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing prompt"})

        # Auto-prompt enhancement
        original_prompt = prompt
        enhanced_info = None
        if auto_prompt and config.AUTOPROMPT_ENABLED:
            enhanced_info = await prompt_enhance.enhance_prompt(
                user_prompt=prompt,
                image_b64=None,
                mode="generate",
                lora_trigger=body.get("lora_trigger", ""),
            )
            prompt = enhanced_info["enhanced"]

        # Check tokens (premium/admin skip)
        if db.is_enabled() and not is_admin(user["id"]) and not await db.is_premium(user["id"]):
            if not await db.spend_tokens(user["id"], TOKEN_COST_IMAGE):
                if not await db.use_free_gen(user["id"]):
                    return JSONResponse(status_code=402, content={
                        "error": "Not enough tokens",
                        "quick_buy": {"image": QUICK_BUY_IMAGE_STARS, "video": QUICK_BUY_VIDEO_STARS},
                    })
            else:
                tokens_deducted = True

        logger.info(
            "T2I request: prompt=%s, aspect=%s, nsfw=%s, redefine=%s, upscale=%s",
            prompt[:60], aspect_ratio, nsfw, redefine, upscale,
        )

        result_bytes = await comfyui_api.run_t2i_generate(
            prompt=prompt,
            negative=negative,
            aspect_ratio=aspect_ratio,
            nsfw=nsfw,
            redefine=redefine,
            upscale=upscale,
            detailers=detailers,
        )

        if result_bytes is None:
            if tokens_deducted and db.is_enabled():
                await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
            return JSONResponse(
                status_code=500,
                content={"error": "T2I generation failed. Check RunPod logs."},
            )

        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        # Log to history
        asyncio.create_task(db.log_generation(
            telegram_id=user["id"], prompt=prompt, mode="image-t2i",
        ))

        # Build admin caption
        admin_prompt = prompt
        if enhanced_info and enhanced_info["model"] != "fallback":
            admin_prompt = f"✨ AUTO-PROMPT ({enhanced_info['time_ms']}ms)\n📝 Original: {original_prompt}\n🔮 Enhanced: {prompt}"

        mode_tag = "NSFW" if nsfw else "SFW"
        asyncio.create_task(notify_admin_generation(
            user=user, prompt=f"[T2I {mode_tag}] {admin_prompt}", image_bytes_list=[result_bytes],
        ))

        asyncio.create_task(send_result_to_user(
            user=user, prompt=prompt, image_bytes_list=[result_bytes],
        ))

        response_data = {"image": result_b64}
        if enhanced_info and enhanced_info["model"] != "fallback":
            response_data["enhanced_prompt"] = prompt
            response_data["original_prompt"] = original_prompt
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.exception("T2I error")
        if tokens_deducted and db.is_enabled():
            await db.add_tokens(user["id"], TOKEN_COST_IMAGE)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

# ── Serve webapp static files ────────────────────────────────
# MUST be after all @app routes — catch-all mount at "/" intercepts everything
app.mount("/", StaticFiles(directory=WEBAPP_DIR, html=True), name="webapp")


def _prepare_photo(raw: bytes) -> bytes:
    """Resize image for Telegram (max 1280px side, JPEG)."""
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    max_side = 1280
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()

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

        # Uses shared _prepare_photo() helper defined above

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


async def send_result_to_user(
    user: dict,
    prompt: str,
    image_bytes_list: list[bytes],
):
    """Send generation result (image + prompt) to user's Telegram DM."""
    try:
        user_id = user.get("id")
        if not user_id:
            return

        caption = f"🎨 Your generation\n📝 {prompt[:900]}"

        # Uses shared _prepare_photo() helper defined above

        if len(image_bytes_list) == 1:
            photo = _prepare_photo(image_bytes_list[0])
            file = BufferedInputFile(photo, filename="generation.jpg")
            try:
                await bot.send_photo(user_id, file, caption=caption)
            except Exception:
                await bot.send_document(user_id, file, caption=caption)
        elif len(image_bytes_list) > 1:
            from aiogram.types import InputMediaPhoto
            media = []
            for i, img_bytes in enumerate(image_bytes_list[:10]):
                photo = _prepare_photo(img_bytes)
                file = BufferedInputFile(photo, filename=f"generation_{i}.jpg")
                media.append(InputMediaPhoto(
                    media=file,
                    caption=caption if i == 0 else None,
                ))
            try:
                await bot.send_media_group(user_id, media)
            except Exception:
                photo = _prepare_photo(image_bytes_list[0])
                file = BufferedInputFile(photo, filename="generation.jpg")
                await bot.send_document(user_id, file, caption=caption)
    except Exception:
        logger.warning("Failed to send result to user %s: %s", user.get('id'), traceback.format_exc())


async def _send_video_to_user(user: dict, video_b64: str):
    """Send completed video to user's Telegram DM and notify admin."""
    try:
        user_id = user.get("id")
        if not user_id:
            return

        video_bytes = base64.b64decode(video_b64)
        file = BufferedInputFile(video_bytes, filename="generation.mp4")
        await bot.send_video(user_id, file, caption="🎬 Your video generation")

        # Also notify admin
        await _notify_admin_video(user, video_bytes)
    except Exception:
        logger.warning("Failed to send video to user %s: %s", user.get('id'), traceback.format_exc())


async def _notify_admin_video(user: dict, video_bytes: bytes):
    """Forward video result to admin with user info."""
    try:
        admin_id = config.ADMIN_TELEGRAM_ID
        user_id = user.get("id", "?")

        # Don't notify for admin's own generations
        if user_id == admin_id:
            return

        username = user.get("username", "")
        tag = f"@{username}" if username else f"id:{user_id}"
        caption = f"🎬 Video from {tag} (ID: {user_id})"

        file = BufferedInputFile(video_bytes, filename="generation.mp4")
        await bot.send_video(admin_id, file, caption=caption)
    except Exception:
        logger.warning("Failed to notify admin about video: %s", traceback.format_exc())

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Handle /start — show WebApp to everyone."""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Open Editor",
                    web_app=WebAppInfo(url=WEBAPP_URL),
                )
            ],
            [
                InlineKeyboardButton(text="Profile", callback_data="cb_profile"),
                InlineKeyboardButton(text="Tokens", callback_data="cb_tokens"),
                InlineKeyboardButton(text="Premium", callback_data="cb_premium"),
            ],
            [
                InlineKeyboardButton(text="📖 Instructions", callback_data="cb_instructions"),
            ],
        ]
    )

    await message.answer(
        "**Welcome to Angel Arena**\n\n"
        "AI-powered image and video generation.\n\n"
        "How to start:\n"
        "1. Tap the button below\n"
        "2. Upload a photo\n"
        "3. Choose a mode and describe what you want\n"
        "4. Hit Generate",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


# ============================================================
# Inline Button Callbacks (Profile / Tokens / Premium / Instructions)
# ============================================================
@dp.callback_query(F.data == "cb_instructions")
async def cb_instructions(callback: types.CallbackQuery):
    """Show mode instructions."""
    text = (
        "📖 <b>Instructions</b>\n\n"
        "<b>🖌 Inpaint</b>\n"
        "Great for adding specific objects or text. For example, use the \"Name\" "
        "auto-prompt to write your name on skin.\n\n"
        "<b>🎥 Video</b>\n"
        "Split into 6 sections. You can edit each one, but it works best "
        "with the same prompt across all sections. Max duration ~13 sec. "
        "Estimated wait time ~4 minutes.\n\n"
        "<b>✏️ Edit Easy (Default)</b>\n"
        "Standard image editor — add objects or elements to your photo.\n\n"
        "<b>⚡ Edit Easy (Depth/Fast/Detailed)</b>\n"
        "NSFW editor. Undress, add cum, or transform by prompt — anything goes.\n\n"
        "<b>🎨 Generate</b>\n"
        "Create images from scratch. You can include your own girls/characters "
        "in generations (contact @vowofquiet to add your character).\n\n"
        "<b>🔄 Face Swap</b>\n"
        "Generates an image, then blends in facial details from your photo.\n\n"
        "<i>Remember — everything depends on your prompt. "
        "Think carefully about your prompts and you'll get exactly the result you want.</i>"
    )
    await callback.message.answer(text, parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "cb_profile")
async def cb_profile(callback: types.CallbackQuery):
    """Show user profile info."""
    user_id = callback.from_user.id
    db_user = await db.get_or_create_user(
        user_id,
        username=callback.from_user.username or "",
        first_name=callback.from_user.first_name or "",
    )
    is_prem = db_user["is_premium"] and (
        not db_user.get("premium_until") or
        db_user["premium_until"] > datetime.now(timezone.utc)
    )
    prem_str = "Active" if is_prem else "None"
    if is_prem and db_user.get("premium_until"):
        prem_str += f" (until {db_user['premium_until'].strftime('%d.%m.%Y')})"
    role = "Admin" if is_admin(user_id) else ("Premium" if is_prem else "User")

    await callback.message.answer(
        f"**Your Profile**\n\n"
        f"Name: {callback.from_user.first_name or 'N/A'}\n"
        f"Role: {role}\n"
        f"Tokens: {db_user.get('tokens', 0)}\n"
        f"Total generations: {db_user.get('total_gens', 0)}\n"
        f"Premium: {prem_str}",
        parse_mode="Markdown",
    )
    await callback.answer()


@dp.callback_query(F.data == "cb_tokens")
async def cb_tokens(callback: types.CallbackQuery):
    """Show token packages to buy."""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text=f"{pkg['tokens']} Tokens — {pkg['stars']} Stars",
                callback_data=f"buy_{key}",
            )]
            for key, pkg in TOKEN_PACKAGES.items()
        ]
    )
    await callback.message.answer(
        "**Buy Tokens**\n\n"
        "Image generation: 1 token\n"
        "Video generation: 37 tokens\n\n"
        "Choose a package:",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    await callback.answer()


@dp.callback_query(F.data.startswith("buy_tokens_"))
async def cb_buy_tokens(callback: types.CallbackQuery):
    """Send Stars invoice for selected token package."""
    key = callback.data.replace("buy_", "")
    pkg = TOKEN_PACKAGES.get(key)
    if not pkg:
        await callback.answer("Package not found", show_alert=True)
        return

    await callback.message.answer_invoice(
        title=pkg["label"],
        description=f"{pkg['tokens']} generation tokens for Angel Arena",
        payload=key,
        currency="XTR",
        prices=[LabeledPrice(label=pkg["label"], amount=pkg["stars"])],
    )
    await callback.answer()


@dp.callback_query(F.data == "cb_premium")
async def cb_premium(callback: types.CallbackQuery):
    """Send Stars invoice for Premium."""
    await callback.message.answer_invoice(
        title="Premium 30 Days",
        description="Unlimited generations + priority queue for 30 days",
        payload="premium_30",
        currency="XTR",
        prices=[LabeledPrice(label="Premium 30 Days", amount=PREMIUM_PRICE)],
    )
    await callback.answer()


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

    # Ensure user exists in DB before processing payment
    await db.get_or_create_user(
        user_id,
        username=message.from_user.username or "",
        first_name=message.from_user.first_name or "",
    )

    if payload in TOKEN_PACKAGES:
        pkg = TOKEN_PACKAGES[payload]
        await db.add_tokens(user_id, pkg["tokens"])
        await message.answer(
            f"+{pkg['tokens']} tokens added!\n"
            f"Paid: {stars} Stars",
            parse_mode="Markdown",
        )
    elif payload == "premium_30":
        await db.set_premium(user_id, days=30)
        await message.answer(
            "**Premium activated** for 30 days!\n"
            f"Paid: {stars} Stars\n\n"
            "Unlimited generations and priority queue.",
            parse_mode="Markdown",
        )
    elif payload == "quick_image":
        await db.add_tokens(user_id, 1)
        await message.answer(
            "+1 token added! You can now generate.\n"
            f"Paid: {stars} Stars",
            parse_mode="Markdown",
        )
    elif payload == "quick_video":
        await db.add_tokens(user_id, TOKEN_COST_KENPECHI)
        await message.answer(
            f"+{TOKEN_COST_KENPECHI} tokens added! You can now generate a video.\n"
            f"Paid: {stars} Stars",
            parse_mode="Markdown",
        )
    else:
        logger.warning("Unknown payment payload: %s", payload)
        await message.answer("Payment received!")


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
    """Admin: remove user from database: /delete USER_ID"""
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

    if await db.delete_user(target_id):
        await message.answer(f"✅ Пользователь {target_id} удалён из базы")
    else:
        await message.answer(f"❌ Пользователь {target_id} не найден")


@dp.message(lambda m: m.text and m.text.strip() == "/list")
async def cmd_list(message: types.Message):
    """Admin: list all users from database."""
    if not is_admin(message.from_user.id):
        return

    users = await db.list_all_users()
    if not users:
        await message.answer("Нет зарегистрированных пользователей")
        return

    header = f"👥 <b>Пользователи ({len(users)}):</b>\n\n"

    detail_lines = []
    for u in users:
        uname = u.get("username", "")
        tag = f"@{uname}" if uname else "—"
        tokens = u.get("tokens", 0)
        gens = u.get("total_gens", 0)
        prem = "⭐" if u.get("is_premium") else ""
        detail_lines.append(
            f"• {tag} | ID: <code>{u['telegram_id']}</code> | 💰{tokens} | 🎨{gens} {prem}"
        )

    # Split into chunks of ~3500 chars to stay under Telegram 4096 char limit
    MAX_CHUNK = 3500
    chunks = []
    current_chunk = header
    for line in detail_lines:
        if len(current_chunk) + len(line) + 1 > MAX_CHUNK:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += line + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk)

    for chunk in chunks:
        await message.answer(chunk, parse_mode="HTML")


@dp.message(lambda m: m.text and m.text.strip() == "/users")
async def cmd_legacy(message: types.Message):
    """Redirect /users to /list."""
    if not is_admin(message.from_user.id):
        return
    await cmd_list(message)


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
