"""Telegram Inpaint Bot — aiogram 3 + FastAPI server."""

import asyncio
import base64
import io
import logging
import os
import sys

import uvicorn
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    WebAppInfo,
)
from fastapi import FastAPI, Request
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
    # Local development: use ngrok
    try:
        from pyngrok import ngrok, conf
        NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "39gBaAKkXfUD2hnB4yfceuXsIx7_gN4xKA1d8ZHWfsmmNt9e")
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        logger.info("Starting ngrok tunnel on port %d...", config.SERVER_PORT)
        tunnel = ngrok.connect(config.SERVER_PORT, "http")
        WEBAPP_URL = tunnel.public_url.replace("http://", "https://")
        logger.info("✅ ngrok tunnel: %s", WEBAPP_URL)
    except ImportError:
        logger.error("WEBAPP_URL not set and pyngrok not installed!")
        sys.exit(1)
else:
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
    if not auth.is_user_registered(user["id"]):
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


@app.post("/api/inpaint")
async def api_inpaint(request: Request):
    """Run inpainting via RunPod Serverless."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        # Parse JSON body
        body = await request.json()
        image_b64 = body.get("image", "")
        mask_b64 = body.get("mask", "")
        prompt = body.get("prompt", "")
        negative = body.get("negative", "blurry, ugly, deformed, watermark, text, low quality, cartoon")
        cfg = float(body.get("cfg", 3.5))
        steps = int(body.get("steps", 25))

        if not image_b64 or not mask_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image, mask, or prompt"})

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
            return JSONResponse(
                status_code=500,
                content={"error": "Inpainting failed. Check RunPod logs."},
            )

        # Return image as base64
        result_b64 = base64.b64encode(result_bytes).decode("utf-8")
        return JSONResponse(content={"image": result_b64})

    except Exception as e:
        logger.exception("Inpaint error")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/api/video")
async def api_video(request: Request):
    """Generate video via WAN 2.2 I2V on RunPod Serverless."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
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

        if not image_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image or prompt"})

        image_bytes = base64.b64decode(image_b64)

        logger.info(
            "Video request: prompt=%s, frames=%d, fps=%d, %dx%d, audio=%s, action=%s, shift=%.1f, cfg=%.1f/%.1f",
            prompt[:60], frames, fps, width, height, audio_enabled, action, shift, cfg_high, cfg_low,
        )

        result_bytes = await comfyui_api.run_video(
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
        )

        if result_bytes is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Video generation failed. Check RunPod logs."},
            )

        result_b64 = base64.b64encode(result_bytes).decode("utf-8")
        return JSONResponse(content={"video": result_b64})

    except Exception as e:
        logger.exception("Video error")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/api/video/cancel/{job_id}")
async def api_video_cancel(job_id: str, request: Request):
    """Cancel a running video generation job on RunPod."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
        success = await comfyui_api.cancel_job(job_id)
        if success:
            return JSONResponse(content={"status": "cancelled", "job_id": job_id})
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to cancel job"},
            )
    except Exception as e:
        logger.exception("Cancel error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/image-edit")
async def api_image_edit(request: Request):
    """Edit image via Flux 2 Klein 9B on RunPod Serverless."""
    user = await require_auth(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    try:
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

        if not image_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image or prompt"})

        image_bytes = base64.b64decode(image_b64)
        image2_bytes = base64.b64decode(image2_b64) if image2_b64 else None

        logger.info(
            "Image edit request: prompt=%s, denoise=%.2f, has_ref=%s, lora=%s",
            prompt[:60], denoise, image2_bytes is not None, lora_name or "none",
        )

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
            return JSONResponse(
                status_code=500,
                content={"error": "Image editing failed. Check RunPod logs."},
            )

        result_b64 = base64.b64encode(result_bytes).decode("utf-8")
        return JSONResponse(content={"image": result_b64})

    except Exception as e:
        logger.exception("Image edit error")
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
    try:
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

        if not image_b64 or not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing image or prompt"})

        image_bytes = base64.b64decode(image_b64)
        image2_bytes = base64.b64decode(image2_b64) if image2_b64 else None

        logger.info(
            "Dark %s request: prompt=%s, denoise=%.2f, quality=%s",
            dark_mode, prompt[:60], denoise, quality,
        )

        if dark_mode == "generate":
            result_bytes = await comfyui_api.run_dark_generate(
                image_bytes=image_bytes,
                prompt=prompt,
                negative=negative,
                denoise=denoise,
                steps=steps,
                cfg=cfg,
                quality=quality,
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
            return JSONResponse(
                status_code=500,
                content={"error": "Dark Beast editing failed. Check RunPod logs."},
            )

        result_b64 = base64.b64encode(result_bytes).decode("utf-8")
        return JSONResponse(content={"image": result_b64})

    except Exception as e:
        logger.exception("Dark edit error")
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


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Handle /start — show WebApp only to registered users."""
    user_id = message.from_user.id

    if not auth.is_user_registered(user_id) and not is_admin(user_id):
        await message.answer(
            "🔒 **Доступ ограничен**\n\n"
            "Чтобы пользоваться ботом, введи инвайт-код:\n"
            "`/invite ВАШ_КОД`",
            parse_mode="Markdown",
        )
        return

    # Auto-register admin if not registered
    if is_admin(user_id) and not auth.is_user_registered(user_id):
        users = auth.list_users()
        if str(user_id) not in users:
            import time as _time
            auth._save_json(auth.USERS_FILE, {
                **users,
                str(user_id): {
                    "username": message.from_user.username or "admin",
                    "registered_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
                    "code_used": "ADMIN",
                },
            })

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


@dp.message(lambda m: m.text and m.text.startswith("/invite"))
async def cmd_invite(message: types.Message):
    """Register with invite code: /invite CODE"""
    user_id = message.from_user.id

    if auth.is_user_registered(user_id):
        await message.answer("✅ Ты уже зарегистрирован! Нажми /start")
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("❌ Использование: `/invite ВАШ_КОД`", parse_mode="Markdown")
        return

    code = parts[1].strip()
    username = message.from_user.username or message.from_user.first_name or str(user_id)

    if auth.validate_and_use_code(code, user_id, username):
        logger.info("User %s (%s) registered with code %s", user_id, username, code)
        await message.answer(
            "✅ **Добро пожаловать!**\n\n"
            "Инвайт-код принят. Нажми /start чтобы открыть редактор.",
            parse_mode="Markdown",
        )
    else:
        await message.answer("❌ Неверный или уже использованный код.")


@dp.message(lambda m: m.text and m.text.startswith("/generate"))
async def cmd_generate(message: types.Message):
    """Admin: create invite code: /generate CODE"""
    if not is_admin(message.from_user.id):
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Использование: `/generate КОД`", parse_mode="Markdown")
        return

    code = parts[1].strip()
    if auth.create_code(code):
        await message.answer(f"✅ Инвайт-код создан: `{code}`", parse_mode="Markdown")
    else:
        await message.answer(f"❌ Код `{code}` уже существует", parse_mode="Markdown")


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


@dp.message(lambda m: m.text and m.text.strip() == "/users")
async def cmd_users(message: types.Message):
    """Admin: list registered users."""
    if not is_admin(message.from_user.id):
        return

    users = auth.list_users()
    if not users:
        await message.answer("Нет зарегистрированных пользователей")
        return

    lines = ["👥 **Пользователи:**\n"]
    for uid, info in users.items():
        lines.append(f"• `{uid}` — @{info.get('username', '?')} (код: {info.get('code_used', '?')})")
    await message.answer("\n".join(lines), parse_mode="Markdown")


@dp.message(lambda m: m.text and m.text.strip() == "/codes")
async def cmd_codes(message: types.Message):
    """Admin: list invite codes."""
    if not is_admin(message.from_user.id):
        return

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

    # Run bot and server concurrently
    await asyncio.gather(
        dp.start_polling(bot),
        server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
