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


# ============================================================
# Telegram Bot (aiogram 3)
# ============================================================
bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Handle /start command."""
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
        "👋 **Привет! Я Inpaint-бот.**\n\n"
        "Нажми кнопку ниже, чтобы открыть редактор:\n"
        "1. 📷 Загрузи фото\n"
        "2. 🖌️ Нарисуй маску кистью\n"
        "3. ✏️ Напиши промпт\n"
        "4. 🚀 Нажми «Генерировать»\n\n"
        "💡 *RunPod запустится автоматически при генерации*",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


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
