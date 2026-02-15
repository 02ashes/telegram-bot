"""Configuration — uses environment variables for production (Railway)."""

import os

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8286861040:AAEVN-VzT_jDu2krXzbSbuU-gmlgaU52-5Q")

# RunPod
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "rpa_P77APDTOFJG4BMDST9YKG3HR732Y8ULILWDL1BEM1q2gkw")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "hseftfihe7ikxk")

# ComfyUI on RunPod
COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", f"https://{RUNPOD_POD_ID}-8188.proxy.runpod.net")

# Bot behavior
IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "300"))

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.environ.get("PORT", "8080"))

# WebApp URL — set automatically by Railway or manually
WEBAPP_URL = os.environ.get("WEBAPP_URL", "")
