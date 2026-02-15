"""Configuration — uses environment variables for production (Railway)."""

import os

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8286861040:AAEVN-VzT_jDu2krXzbSbuU-gmlgaU52-5Q")

# RunPod Serverless
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "rpa_P77APDTOFJG4BMDST9YKG3HR732Y8ULILWDL1BEM1q2gkw")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "wfawb9z9cb79ws")

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.environ.get("PORT", "8080"))

# WebApp URL — set automatically by Railway or manually
WEBAPP_URL = os.environ.get("WEBAPP_URL", "")
