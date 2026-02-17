"""Configuration — uses environment variables for production (Railway)."""

import os

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# RunPod Serverless
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.environ.get("PORT", "8080"))

# WebApp URL — set automatically by Railway or manually
WEBAPP_URL = os.environ.get("WEBAPP_URL", "")

