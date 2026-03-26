"""Configuration — uses environment variables for production (Railway)."""

import os

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# RunPod Serverless
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "6eoat0459ga8g5")
RUNPOD_KENPECHI_ENDPOINT_ID = os.environ.get("RUNPOD_KENPECHI_ENDPOINT_ID", "xpk6atr8n1ahmc")

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.environ.get("PORT", "8080"))

# WebApp URL — set automatically by Railway or manually
WEBAPP_URL = os.environ.get("WEBAPP_URL", "")

# Admin — only this Telegram user can manage invite codes
ADMIN_TELEGRAM_ID = int(os.environ.get("ADMIN_TELEGRAM_ID", "1946394239"))

# Database (PostgreSQL on Railway)
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# SiliconFlow API (Auto-Prompt Enhancement)
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-hamcvdnbovzdvstidyuiyjpuqlgygjhhfbbsshgjafydtrxm")
SILICONFLOW_API_URL = os.environ.get(
    "SILICONFLOW_API_URL", "https://api.siliconflow.com/v1"
)
AUTOPROMPT_ENABLED = os.environ.get("AUTOPROMPT_ENABLED", "true").lower() == "true"
