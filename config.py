"""Configuration — uses environment variables for production (Railway)."""

import os

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# RunPod Serverless
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "6eoat0459ga8g5")
RUNPOD_KENPECHI_ENDPOINT_ID = os.environ.get("RUNPOD_KENPECHI_ENDPOINT_ID", "xpk6atr8n1ahmc")
RUNPOD_V9_ENDPOINT_ID = os.environ.get("RUNPOD_V9_ENDPOINT_ID", "ewzgy1zj22j1s5")

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.environ.get("PORT", "8080"))

# WebApp URL — set automatically by Railway or manually
WEBAPP_URL = os.environ.get("WEBAPP_URL", "")

# Admin — only this Telegram user can manage invite codes
ADMIN_TELEGRAM_ID = int(os.environ.get("ADMIN_TELEGRAM_ID", "1946394239"))

# Database (PostgreSQL on Railway)
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# API Keys (LLM)
XAI_API_KEY = os.environ.get("XAI_API_KEY", "xai-d2Qw3J8Xjy0pTH6zuchwG0oioTKOMj5kIRblNtnEyMeyfQsOygXVZZ2okRDYVSMrAcUF2V9yKDHa0DXv")
XAI_API_URL = os.environ.get("XAI_API_URL", "https://api.x.ai/v1")

AUTOPROMPT_ENABLED = os.environ.get("AUTOPROMPT_ENABLED", "true").lower() == "true"

# TTS (xAI Voice API)
TTS_DEFAULT_VOICE_ID = os.environ.get("TTS_DEFAULT_VOICE_ID", "htep5zqnavbz")
