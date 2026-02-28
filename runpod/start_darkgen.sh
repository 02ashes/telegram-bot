#!/bin/bash
# start_darkgen.sh — Dark Generate endpoint (baked models, no Network Volume)
# 1. Launch ComfyUI in background
# 2. Wait for API readiness
# 3. Start RunPod handler in foreground

set -e

echo "worker-darkgen: Starting ComfyUI (baked models, no Network Volume)"

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /comfyui && python -u main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch \
    --gpu-only \
    --use-sage-attention \
    --fast &

COMFYUI_PID=$!

echo "worker-darkgen: Starting RunPod Handler"

# Wait for ComfyUI API to be ready (max ~60s)
MAX_ATTEMPTS=120
ATTEMPT=0
until curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "worker-darkgen - ComfyUI failed to start after ${MAX_ATTEMPTS} attempts"
        kill $COMFYUI_PID 2>/dev/null || true
        exit 1
    fi
    sleep 0.5
done

echo "worker-darkgen - ComfyUI API is ready"

cd / && python -u handler.py

# If handler exits, also kill ComfyUI
kill $COMFYUI_PID 2>/dev/null || true
