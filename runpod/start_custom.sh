#!/bin/bash
echo "=== Starting ComfyUI (optimized, no lowvram) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable ComfyUI-Manager to skip 2-min registry fetch
export COMFYUI_MANAGER_DISABLED=1

cd /comfyui && python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch \
    --gpu-only \
    --fast \
    --use-pytorch-cross-attention &

echo "Waiting for ComfyUI to be ready..."
until curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; do
    sleep 1
done
echo "ComfyUI is ready!"

cd / && python handler.py
