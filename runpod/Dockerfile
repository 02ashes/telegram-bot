# ============================================================
# Custom ComfyUI Worker for RunPod Serverless — H100 80GB HBM3
# Base: nvidia/cuda:13.0.1-runtime-ubuntu22.04 (gcc for Triton JIT)
# PyTorch: cu128 (cu130 has cuBLAS bug on H100)
# Models: loaded from Network Volume at runtime
#
# Performance stack:
#   1. SageAttention (PyPI v1.0.6) → Triton JIT attention (gcc required)
#   2. --use-sage-attention → global SageAttention for ALL models
#   3. --fast               → fp16 accumulation + all perf features
#   4. --gpu-only           → zero CPU offload (80GB VRAM)
#   5. comfy_kitchen cuda   → optimized RoPE/quantize ops
#   NOTE: RunPod 'Allowed CUDA versions' must include 13.0
# ============================================================
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

# ---- System packages ----
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip \
    git curl wget \
    gcc g++ \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# ---- Python virtual environment ----
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- PyTorch cu128 ----
# cu130 has cuBLAS bug on H100 (CUBLAS_STATUS_INVALID_VALUE in cublasSgemm)
# cu128 works reliably and bundles its own CUDA libs (runs fine on CUDA 13 runtime)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ---- ComfyUI ----
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN pip install -r requirements.txt --quiet 2>/dev/null || true

# Force-update to latest (CACHEBUST invalidates this layer)
ARG CACHEBUST=1
RUN git fetch origin && git reset --hard origin/master && \
    pip install -r requirements.txt --quiet 2>/dev/null || true

# ============================================================
# CUSTOM NODES
# ============================================================
WORKDIR /comfyui/custom_nodes

# VideoHelperSuite (VHS_VideoCombine for MP4 output)
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt --quiet

# KJNodes (INTConstant, etc.)
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && (pip install -r requirements.txt --quiet 2>/dev/null || true)

# rgthree (Seed node)
RUN git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git && \
    (cd rgthree-comfy && pip install -r requirements.txt --quiet 2>/dev/null || true)

# MMAudio (audio generation for video)
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-MMAudio.git && \
    cd ComfyUI-MMAudio && (pip install -r requirements.txt --quiet 2>/dev/null || true)

# vrgamedevgirl nodes
RUN git clone --depth 1 https://github.com/vrgamegirl19/comfyui-vrgamedevgirl.git && \
    (cd comfyui-vrgamedevgirl && pip install -r requirements.txt --quiet 2>/dev/null || true)

# ComfyUI Essentials (ImageResize+ for Klein advanced edit)
RUN git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git && \
    (cd ComfyUI_essentials && pip install -r requirements.txt --quiet 2>/dev/null || true)

# DepthAnything V2 (body shape preservation for Klein advanced edit)
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-DepthAnythingV2.git && \
    (cd ComfyUI-DepthAnythingV2 && pip install -r requirements.txt --quiet 2>/dev/null || true)

# ============================================================
# PERFORMANCE PACKAGES
# ============================================================
WORKDIR /comfyui

# Core Python deps
RUN pip install --no-cache-dir \
    opencv-python-headless \
    accelerate \
    imageio-ffmpeg \
    ffmpeg-python \
    torchdiffeq \
    open-clip-torch \
    librosa \
    omegaconf \
    toml \
    "diffusers>=0.31.0"

# SageAttention (Triton JIT — gcc installed above)
RUN pip install --no-cache-dir sageattention && \
    python -c "import sageattention; print('SageAttention OK:', sageattention.__file__)"

# RunPod SDK
RUN pip install --no-cache-dir runpod requests websocket-client

# ============================================================
# CONFIG & HANDLER
# ============================================================

# Disable ComfyUI-Manager (saves cold start time)
RUN rm -rf /comfyui/custom_nodes/ComfyUI-Manager

# Extra model paths (Network Volume)
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# ComfyUI-Manager offline config (prevent network fetches)
RUN mkdir -p /comfyui/user/default/ComfyUI-Manager && \
    echo -e "[manager]\nnetwork_mode = offline" > /comfyui/user/default/ComfyUI-Manager/config.ini

# Patched handler (VHS video support)
COPY handler.py /handler.py

# Start script (fix Windows CRLF → LF)
COPY start.sh /start.sh
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh

WORKDIR /
CMD ["/start.sh"]
