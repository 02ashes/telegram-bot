# ============================================================
# Custom ComfyUI Worker for RunPod Serverless
# Base: runpod/worker-comfyui (latest base, no models)
# Adds: custom nodes, patched handler for video output
# Models: loaded from Network Volume at runtime
# ============================================================
FROM runpod/worker-comfyui:5.7.0-base AS base

# ---- Update ComfyUI to latest (need Flux 2 Klein / Qwen3 support) ----
WORKDIR /comfyui
ARG CACHEBUST=1
RUN git fetch origin && git reset --hard origin/master && \
    pip install -r requirements.txt --quiet 2>/dev/null || true

# ---- Install custom nodes ----
WORKDIR /comfyui/custom_nodes

# VideoHelperSuite (VHS_VideoCombine for MP4 output)
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt --quiet

# KJNodes (INTConstant etc.)
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

# NOTE: TeaCache custom node removed — using ComfyUI's built-in EasyCache instead

# ---- Install additional Python deps ----
# Matching user's verified working setup
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

# NOTE: comfy_kitchen cuda/triton backends require CUDA Runtime 13.0+
# Base image has CUDA 12.6, so we stay on cu128 (eager backend, ~10% slower but works fine)
# When RunPod updates base images to CUDA 13.0, uncomment the line below:
# RUN pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Install Triton (may still help with some operations)
RUN pip install --no-cache-dir triton

# ---- Remove ComfyUI-Manager (saves ~2 min on cold start) ----
RUN rm -rf /comfyui/custom_nodes/ComfyUI-Manager

# ---- Custom extra_model_paths.yaml for Network Volume ----
WORKDIR /comfyui
COPY extra_model_paths.yaml ./

# ---- Patched handler that supports VHS video output ----
COPY handler.py /handler.py

# ---- Patch /start.sh: add gpu-only + fast flags ----
RUN sed -i 's|python -u /comfyui/main.py|python -u /comfyui/main.py --gpu-only --fast|g' /start.sh

WORKDIR /
CMD ["/start.sh"]
