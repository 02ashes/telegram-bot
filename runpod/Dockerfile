# ============================================================
# Custom ComfyUI Worker for RunPod Serverless
# Base: runpod/worker-comfyui (latest base, no models)
# Adds: custom nodes, patched handler for video output
# Models: loaded from Network Volume at runtime
# ============================================================
FROM runpod/worker-comfyui:5.7.0-base AS base

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
    toml

# ---- Custom extra_model_paths.yaml for Network Volume ----
WORKDIR /comfyui
COPY extra_model_paths.yaml ./

# ---- Patched handler that supports VHS video output ----
COPY handler.py /handler.py

WORKDIR /
CMD ["/start.sh"]
