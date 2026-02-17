# Custom ComfyUI Worker for RunPod Serverless

Custom Docker image extending `runpod/worker-comfyui:5.5.1-base` with:
- **VHS_VideoCombine** video output support (patched handler)
- VideoHelperSuite, KJNodes, rgthree, MMAudio custom nodes
- All Python dependencies baked in

## Quick Start

### 1. Build & Push Docker Image

```bash
cd telegram_bot/runpod
docker build -t 02ashes/worker-comfyui:wan-video .
docker push 02ashes/worker-comfyui:wan-video
```

### 2. Setup Network Volume

Create a temporary GPU pod on RunPod with your Network Volume, then:

```bash
bash setup_volume.sh
```

This downloads ~60GB of models. Takes 15-30 min on a fast connection.
**Only needs to be run once.** The volume persists after stopping the pod.

### 3. Create RunPod Serverless Template

In [RunPod Console](https://www.runpod.io/console/serverless):
1. **Template** → Create New
   - Docker Image: `02ashes/worker-comfyui:wan-video`
   - Container Disk: `20 GB`
   - Volume Mount: attach your Network Volume
2. **Endpoint** → Create with your template
   - Min Workers: 0 (auto-scale)
   - Max Workers: 1-3 (depends on budget)
   - GPU: RTX 4090 or A6000 recommended

### 4. Set Environment Variables (Railway)

```
TELEGRAM_BOT_TOKEN=your_token
RUNPOD_API_KEY=your_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

## File Structure

```
runpod/
├── Dockerfile              # Extends base worker with custom nodes
├── handler.py              # Patched: also collects VHS "gifs" output
├── extra_model_paths.yaml  # Maps Network Volume model dirs
├── setup_volume.sh         # One-time model download script
└── README.md               # This file
```

## Why a Custom Handler?

The stock `handler.py` from `runpod-workers/worker-comfyui` only processes
`node_output["images"]`. But `VHS_VideoCombine` outputs under the `"gifs"` key.
Without the patch, **video files are silently dropped**.
