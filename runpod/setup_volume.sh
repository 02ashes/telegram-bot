#!/bin/bash
# =============================================================================
# Network Volume Setup Script for RunPod Serverless
# Run ONCE on a temporary GPU pod with your Network Volume attached.
#
# Usage:
#   1. Create a pod on RunPod with your Network Volume mounted
#   2. Upload this script to the pod
#   3. Run: HF_TOKEN=hf_xxx bash setup_volume.sh
#   4. Wait for all downloads to complete (~60GB total)
#   5. Stop and delete the pod (the volume persists)
#
# Based on user's verified working pod setup.
# =============================================================================

set +e  # Don't stop on errors — continue downloading remaining models

VOLUME_PATH="${VOLUME_PATH:-/workspace}"
MODELS_DIR="${VOLUME_PATH}/models"

# HuggingFace token for gated/NSFW repos
HF_TOKEN="${HF_TOKEN:-}"

echo "============================================"
echo "  RunPod Network Volume Setup"
echo "  Target: ${VOLUME_PATH}"
echo "============================================"

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Some downloads may fail."
    echo "Usage: HF_TOKEN=hf_xxx bash setup_volume.sh"
fi

# Check that volume is mounted
if [ ! -d "$VOLUME_PATH" ]; then
    echo "ERROR: Network Volume not found at ${VOLUME_PATH}"
    echo "Make sure you have a Network Volume attached!"
    exit 1
fi

# Create directory structure
echo ""
echo "--- Creating directory structure ---"
mkdir -p "${MODELS_DIR}/diffusion_models"
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/clip"
mkdir -p "${MODELS_DIR}/vae"
mkdir -p "${MODELS_DIR}/loras"
mkdir -p "${MODELS_DIR}/unet"
mkdir -p "${MODELS_DIR}/mmaudio"
echo "Done."

# Helper: download with HF token
download() {
    local url="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        echo "  [SKIP] $(basename "$dest") already exists"
        return 0
    fi

    echo "  [DOWNLOAD] $(basename "$dest") ..."
    if [ -n "$HF_TOKEN" ]; then
        wget -q --show-progress --header="Authorization: Bearer ${HF_TOKEN}" -O "$dest" "$url" || {
            echo "  [RETRY] Failed, retrying in 3s..."
            sleep 3
            wget -q --show-progress --header="Authorization: Bearer ${HF_TOKEN}" -O "$dest" "$url" || {
                echo "  [FAILED] $(basename "$dest") — skipping"
                rm -f "$dest"
                return 1
            }
        }
    else
        wget -q --show-progress -O "$dest" "$url" || {
            echo "  [RETRY] Failed, retrying in 3s..."
            sleep 3
            wget -q --show-progress -O "$dest" "$url" || {
                echo "  [FAILED] $(basename "$dest") — skipping"
                rm -f "$dest"
                return 1
            }
        }
    fi
}

# =============================================================================
# [1] WAN 2.2 Remix NSFW UNET Models
# Source: FX-FeiHou/wan2.2-Remix (original author)
# Using fp8 versions (14.3 GB each instead of 28.6 GB fp16)
# =============================================================================
echo ""
echo "--- [1/6] WAN 2.2 Remix NSFW UNET Models (fp8) ---"

download \
    "https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v2.1.safetensors" \
    "${MODELS_DIR}/diffusion_models/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v2.1.safetensors"

download \
    "https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v2.1.safetensors" \
    "${MODELS_DIR}/diffusion_models/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v2.1.safetensors"

# =============================================================================
# [2] CLIP Text Encoder (text_encoders/)
# Source: Osrivers (verified working)
# =============================================================================
echo ""
echo "--- [2/6] CLIP Text Encoder ---"

download \
    "https://huggingface.co/Osrivers/nsfw_wan_umt5-xxl_fp8_scaled.safetensors/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" \
    "${MODELS_DIR}/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors"

# =============================================================================
# [3] VAE
# Source: Comfy-Org/Wan_2.2_ComfyUI_Repackaged
# =============================================================================
echo ""
echo "--- [3/6] VAE ---"

download \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    "${MODELS_DIR}/vae/wan_2.1_vae.safetensors"

# =============================================================================
# [4] WAN 2.2 NSFW Action LoRAs (video generation)
# Source: tamin-akin/wan2.2-nsfw-lora
# Action-specific LoRAs for realistic NSFW video interactions
# =============================================================================
echo ""
echo "--- [4/8] WAN 2.2 NSFW Action LoRAs ---"

# POV Blowjob
download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/pov-blowjob-i2v-v1.2.safetensors" \
    "${MODELS_DIR}/loras/pov-blowjob-i2v-v1.2.safetensors"

# Deepthroat (high + low noise for dual sampler)
download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/jfj-deepthroat-W22-I2V-HN.safetensors" \
    "${MODELS_DIR}/loras/jfj-deepthroat-W22-I2V-HN.safetensors"

download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/jfj-deepthroat-W22-I2V-LN.safetensors" \
    "${MODELS_DIR}/loras/jfj-deepthroat-W22-I2V-LN.safetensors"

# Front doggy
download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/front_doggy_plow_v1_1_wan.safetensors" \
    "${MODELS_DIR}/loras/front_doggy_plow_v1_1_wan.safetensors"

# POV Missionary (high + low noise)
download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/pov-missionary-i2v-high-v1.0.safetensors" \
    "${MODELS_DIR}/loras/pov-missionary-i2v-high-v1.0.safetensors"

download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/pov-missionary-i2v-low-v1.0.safetensors" \
    "${MODELS_DIR}/loras/pov-missionary-i2v-low-v1.0.safetensors"

# Side sex
download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/side-sex-i2v-v10.safetensors" \
    "${MODELS_DIR}/loras/side-sex-i2v-v10.safetensors"

# Reverse cowgirl (high + low noise)
download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/wan22.r3v3rs3_c0wg1rl-14b-High-i2v_e70.safetensors" \
    "${MODELS_DIR}/loras/wan22.r3v3rs3_c0wg1rl-14b-High-i2v_e70.safetensors"

download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/wan22.r3v3rs3_c0wg1rl-14b-Low-i2v_e70.safetensors" \
    "${MODELS_DIR}/loras/wan22.r3v3rs3_c0wg1rl-14b-Low-i2v_e70.safetensors"

# Fingering (high + low noise)
download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/fingering-high-v1.0.safetensors" \
    "${MODELS_DIR}/loras/fingering-high-v1.0.safetensors"

download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/fingering-low-v1.0.safetensors" \
    "${MODELS_DIR}/loras/fingering-low-v1.0.safetensors"

# Nipple stroke (high + low noise)
download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/nipple_stroke_WAN22_I2V_v1_high_noise.safetensors" \
    "${MODELS_DIR}/loras/nipple_stroke_WAN22_I2V_v1_high_noise.safetensors"

download \
    "https://huggingface.co/tamin-akin/wan2.2-nsfw-lora/resolve/main/nipple_stroke_WAN22_I2V_v1_low_noise.safetensors" \
    "${MODELS_DIR}/loras/nipple_stroke_WAN22_I2V_v1_low_noise.safetensors"

# =============================================================================
# [5] Flux Fill Models (for inpainting)
# =============================================================================
echo ""
echo "--- [5/8] Flux Fill Models ---"

download \
    "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors" \
    "${MODELS_DIR}/unet/flux1-fill-dev.safetensors"

download \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
    "${MODELS_DIR}/clip/clip_l.safetensors"

download \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors" \
    "${MODELS_DIR}/clip/t5xxl_fp8_e4m3fn.safetensors"

download \
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors" \
    "${MODELS_DIR}/vae/ae.safetensors"

# =============================================================================
# [6] Flux 2 Klein 9B (image editing)
# Source: black-forest-labs/FLUX.2-klein-9B (gated — requires HF_TOKEN)
# Reuses same CLIP encoders and VAE as Flux Fill above
# =============================================================================
echo ""
echo "--- [6/8] Flux 2 Klein 9B ---"

download \
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors" \
    "${MODELS_DIR}/diffusion_models/flux-2-klein-9b.safetensors"

# Qwen3 8B text encoder (required by Flux 2 Klein, NOT the same as Flux 1 encoders)
download \
    "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors" \
    "${MODELS_DIR}/text_encoders/qwen_3_8b.safetensors"

# Flux 2 VAE (different from Flux 1 ae.safetensors)
download \
    "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors" \
    "${MODELS_DIR}/vae/flux2-vae.safetensors"

# DepthAnything V2 (body shape preservation for Klein advanced edit)
# Small model (~25MB) — auto-downloaded by node, but pre-download for cold starts
mkdir -p "${MODELS_DIR}/depthanything"
download \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.safetensors" \
    "${MODELS_DIR}/depthanything/depth_anything_v2_vits.safetensors"

# =============================================================================
# [6] MMAudio (ALL required files — 5 total)
# Sources: cloud19/NSFW_MMaudio + kijai/MMAudio_safetensors
# =============================================================================
echo ""
echo "--- [7/8] MMAudio (5 files) ---"

download \
    "https://huggingface.co/cloud19/NSFW_MMaudio/resolve/main/nsfw_gold_8.5k_final.pth" \
    "${MODELS_DIR}/mmaudio/mmaudio_large_44k_nsfw_gold_8.5k_final.pth"

download \
    "https://huggingface.co/kijai/MMAudio_safetensors/resolve/main/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors" \
    "${MODELS_DIR}/mmaudio/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"

download \
    "https://huggingface.co/kijai/MMAudio_safetensors/resolve/main/mmaudio_synchformer_fp32.safetensors" \
    "${MODELS_DIR}/mmaudio/mmaudio_synchformer_fp32.safetensors"

download \
    "https://huggingface.co/kijai/MMAudio_safetensors/resolve/main/mmaudio_vae_44k_fp16.safetensors" \
    "${MODELS_DIR}/mmaudio/mmaudio_vae_44k_fp16.safetensors"

download \
    "https://huggingface.co/kijai/MMAudio_safetensors/resolve/main/bigvgan_v2_44khz_128band_512x.safetensors" \
    "${MODELS_DIR}/mmaudio/bigvgan_v2_44khz_128band_512x.safetensors"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Models installed:"
find "${MODELS_DIR}" -type f \( -name "*.safetensors" -o -name "*.pth" \) | sort | while read f; do
    size=$(du -h "$f" | cut -f1)
    echo "  ${size}  $(basename "$f")"
done
echo ""
echo "Total volume usage:"
du -sh "${VOLUME_PATH}"
echo ""
echo "You can now stop this pod. The Network Volume will persist."
echo "Use this volume with your RunPod Serverless template."
