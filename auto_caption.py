"""
Auto-captioning script for LoRA training datasets.
Uses BLIP-large for image descriptions.

Usage:
    python auto_caption.py --dir "AI TRAIN/Misu" --trigger "misu"
    python auto_caption.py --dir "AI TRAIN/Anya" --trigger "anya" --overwrite
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_model(device="cuda"):
    """Load BLIP-large model for captioning."""
    model_id = "Salesforce/blip-image-captioning-large"
    print(f"Loading {model_id}...")

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)

    print("Model loaded!")
    return processor, model


def caption_image(processor, model, image_path, device="cuda"):
    """Generate a detailed caption for a single image."""
    image = Image.open(image_path).convert("RGB")

    # Conditional captioning with prompt for more detail
    text = "a photo of"
    inputs = processor(image, text, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            repetition_penalty=1.5,
        )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def clean_caption_for_lora(raw_caption: str, trigger: str) -> str:
    """Clean up auto-generated caption for LoRA training."""
    caption = raw_caption.strip()

    # Remove "a photo of" prefix if present
    for prefix in ["a photo of ", "a photograph of ", "a picture of "]:
        if caption.lower().startswith(prefix):
            caption = caption[len(prefix):]
            break

    # Remove identity phrases that LoRA should learn itself
    for phrase in ["a young woman", "a woman", "a girl"]:
        if caption.lower().startswith(phrase):
            caption = caption[len(phrase):].lstrip(" ,")
            caption = "a woman " + caption
            break

    # Build final: trigger + description
    return f"{trigger}, {caption}"


def main():
    parser = argparse.ArgumentParser(description="Auto-caption images for LoRA training")
    parser.add_argument("--dir", required=True, help="Directory with images")
    parser.add_argument("--trigger", required=True, help="Trigger word (e.g. 'misu')")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    image_dir = Path(args.dir)
    images = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images in {image_dir}")
    processor, model = load_model(args.device)

    for i, img_path in enumerate(images):
        txt_path = img_path.with_suffix(".txt")

        if txt_path.exists() and not args.overwrite:
            existing = txt_path.read_text().strip()
            if len(existing) > len(args.trigger) + 5:
                print(f"[{i+1}/{len(images)}] SKIP {img_path.name}")
                continue

        print(f"[{i+1}/{len(images)}] {img_path.name}...", end=" ")

        try:
            raw = caption_image(processor, model, img_path, args.device)
            caption = clean_caption_for_lora(raw, args.trigger)
            txt_path.write_text(caption, encoding="utf-8")
            print(f"→ {caption[:80]}...")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n✅ Done! {len(images)} captions saved.")
    print(f"⚠️  ОБЯЗАТЕЛЬНО проверь каждый .txt вручную!")


if __name__ == "__main__":
    main()
