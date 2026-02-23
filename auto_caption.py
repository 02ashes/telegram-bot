"""Auto-caption images for LoRA training using BLIP-2 or simple descriptions.

Usage:
    python auto_caption.py --dir "D:\AI TRAIN\Misu" --trigger "misu"

Creates a .txt file for each image with a trigger word + auto-generated caption.
"""

import argparse
import os
from pathlib import Path

# Try to use transformers for auto-captioning
try:
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    HAS_BLIP = True
except ImportError:
    HAS_BLIP = False
    print("⚠️  transformers/torch not installed. Using manual caption mode.")
    print("    Install with: pip install transformers torch pillow")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def auto_caption_blip(image_path: str) -> str:
    """Generate caption using BLIP model."""
    global processor, model
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def manual_caption_template(filename: str) -> str:
    """Generate a basic template caption for manual editing."""
    return "a young woman, portrait photo, looking at camera"


def process_directory(directory: str, trigger_word: str, use_blip: bool = True):
    """Process all images in directory, creating .txt caption files."""
    dir_path = Path(directory)
    images = [f for f in dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

    if not images:
        print(f"❌ No images found in {directory}")
        return

    print(f"📸 Found {len(images)} images in {directory}")
    print(f"🏷️  Trigger word: {trigger_word}")
    print()

    for img_path in sorted(images):
        txt_path = img_path.with_suffix(".txt")

        # Skip if caption already exists
        if txt_path.exists():
            print(f"  ⏭️  {img_path.name} — caption exists, skipping")
            continue

        if use_blip and HAS_BLIP:
            caption = auto_caption_blip(str(img_path))
        else:
            caption = manual_caption_template(img_path.name)

        full_caption = f"{trigger_word}, {caption}"
        txt_path.write_text(full_caption, encoding="utf-8")
        print(f"  ✅ {img_path.name} → {full_caption[:80]}...")

    print()
    print(f"✅ Done! Captions saved as .txt files in {directory}")
    print("📝 Review and edit the .txt files if needed before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-caption images for LoRA training")
    parser.add_argument("--dir", required=True, help="Directory with images")
    parser.add_argument("--trigger", required=True, help="Trigger word (e.g. 'misu')")
    parser.add_argument("--no-blip", action="store_true", help="Skip BLIP, use templates")
    args = parser.parse_args()

    use_blip = HAS_BLIP and not args.no_blip

    if use_blip:
        print("🧠 Loading BLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device)
        print(f"✅ BLIP loaded on {device}")

    process_directory(args.dir, args.trigger, use_blip)
