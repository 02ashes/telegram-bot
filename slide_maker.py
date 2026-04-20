import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Slide Maker — generates dark-themed presentation cards for Telegram posts.

Usage:
    python slide_maker.py

Customize the SLIDES list at the bottom to create your own posts.
Screenshots are auto-inserted with rounded corners and phone-frame styling.
"""

import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Design tokens ──────────────────────────────────────────────
BG_COLOR_TOP = (12, 15, 30)        # dark navy
BG_COLOR_BOT = (5, 5, 15)         # near black
ACCENT = (0, 200, 255)            # electric cyan
ACCENT_DIM = (0, 100, 140)
WHITE = (255, 255, 255)
GRAY = (160, 165, 180)
DARK_CARD = (20, 24, 45, 220)     # semi-transparent card
RED = (255, 80, 80)
GREEN = (80, 255, 120)

SLIDE_W, SLIDE_H = 1080, 1080     # square for Telegram
PADDING = 60
OUTPUT_DIR = Path("slides_output")


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try to load a good font, fall back to default."""
    font_names = [
        "seguisb.ttf" if bold else "segoeui.ttf",  # Windows Segoe UI
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
    ]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    # Try system font paths
    for path in [r"C:\Windows\Fonts", "/usr/share/fonts"]:
        for name in font_names:
            full = os.path.join(path, name)
            if os.path.exists(full):
                try:
                    return ImageFont.truetype(full, size)
                except (OSError, IOError):
                    continue
    return ImageFont.load_default()


def _gradient_bg() -> Image.Image:
    """Create a vertical gradient background."""
    img = Image.new("RGB", (SLIDE_W, SLIDE_H))
    draw = ImageDraw.Draw(img)
    for y in range(SLIDE_H):
        t = y / SLIDE_H
        r = int(BG_COLOR_TOP[0] * (1 - t) + BG_COLOR_BOT[0] * t)
        g = int(BG_COLOR_TOP[1] * (1 - t) + BG_COLOR_BOT[1] * t)
        b = int(BG_COLOR_TOP[2] * (1 - t) + BG_COLOR_BOT[2] * t)
        draw.line([(0, y), (SLIDE_W, y)], fill=(r, g, b))
    return img


def _draw_rounded_rect(draw: ImageDraw.Draw, xy, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _add_grid_overlay(img: Image.Image, spacing=60, alpha=12):
    """Add subtle grid lines."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(0, SLIDE_W, spacing):
        draw.line([(x, 0), (x, SLIDE_H)], fill=(*ACCENT, alpha), width=1)
    for y in range(0, SLIDE_H, spacing):
        draw.line([(0, y), (SLIDE_W, y)], fill=(*ACCENT, alpha), width=1)
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def _add_accent_line(draw: ImageDraw.Draw, y: int, width: int = 3):
    """Draw a horizontal accent line."""
    draw.line([(PADDING, y), (SLIDE_W - PADDING, y)], fill=ACCENT, width=width)


def _insert_screenshot(img: Image.Image, screenshot_path: str,
                       x: int, y: int, max_w: int, max_h: int,
                       corner_radius: int = 20, border_color=ACCENT, border_width: int = 3):
    """Insert a screenshot with rounded corners and border into the slide."""
    scr = Image.open(screenshot_path).convert("RGBA")
    # Scale to fit
    ratio = min(max_w / scr.width, max_h / scr.height)
    new_w, new_h = int(scr.width * ratio), int(scr.height * ratio)
    scr = scr.resize((new_w, new_h), Image.LANCZOS)

    # Create rounded mask
    mask = Image.new("L", (new_w, new_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(0, 0), (new_w - 1, new_h - 1)],
                                 radius=corner_radius, fill=255)

    # Apply mask
    output = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    output.paste(scr, mask=mask)

    # Draw border on the main image
    img_draw = ImageDraw.Draw(img.convert("RGBA") if img.mode != "RGBA" else img)
    _draw_rounded_rect(img_draw,
                       (x - border_width, y - border_width,
                        x + new_w + border_width, y + new_h + border_width),
                       radius=corner_radius + border_width,
                       outline=border_color, width=border_width)

    # Paste screenshot
    img.paste(output, (x, y), output)
    return new_w, new_h


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ══════════════════════════════════════════════════════════════
# SLIDE TYPES
# ══════════════════════════════════════════════════════════════

def make_title_slide(title: str, subtitle: str, output_name: str):
    """Title card with large text."""
    img = _gradient_bg()
    _add_grid_overlay(img)
    draw = ImageDraw.Draw(img)

    title_font = _get_font(72, bold=True)
    sub_font = _get_font(36)

    # Center title
    lines = _wrap_text(title, title_font, SLIDE_W - PADDING * 2)
    total_h = len(lines) * 85
    start_y = (SLIDE_H - total_h) // 2 - 40

    for i, line in enumerate(lines):
        bbox = title_font.getbbox(line)
        w = bbox[2] - bbox[0]
        x = (SLIDE_W - w) // 2
        draw.text((x, start_y + i * 85), line, font=title_font, fill=WHITE)

    # Subtitle
    sub_y = start_y + len(lines) * 85 + 20
    bbox = sub_font.getbbox(subtitle)
    w = bbox[2] - bbox[0]
    draw.text(((SLIDE_W - w) // 2, sub_y), subtitle, font=sub_font, fill=ACCENT)

    # Accent lines
    _add_accent_line(draw, sub_y + 60)
    _add_accent_line(draw, sub_y - 10, width=1)

    _save(img, output_name)


def make_bullets_slide(title: str, bullets: list[str], output_name: str,
                       screenshot_path: str = None):
    """Slide with title + bullet points + optional screenshot."""
    img = _gradient_bg()
    _add_grid_overlay(img)
    draw = ImageDraw.Draw(img)

    title_font = _get_font(52, bold=True)
    bullet_font = _get_font(32)

    # Title
    draw.text((PADDING, PADDING), title, font=title_font, fill=WHITE)
    _add_accent_line(draw, PADDING + 70, width=2)

    # If screenshot, split layout
    if screenshot_path and os.path.exists(screenshot_path):
        # Left: bullets, Right: screenshot
        text_w = SLIDE_W // 2 - PADDING
        scr_x = SLIDE_W // 2 + 20
        scr_y = PADDING + 100
        scr_max_w = SLIDE_W // 2 - PADDING - 20
        scr_max_h = SLIDE_H - PADDING * 2 - 120

        img = img.convert("RGBA")
        _insert_screenshot(img, screenshot_path, scr_x, scr_y, scr_max_w, scr_max_h)
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
    else:
        text_w = SLIDE_W - PADDING * 2

    # Bullets
    y = PADDING + 110
    for bullet in bullets:
        lines = _wrap_text(bullet, bullet_font, text_w - 40)
        # Bullet card background
        card_h = len(lines) * 42 + 24
        card_img = Image.new("RGBA", (text_w, card_h), DARK_CARD)
        card_draw = ImageDraw.Draw(card_img)
        _draw_rounded_rect(card_draw, (0, 0, text_w - 1, card_h - 1),
                          radius=12, outline=ACCENT_DIM, width=1)

        # Paste card
        img.paste(Image.alpha_composite(
            img.crop((PADDING, y, PADDING + text_w, y + card_h)).convert("RGBA"),
            card_img
        ).convert("RGB"), (PADDING, y))

        draw = ImageDraw.Draw(img)
        for j, line in enumerate(lines):
            prefix = "  " if j > 0 else ""
            draw.text((PADDING + 20, y + 12 + j * 42), prefix + line,
                     font=bullet_font, fill=GRAY)

        y += card_h + 16

    _save(img, output_name)


def make_comparison_slide(title: str,
                          bad_label: str, bad_text: str,
                          good_label: str, good_text: str,
                          output_name: str,
                          bad_screenshot: str = None,
                          good_screenshot: str = None):
    """Side-by-side comparison slide with optional screenshots."""
    img = _gradient_bg()
    _add_grid_overlay(img)
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)

    title_font = _get_font(48, bold=True)
    label_font = _get_font(36, bold=True)
    text_font = _get_font(24)

    # Title
    bbox = title_font.getbbox(title)
    w = bbox[2] - bbox[0]
    draw.text(((SLIDE_W - w) // 2, PADDING), title, font=title_font, fill=WHITE)

    half_w = (SLIDE_W - PADDING * 3) // 2
    left_x = PADDING
    right_x = PADDING * 2 + half_w
    start_y = PADDING + 80

    for side, (label, text, color, scr_path, x_pos) in enumerate([
        (bad_label, bad_text, RED, bad_screenshot, left_x),
        (good_label, good_text, GREEN, good_screenshot, right_x),
    ]):
        # Label
        draw.text((x_pos + 10, start_y), label, font=label_font, fill=color)

        # Text card
        card_y = start_y + 55
        lines = _wrap_text(text, text_font, half_w - 30)
        text_block_h = len(lines) * 34 + 20

        card = Image.new("RGBA", (half_w, text_block_h), (*DARK_CARD[:3], 180))
        card_draw = ImageDraw.Draw(card)
        _draw_rounded_rect(card_draw, (0, 0, half_w - 1, text_block_h - 1),
                          radius=12, outline=color + (100,), width=2)

        img.paste(Image.alpha_composite(
            img.crop((x_pos, card_y, x_pos + half_w, card_y + text_block_h)),
            card
        ), (x_pos, card_y))

        draw = ImageDraw.Draw(img)
        for j, line in enumerate(lines):
            draw.text((x_pos + 15, card_y + 10 + j * 34), line,
                     font=text_font, fill=GRAY)

        # Screenshot below text
        if scr_path and os.path.exists(scr_path):
            scr_y = card_y + text_block_h + 20
            max_scr_h = SLIDE_H - scr_y - PADDING
            _insert_screenshot(img, scr_path, x_pos, scr_y,
                             half_w, max_scr_h, border_color=color)

    _save(img.convert("RGB"), output_name)


def make_cta_slide(title: str, bot_username: str, subtitle: str, output_name: str):
    """Call-to-action slide promoting the bot."""
    img = _gradient_bg()
    _add_grid_overlay(img)
    draw = ImageDraw.Draw(img)

    title_font = _get_font(64, bold=True)
    bot_font = _get_font(48, bold=True)
    sub_font = _get_font(30)

    # Title
    bbox = title_font.getbbox(title)
    w = bbox[2] - bbox[0]
    cx = (SLIDE_W - w) // 2
    cy = SLIDE_H // 2 - 120
    draw.text((cx, cy), title, font=title_font, fill=WHITE)

    # Accent underline
    draw.rounded_rectangle((cx, cy + 80, cx + w, cy + 86), radius=3, fill=ACCENT)

    # Bot name
    bot_text = f"@{bot_username}"
    bbox = bot_font.getbbox(bot_text)
    bw = bbox[2] - bbox[0]
    draw.text(((SLIDE_W - bw) // 2, cy + 120), bot_text, font=bot_font, fill=ACCENT)

    # Subtitle
    lines = _wrap_text(subtitle, sub_font, SLIDE_W - PADDING * 2)
    for i, line in enumerate(lines):
        bbox = sub_font.getbbox(line)
        lw = bbox[2] - bbox[0]
        draw.text(((SLIDE_W - lw) // 2, cy + 200 + i * 45), line,
                 font=sub_font, fill=GRAY)

    # Decorative elements
    _add_accent_line(draw, PADDING + 20, width=1)
    _add_accent_line(draw, SLIDE_H - PADDING - 20, width=1)

    _save(img, output_name)


def _save(img: Image.Image, name: str):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    img.save(path, "PNG", quality=95)
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════
# YOUR SLIDES — EDIT THIS SECTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🎨 Generating slides...\n")

    # ── Slide 1: Title ──
    make_title_slide(
        title="КАК ПИСАТЬ ПРОМПТЫ",
        subtitle="для реалистичных AI фото",
        output_name="01_title",
    )

    # ── Slide 2: Bad vs Good ──
    make_comparison_slide(
        title="❌ vs ✅",
        bad_label="❌ Плохой промпт",
        bad_text="beautiful girl on beach, realistic, 8k, high quality",
        good_label="✅ Хороший промпт",
        good_text="A young woman with messy sun-bleached hair, "
                  "freckles across her nose, taking a selfie on a "
                  "crowded beach. Shot on iPhone 15 Pro, slight "
                  "overexposure from direct sunlight, sand particles "
                  "on her skin, casual unposed expression...",
        output_name="02_comparison",
        # bad_screenshot="path/to/bad_result.png",    # ← вставь скриншот
        # good_screenshot="path/to/good_result.png",  # ← вставь скриншот
    )

    # ── Slide 3: 5 Rules ──
    make_bullets_slide(
        title="5 ПРАВИЛ ИДЕАЛЬНОГО ПРОМПТА",
        bullets=[
            "📸  Описывай камеру — \"Shot on iPhone 15 Pro, f/1.9\"",
            "💡  Свет = реализм — окно, лампа, НЕ софтбокс",
            "🏠  Конкретное место — грязная комната, а не \"studio\"",
            "👁  Детали лица — веснушки, поры, текстура кожи",
            "🎨  Amateur качество — grain, slight blur, неидеальный кадр",
        ],
        output_name="03_rules",
        # screenshot_path="path/to/bot_screenshot.png",  # ← скриншот бота
    )

    # ── Slide 4: CTA ──
    make_cta_slide(
        title="ПОПРОБУЙ САМ ✨",
        bot_username="YourBotName",  # ← замени на имя бота
        subtitle="Пиши простой промпт — бот сделает его идеальным. "
                 "Автоматический промт-улучшатель встроен!",
        output_name="04_cta",
    )

    print(f"\n🎉 Done! Slides saved to: {OUTPUT_DIR.absolute()}")
