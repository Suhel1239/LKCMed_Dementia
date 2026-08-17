"""
labelme JSON Segmentation -> Image
====================================
Converts a labelme-format JSON file into:
  1. A filled mask image  (label colours per class)
  2. An overlay image     (original image + semi-transparent mask)
  3. An outline image     (original image + polygon outlines)

The JSON is expected to have:
  - shapes[].label   : class name
  - shapes[].points  : [[x,y], ...] polygon vertices
  - imageData        : base64-encoded original image (optional)
  - imagePath        : path to original image (fallback if no imageData)

Run:
  python labelme_seg_to_image.py --json path/to/annotation.json
  python labelme_seg_to_image.py --json path/to/annotation.json --out_dir results/
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Colour palette  (one colour per unique label, RGBA)
# ---------------------------------------------------------------------------

PALETTE = [
    (255,   0,   0, 128),   # red
    (  0, 255,   0, 128),   # green
    (  0,   0, 255, 128),   # blue
    (255, 255,   0, 128),   # yellow
    (255,   0, 255, 128),   # magenta
    (  0, 255, 255, 128),   # cyan
    (255, 128,   0, 128),   # orange
    (128,   0, 255, 128),   # purple
]


def colour_for(label: str, label_index: dict) -> tuple:
    """Return a consistent RGBA colour for a label string."""
    if label not in label_index:
        label_index[label] = len(label_index)
    return PALETTE[label_index[label] % len(PALETTE)]


# ---------------------------------------------------------------------------
# Load base image
# ---------------------------------------------------------------------------

def load_base_image(data: dict, json_path: str) -> Image.Image:
    """Load the original image from imageData (base64) or imagePath."""
    if data.get("imageData"):
        raw = base64.b64decode(data["imageData"])
        return Image.open(io.BytesIO(raw)).convert("RGBA")

    img_path = data.get("imagePath", "")
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(json_path), img_path)

    if os.path.isfile(img_path):
        return Image.open(img_path).convert("RGBA")

    raise FileNotFoundError(
        "No imageData in JSON and imagePath not found: " + img_path
    )


# ---------------------------------------------------------------------------
# Core render functions
# ---------------------------------------------------------------------------

def make_mask(shapes: list, size: tuple, label_index: dict) -> Image.Image:
    """
    Solid-colour filled mask on a transparent background.
    size = (width, height)
    """
    mask = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        pts = [tuple(p) for p in shape["points"]]
        colour = colour_for(shape["label"], label_index)
        draw.polygon(pts, fill=colour)
    return mask


def make_outline(shapes: list, size: tuple, label_index: dict,
                 line_width: int = 2) -> Image.Image:
    """Polygon outlines only, on transparent background."""
    outline = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(outline)
    for shape in shapes:
        pts = [tuple(p) for p in shape["points"]]
        r, g, b, _ = colour_for(shape["label"], label_index)
        draw.polygon(pts, outline=(r, g, b, 255), width=line_width)
    return outline


def draw_legend(image: Image.Image, label_index: dict) -> Image.Image:
    """Add a small legend in the top-left corner."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y, box_h = 8, 8, 18
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for label, idx in label_index.items():
        r, g, b, _ = PALETTE[idx % len(PALETTE)]
        draw.rectangle([x, y, x + box_h, y + box_h], fill=(r, g, b, 220))
        draw.text((x + box_h + 4, y + 2), label, fill=(255, 255, 255, 255),
                  font=font)
        y += box_h + 4
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(json_path: str, out_dir: str | None = None,
            line_width: int = 2, save_mask: bool = True,
            save_overlay: bool = True, save_outline: bool = True):

    json_path = str(Path(json_path).resolve())
    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        print("[WARN] No shapes found in JSON. Nothing to render.")
        return

    base_img = load_base_image(data, json_path)
    W, H = base_img.size
    print(f"[INFO] Base image: {W}x{H},  {len(shapes)} shape(s)")

    label_index: dict = {}
    mask_layer  = make_mask(shapes, (W, H), label_index)
    out_layer   = make_outline(shapes, (W, H), label_index, line_width)

    stem = Path(json_path).stem
    out_dir = Path(out_dir) if out_dir else Path(json_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if save_mask:
        mask_rgb = Image.new("RGB", (W, H), (0, 0, 0))
        mask_rgb.paste(mask_layer, mask=(mask_layer.split()[3]))
        mask_rgb = draw_legend(mask_rgb.convert("RGBA"), label_index)
        path = out_dir / f"{stem}_mask.png"
        mask_rgb.convert("RGB").save(path)
        results["mask"] = str(path)
        print(f"[SAVED] mask    -> {path}")

    if save_overlay:
        overlay = Image.alpha_composite(base_img, mask_layer)
        overlay = draw_legend(overlay, label_index)
        path = out_dir / f"{stem}_overlay.png"
        overlay.convert("RGB").save(path)
        results["overlay"] = str(path)
        print(f"[SAVED] overlay -> {path}")

    if save_outline:
        outline_img = Image.alpha_composite(base_img, out_layer)
        outline_img = draw_legend(outline_img, label_index)
        path = out_dir / f"{stem}_outline.png"
        outline_img.convert("RGB").save(path)
        results["outline"] = str(path)
        print(f"[SAVED] outline -> {path}")

    print(f"\nLabels found: {list(label_index.keys())}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert labelme JSON segmentation to image outputs")
    parser.add_argument("--json",     required=True, help="Path to labelme JSON file")
    parser.add_argument("--out_dir",  default=None,  help="Output directory (default: same as JSON)")
    parser.add_argument("--line_width", type=int, default=2, help="Polygon outline width (default: 2)")
    parser.add_argument("--no_mask",    action="store_true", help="Skip saving mask image")
    parser.add_argument("--no_overlay", action="store_true", help="Skip saving overlay image")
    parser.add_argument("--no_outline", action="store_true", help="Skip saving outline image")
    args = parser.parse_args()

    convert(
        json_path   = args.json,
        out_dir     = args.out_dir,
        line_width  = args.line_width,
        save_mask   = not args.no_mask,
        save_overlay= not args.no_overlay,
        save_outline= not args.no_outline,
    )
