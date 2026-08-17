"""labelme JSON Segmentation -> Image
====================================
Converts labelme-format JSON file(s) into:
  1. A filled mask image  (label colours per class)
  2. An overlay image     (original image + semi-transparent mask)
  3. An outline image     (original image + polygon outlines)

Single file:
  python labelme_seg_to_image.py --json path/to/annotation.json

Batch (directory of JSON files):
  python labelme_seg_to_image.py --json_dir path/to/jsons/ --out_dir results/
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Colour palette  (one colour per unique label, RGBA)
# ---------------------------------------------------------------------------

PALETTE = [
    (255,   0,   0, 128),
    (  0, 255,   0, 128),
    (  0,   0, 255, 128),
    (255, 255,   0, 128),
    (255,   0, 255, 128),
    (  0, 255, 255, 128),
    (255, 128,   0, 128),
    (128,   0, 255, 128),
]


def colour_for(label: str, label_index: dict) -> tuple:
    if label not in label_index:
        label_index[label] = len(label_index)
    return PALETTE[label_index[label] % len(PALETTE)]


# ---------------------------------------------------------------------------
# Load base image
# ---------------------------------------------------------------------------

def load_base_image(data: dict, json_path: str) -> Image.Image:
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
    mask = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        pts = [tuple(p) for p in shape["points"]]
        colour = colour_for(shape["label"], label_index)
        draw.polygon(pts, fill=colour)
    return mask


def make_outline(shapes: list, size: tuple, label_index: dict,
                 line_width: int = 2) -> Image.Image:
    outline = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(outline)
    for shape in shapes:
        pts = [tuple(p) for p in shape["points"]]
        r, g, b, _ = colour_for(shape["label"], label_index)
        draw.polygon(pts, outline=(r, g, b, 255), width=line_width)
    return outline


def draw_legend(image: Image.Image, label_index: dict) -> Image.Image:
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
        draw.text((x + box_h + 4, y + 2), label, fill=(255, 255, 255, 255), font=font)
        y += box_h + 4
    return img


# ---------------------------------------------------------------------------
# Per-file converter
# ---------------------------------------------------------------------------

def convert(json_path: str, out_dir: str | None = None,
            line_width: int = 2, save_mask: bool = True,
            save_overlay: bool = True, save_outline: bool = True):

    json_path = str(Path(json_path).resolve())
    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        print(f"[WARN] {json_path}: no shapes found, skipping.")
        return None

    base_img = load_base_image(data, json_path)
    W, H = base_img.size
    print(f"[INFO] {Path(json_path).name}: {W}x{H}, {len(shapes)} shape(s)")

    label_index: dict = {}
    mask_layer = make_mask(shapes, (W, H), label_index)
    out_layer  = make_outline(shapes, (W, H), label_index, line_width)

    stem = Path(json_path).stem
    out_path = Path(out_dir) if out_dir else Path(json_path).parent
    out_path.mkdir(parents=True, exist_ok=True)

    results = {}

    if save_mask:
        mask_rgb = Image.new("RGB", (W, H), (0, 0, 0))
        mask_rgb.paste(mask_layer, mask=mask_layer.split()[3])
        mask_rgb = draw_legend(mask_rgb.convert("RGBA"), label_index)
        p = out_path / f"{stem}_mask.png"
        mask_rgb.convert("RGB").save(p)
        results["mask"] = str(p)
        print(f"  mask    -> {p}")

    if save_overlay:
        overlay = Image.alpha_composite(base_img, mask_layer)
        overlay = draw_legend(overlay, label_index)
        p = out_path / f"{stem}_overlay.png"
        overlay.convert("RGB").save(p)
        results["overlay"] = str(p)
        print(f"  overlay -> {p}")

    if save_outline:
        outline_img = Image.alpha_composite(base_img, out_layer)
        outline_img = draw_legend(outline_img, label_index)
        p = out_path / f"{stem}_outline.png"
        outline_img.convert("RGB").save(p)
        results["outline"] = str(p)
        print(f"  outline -> {p}")

    return results


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def convert_directory(json_dir: str, out_dir: str | None = None, **kwargs):
    """Process every *.json file inside json_dir."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {json_dir}")
        return

    print(f"[INFO] Found {len(json_files)} JSON file(s) in {json_dir}")
    ok, fail = 0, 0
    for jf in json_files:
        # If out_dir not given, create a subfolder next to the json_dir
        file_out = out_dir
        try:
            convert(str(jf), out_dir=file_out, **kwargs)
            ok += 1
        except Exception as e:
            print(f"[ERROR] {jf.name}: {e}")
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert labelme JSON segmentation(s) to image outputs")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json",     help="Single labelme JSON file")
    src.add_argument("--json_dir", help="Directory containing multiple labelme JSON files")

    parser.add_argument("--out_dir",    default=None,  help="Output directory")
    parser.add_argument("--line_width", type=int, default=2, help="Outline width (default: 2)")
    parser.add_argument("--no_mask",    action="store_true")
    parser.add_argument("--no_overlay", action="store_true")
    parser.add_argument("--no_outline", action="store_true")
    args = parser.parse_args()

    kwargs = dict(
        line_width   = args.line_width,
        save_mask    = not args.no_mask,
        save_overlay = not args.no_overlay,
        save_outline = not args.no_outline,
    )

    if args.json:
        convert(args.json, out_dir=args.out_dir, **kwargs)
    else:
        convert_directory(args.json_dir, out_dir=args.out_dir, **kwargs)
