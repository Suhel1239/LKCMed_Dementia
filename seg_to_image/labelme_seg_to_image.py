"""labelme JSON Segmentation -> B&W Mask Image
===============================================
Converts labelme-format JSON file(s) into a binary black-and-white mask.
  - White (255) = segmented region
  - Black (0)   = background
  - Output filename matches the JSON filename (e.g. case01.json -> case01.png)

Single file:
  python labelme_seg_to_image.py --json path/to/annotation.json
  python labelme_seg_to_image.py --json path/to/annotation.json --out_dir masks/

Batch (directory of JSON files):
  python labelme_seg_to_image.py --json_dir annotations/ --out_dir masks/
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Load base image (needed only for canvas size)
# ---------------------------------------------------------------------------

def load_base_image(data: dict, json_path: str) -> Image.Image:
    if data.get("imageData"):
        raw = base64.b64decode(data["imageData"])
        return Image.open(io.BytesIO(raw)).convert("L")

    img_path = data.get("imagePath", "")
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(json_path), img_path)

    if os.path.isfile(img_path):
        return Image.open(img_path).convert("L")

    # Fall back to declared image size if present
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if w and h:
        return Image.new("L", (int(w), int(h)), 0)

    raise FileNotFoundError(
        "No imageData in JSON and imagePath not found: " + img_path
    )


# ---------------------------------------------------------------------------
# Per-file converter
# ---------------------------------------------------------------------------

def convert(json_path: str, out_dir: str | None = None):
    json_path = str(Path(json_path).resolve())
    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        print(f"[WARN] {Path(json_path).name}: no shapes found, skipping.")
        return None

    base_img = load_base_image(data, json_path)
    W, H = base_img.size

    mask = Image.new("L", (W, H), 0)          # black canvas
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        pts = [tuple(p) for p in shape["points"]]
        draw.polygon(pts, fill=255)            # white fill

    stem = Path(json_path).stem
    out_path = Path(out_dir) if out_dir else Path(json_path).parent
    out_path.mkdir(parents=True, exist_ok=True)

    p = out_path / f"{stem}.png"
    mask.save(p)
    print(f"[SAVED] {p}")
    return str(p)


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def convert_directory(json_dir: str, out_dir: str | None = None):
    json_files = sorted(Path(json_dir).glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {json_dir}")
        return

    print(f"[INFO] Found {len(json_files)} JSON file(s) in {json_dir}")
    ok, fail = 0, 0
    for jf in json_files:
        try:
            convert(str(jf), out_dir=out_dir)
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
        description="Convert labelme JSON segmentation(s) to B&W mask images")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json",     help="Single labelme JSON file")
    src.add_argument("--json_dir", help="Directory containing multiple labelme JSON files")

    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: same folder as each JSON)")
    args = parser.parse_args()

    if args.json:
        convert(args.json, out_dir=args.out_dir)
    else:
        convert_directory(args.json_dir, out_dir=args.out_dir)
