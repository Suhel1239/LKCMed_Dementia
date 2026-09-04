"""
DICOM + BBox CSV  ->  COCO annotations JSON
============================================

Input:
  - A CSV file with columns:  sopInstanceUid, sx, sy, ex, ey
    (sx,sy = top-left corner;  ex,ey = bottom-right corner)
    Optionally a 'label' column for class names.

  - A root folder containing DICOM files (searched recursively).
    Each .dcm file's SOPInstanceUID tag is used for matching.

Output:
  - coco_annotations.json  in standard COCO format:
      {
        "images":      [{id, file_name, width, height}, ...],
        "annotations": [{id, image_id, category_id, bbox:[x,y,w,h], area}, ...],
        "categories":  [{id, name}, ...]
      }
  - A summary printed to the terminal.

Run:
  python make_coco_annotations.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


# =============================================================================
# CONFIG  -- update these before running
# =============================================================================

CSV_PATH   = "/path/to/annotations.csv"      # <-- your CSV
DICOM_ROOT = "/path/to/dicom_folder"          # <-- root folder with subfolders of .dcm
OUT_JSON   = "/path/to/coco_annotations.json" # <-- where to save the result

UID_COL    = "sopInstanceUid"   # column name for the DICOM UID
SX_COL     = "sx"               # top-left x
SY_COL     = "sy"               # top-left y
EX_COL     = "ex"               # bottom-right x
EY_COL     = "ey"               # bottom-right y
LABEL_COL  = None               # set to column name if you have class labels,
                                 # e.g. LABEL_COL = "finding"  -- else leave None


# =============================================================================
# Step 1 -- scan DICOM folder: SOPInstanceUID -> file path + image size
# =============================================================================

def scan_dicom_folder(root: str) -> dict:
    """
    Returns:
        {
          uid: {
            "path":   str,   # absolute path to .dcm file
            "width":  int,
            "height": int,
          }, ...
        }
    """
    uid_info = {}
    dcm_files = list(Path(root).rglob("*.dcm"))
    print(f"[INFO] scanning {len(dcm_files)} .dcm files...")

    for p in dcm_files:
        try:
            ds  = pydicom.dcmread(str(p), stop_before_pixels=False)
            uid = str(ds.SOPInstanceUID).strip()

            # image dimensions
            rows = int(getattr(ds, "Rows",    getattr(ds, "NumberOfRows",    0)))
            cols = int(getattr(ds, "Columns", getattr(ds, "NumberOfColumns", 0)))

            # fallback: read pixel array shape
            if rows == 0 or cols == 0:
                arr  = ds.pixel_array
                rows = arr.shape[0]
                cols = arr.shape[1]

            uid_info[uid] = {
                "path":   str(p.resolve()),
                "width":  cols,
                "height": rows,
            }
        except Exception as e:
            print(f"[WARN] skipping {p.name}: {e}")

    print(f"[INFO] found {len(uid_info)} valid DICOM files")
    return uid_info


# =============================================================================
# Step 2 -- read CSV and build per-UID annotation list
# =============================================================================

def load_csv_annotations(csv_path: str) -> tuple:
    """
    Returns:
        ann_by_uid : { uid: [{sx,sy,ex,ey,label}, ...] }
        categories : { label_name: category_id }  (always has at least {"object": 1})
    """
    df = pd.read_csv(csv_path)
    df[UID_COL] = df[UID_COL].astype(str).str.strip()

    # build category map
    if LABEL_COL and LABEL_COL in df.columns:
        classes    = sorted(df[LABEL_COL].dropna().astype(str).unique().tolist())
        categories = {c: i + 1 for i, c in enumerate(classes)}
    else:
        categories = {"object": 1}

    ann_by_uid: dict = {}
    for _, row in df.iterrows():
        uid   = str(row[UID_COL]).strip()
        label = str(row[LABEL_COL]).strip() if (LABEL_COL and LABEL_COL in row) else "object"
        entry = {
            "sx":    float(row[SX_COL]),
            "sy":    float(row[SY_COL]),
            "ex":    float(row[EX_COL]),
            "ey":    float(row[EY_COL]),
            "label": label,
        }
        ann_by_uid.setdefault(uid, []).append(entry)

    print(f"[INFO] CSV contains annotations for {len(ann_by_uid)} unique UIDs")
    print(f"[INFO] categories: {categories}")
    return ann_by_uid, categories


# =============================================================================
# Step 3 -- build COCO JSON
# =============================================================================

def build_coco(uid_info: dict, ann_by_uid: dict, categories: dict) -> dict:
    """
    Matches UIDs present in BOTH the DICOM folder and the CSV.
    Bboxes are converted from (sx,sy,ex,ey) to COCO format [x, y, width, height].
    """
    common_uids = sorted(set(uid_info.keys()) & set(ann_by_uid.keys()))
    only_csv    = len(set(ann_by_uid.keys()) - set(uid_info.keys()))
    only_dcm    = len(set(uid_info.keys())   - set(ann_by_uid.keys()))

    print(f"[INFO] matched UIDs : {len(common_uids)}")
    print(f"[INFO] in CSV only  : {only_csv}  (no matching DICOM file found)")
    print(f"[INFO] in DICOM only: {only_dcm}  (no annotation in CSV)")

    if not common_uids:
        raise ValueError(
            "No UIDs matched between CSV and DICOM folder. "
            "Check that sopInstanceUid values in the CSV match the "
            "SOPInstanceUID DICOM tag in the files."
        )

    coco_images      = []
    coco_annotations = []
    ann_id           = 1

    for img_id, uid in enumerate(common_uids, start=1):
        info = uid_info[uid]
        coco_images.append({
            "id":        img_id,
            "file_name": info["path"],   # full path so Dataset can load it
            "width":     info["width"],
            "height":    info["height"],
            "uid":       uid,            # extra field for traceability
        })

        for box in ann_by_uid[uid]:
            sx, sy, ex, ey = box["sx"], box["sy"], box["ex"], box["ey"]
            w = ex - sx
            h = ey - sy

            if w <= 0 or h <= 0:
                print(f"[WARN] degenerate box uid={uid} sx={sx} sy={sy} ex={ex} ey={ey} -- skipped")
                continue

            category_id = categories.get(box["label"], 1)
            coco_annotations.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": category_id,
                "bbox":        [sx, sy, w, h],   # COCO: [x, y, width, height]
                "area":        w * h,
                "iscrowd":     0,
            })
            ann_id += 1

    coco_categories = [
        {"id": cid, "name": name}
        for name, cid in sorted(categories.items(), key=lambda x: x[1])
    ]

    return {
        "images":      coco_images,
        "annotations": coco_annotations,
        "categories":  coco_categories,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  DICOM + CSV  ->  COCO annotations")
    print(f"  CSV        : {CSV_PATH}")
    print(f"  DICOM root : {DICOM_ROOT}")
    print(f"  Output     : {OUT_JSON}")
    print("=" * 60)

    uid_info   = scan_dicom_folder(DICOM_ROOT)
    ann_by_uid, categories = load_csv_annotations(CSV_PATH)
    coco       = build_coco(uid_info, ann_by_uid, categories)

    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(coco, f, indent=2)

    print("=" * 60)
    print(f"  images      : {len(coco['images'])}")
    print(f"  annotations : {len(coco['annotations'])}")
    print(f"  categories  : {coco['categories']}")
    print(f"  Saved -> {OUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
