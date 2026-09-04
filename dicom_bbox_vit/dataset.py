"""
DICOM + BBox -> ViT Dataset
============================
CSV columns expected:
    sopInstanceUid, sx, sy, ex, ey   (+ optional: label)

Folder structure (any nesting depth):
    dicom_root/
        subfolder_A/
            *.dcm
        subfolder_B/
            *.dcm
        ...

Each DICOM file's SOPInstanceUID tag is matched against the CSV.

Outputs per sample:
    image  : FloatTensor (3, H, W)   -- full image, normalised
    boxes  : FloatTensor (N, 4)      -- [sx, sy, ex, ey] in pixel coords
    labels : LongTensor  (N,)        -- class indices (0 if no label col)
    uid    : str                     -- sopInstanceUid
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH        = "/path/to/annotations.csv"   # <-- update
DICOM_ROOT      = "/path/to/dicom_folder"       # <-- update
UID_COL         = "sopInstanceUid"
BBOX_COLS       = ["sx", "sy", "ex", "ey"]      # top-left (sx,sy) bottom-right (ex,ey)
LABEL_COL       = None                          # set to column name if you have classes
IMAGE_SIZE      = 224                           # ViT input size


# ---------------------------------------------------------------------------
# Step 1 -- scan DICOM folder and build uid -> path map
# ---------------------------------------------------------------------------

def build_uid_map(dicom_root: str) -> Dict[str, Path]:
    """Recursively scan for .dcm files and map SOPInstanceUID -> Path."""
    uid_map: Dict[str, Path] = {}
    root = Path(dicom_root)
    dcm_files = list(root.rglob("*.dcm"))
    print(f"[INFO] found {len(dcm_files)} .dcm files under {root}")

    for p in dcm_files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            uid = str(ds.SOPInstanceUID).strip()
            uid_map[uid] = p
        except Exception as e:
            print(f"[WARN] could not read {p}: {e}")

    print(f"[INFO] mapped {len(uid_map)} unique SOPInstanceUIDs")
    return uid_map


# ---------------------------------------------------------------------------
# Step 2 -- load CSV and group annotations by UID
# ---------------------------------------------------------------------------

def load_annotations(
    csv_path: str,
    uid_col: str   = UID_COL,
    bbox_cols: List[str] = BBOX_COLS,
    label_col: Optional[str] = LABEL_COL,
) -> Dict[str, dict]:
    """
    Returns:
        {
          uid: {
            "boxes":  [[sx,sy,ex,ey], ...],
            "labels": [int, ...]
          }, ...
        }
    """
    df = pd.read_csv(csv_path)
    df[uid_col] = df[uid_col].astype(str).str.strip()

    # build label map if label column present
    label_map: Dict[str, int] = {}
    if label_col and label_col in df.columns:
        classes = sorted(df[label_col].dropna().unique().tolist())
        label_map = {str(c): i for i, c in enumerate(classes)}
        print(f"[INFO] classes: {label_map}")

    annotations: Dict[str, dict] = {}
    for uid, grp in df.groupby(uid_col):
        boxes  = grp[bbox_cols].values.tolist()
        if label_col and label_col in grp.columns:
            labels = [label_map.get(str(v), 0) for v in grp[label_col].tolist()]
        else:
            labels = [0] * len(boxes)
        annotations[uid] = {"boxes": boxes, "labels": labels}

    print(f"[INFO] {len(annotations)} UIDs with annotations in CSV")
    return annotations


# ---------------------------------------------------------------------------
# DICOM pixel -> PIL Image
# ---------------------------------------------------------------------------

def dcm_to_pil(path: Path) -> Image.Image:
    ds  = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    # apply rescale if present
    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # window/level if present, else min-max
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth",  None)
    if wc is not None and ww is not None:
        wc = float(wc[0] if hasattr(wc, "__len__") else wc)
        ww = float(ww[0] if hasattr(ww, "__len__") else ww)
        lo, hi = wc - ww / 2, wc + ww / 2
        arr = np.clip(arr, lo, hi)
    else:
        lo, hi = arr.min(), arr.max()

    arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    if arr.ndim == 2:
        img = Image.fromarray(arr).convert("RGB")
    else:                       # (H, W, C) -- rare in DICOM
        img = Image.fromarray(arr[..., :3])
    return img


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DicomBBoxDataset(Dataset):
    """
    Returns:
        image  : FloatTensor (3, IMAGE_SIZE, IMAGE_SIZE)
        target : dict with keys
                   'boxes'  FloatTensor (N, 4)  -- pixel coords scaled to IMAGE_SIZE
                   'labels' LongTensor  (N,)
                   'uid'    str
    """

    def __init__(
        self,
        uid_map: Dict[str, Path],
        annotations: Dict[str, dict],
        image_size: int  = IMAGE_SIZE,
        augment: bool    = False,
    ):
        # only keep UIDs that exist in both CSV and DICOM folder
        common = sorted(set(uid_map.keys()) & set(annotations.keys()))
        if not common:
            raise ValueError("No matching UIDs between CSV and DICOM folder!")
        print(f"[INFO] dataset size: {len(common)} matched samples "
              f"(csv={len(annotations)}, dcm={len(uid_map)})")

        self.uids        = common
        self.uid_map     = uid_map
        self.annotations = annotations
        self.image_size  = image_size

        base_tf = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ]
        aug_tf = ([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ] if augment else [])
        self.transform = transforms.Compose(aug_tf + base_tf)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        uid  = self.uids[idx]
        path = self.uid_map[uid]
        ann  = self.annotations[uid]

        img = dcm_to_pil(path)
        orig_w, orig_h = img.size          # PIL: (W, H)

        image = self.transform(img)        # (3, H', W')

        # scale boxes to new image size
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        scaled_boxes = []
        for sx, sy, ex, ey in ann["boxes"]:
            scaled_boxes.append([
                sx * scale_x,
                sy * scale_y,
                ex * scale_x,
                ey * scale_y,
            ])

        target = {
            "boxes":  torch.tensor(scaled_boxes, dtype=torch.float32),
            "labels": torch.tensor(ann["labels"], dtype=torch.long),
            "uid":    uid,
        }
        return image, target


# ---------------------------------------------------------------------------
# Collate (handles variable number of boxes per image)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# ---------------------------------------------------------------------------
# Build function (call this from your training script)
# ---------------------------------------------------------------------------

def build_dataset(
    csv_path: str    = CSV_PATH,
    dicom_root: str  = DICOM_ROOT,
    image_size: int  = IMAGE_SIZE,
    augment: bool    = False,
) -> DicomBBoxDataset:
    uid_map     = build_uid_map(dicom_root)
    annotations = load_annotations(csv_path)
    return DicomBBoxDataset(uid_map, annotations,
                            image_size=image_size, augment=augment)
