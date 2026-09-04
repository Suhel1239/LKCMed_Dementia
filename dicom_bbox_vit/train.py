"""
ViT Fine-Tuning for DICOM BBox Detection
==========================================
Strategy:
  - ViT-B/16 backbone (ImageNet pretrained), classification head replaced
    with a detection head that predicts ONE box per image.
  - If each image has multiple boxes, each box is treated as a separate sample
    (set MULTI_BOX = True) -- or you can use the mean box (MULTI_BOX = False).
  - Loss: smooth-L1 for bbox regression + cross-entropy for class label.

Run:
    python train.py
"""

import os, sys, warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import vit_b_16, ViT_B_16_Weights

warnings.filterwarnings("ignore")

# local
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataset, collate_fn, IMAGE_SIZE


# =============================================================================
# CONFIG -- update paths and hyper-params
# =============================================================================

CSV_PATH   = "/path/to/annotations.csv"   # <-- update
DICOM_ROOT = "/path/to/dicom_folder"       # <-- update
WEIGHT_DIR = "/path/to/weights"            # <-- update
LOG_PATH   = "/path/to/train_log.txt"      # <-- update

NUM_CLASSES = 1     # set to number of classes (1 if no label column)
MULTI_BOX   = False # True -> expand each box to its own sample

CFG = dict(
    epochs     = 50,
    lr         = 1e-4,
    batch_size = 8,
    val_frac   = 0.2,
    dropout    = 0.3,
    patience   = 10,
    image_size = IMAGE_SIZE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Flat dataset: one sample per box
# =============================================================================

class FlatBoxDataset(Dataset):
    """Wraps DicomBBoxDataset so each (image, box, label) is one sample."""
    def __init__(self, base_ds, multi_box=True):
        self.items = []  # list of (image_tensor, box_tensor, label_int)
        for i in range(len(base_ds)):
            img, target = base_ds[i]
            boxes  = target["boxes"]   # (N, 4)
            labels = target["labels"]  # (N,)
            if multi_box:
                for b, l in zip(boxes, labels):
                    self.items.append((img, b, l))
            else:
                # use the first (or only) box
                self.items.append((img, boxes[0], labels[0]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# =============================================================================
# Model
# =============================================================================

class ViTDetector(nn.Module):
    """
    ViT-B/16 backbone + detection head.
    Outputs:
        box_pred   : (B, 4)           -- predicted [sx, sy, ex, ey] (pixel)
        class_pred : (B, NUM_CLASSES) -- class logits
    """
    def __init__(self, num_classes: int, image_size: int, dropout: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # remove original head
        self.backbone   = nn.Sequential(*list(vit.children())[:-1])  # up to encoder
        self.patch_embed = vit.conv_proj
        self.encoder     = vit.encoder
        self.class_token = vit.class_token

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

        # detection head (trainable always)
        self.bbox_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 4),   nn.Sigmoid(),   # outputs 0-1, scaled to image_size
        )
        self.cls_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        for p in self.bbox_head.parameters(): p.requires_grad = True
        for p in self.cls_head.parameters():  p.requires_grad = True

        self.image_size = image_size
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ViTDetector] trainable params: {trainable:,}  (backbone frozen={freeze_backbone})")

    @torch.no_grad()
    def _vit_features(self, x):
        B   = x.shape[0]
        x   = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        out = self.encoder(torch.cat([cls, x], dim=1))
        return out[:, 0]   # CLS token -> (B, 768)

    def forward(self, x):
        feat      = self._vit_features(x)       # (B, 768)
        box_pred  = self.bbox_head(feat) * self.image_size  # -> pixel coords
        cls_pred  = self.cls_head(feat)
        return box_pred, cls_pred


# =============================================================================
# Training loop
# =============================================================================

def train(resume: bool = False):
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    # datasets
    base_ds  = build_dataset(CSV_PATH, DICOM_ROOT,
                             image_size=CFG["image_size"], augment=True)
    flat_ds  = FlatBoxDataset(base_ds, multi_box=MULTI_BOX)

    # train / val split
    indices  = list(range(len(flat_ds)))
    tr_idx, vl_idx = train_test_split(indices, test_size=CFG["val_frac"],
                                      random_state=42)
    tr_ds = Subset(flat_ds, tr_idx)
    vl_ds = Subset(flat_ds, vl_idx)
    print(f"[INFO] train={len(tr_ds)}  val={len(vl_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=CFG["batch_size"],
                           shuffle=True,  num_workers=0)
    vl_loader = DataLoader(vl_ds, batch_size=CFG["batch_size"],
                           shuffle=False, num_workers=0)

    model     = ViTDetector(NUM_CLASSES, CFG["image_size"],
                            CFG["dropout"]).to(device)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG["lr"], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6)

    bbox_loss_fn = nn.SmoothL1Loss()
    cls_loss_fn  = nn.CrossEntropyLoss()

    best_val  = float("inf")
    no_imp    = 0
    start_ep  = 1
    best_path = os.path.join(WEIGHT_DIR, "best.pth")
    last_path = os.path.join(WEIGHT_DIR, "last.pth")

    if resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val  = ckpt["best_val"]
        no_imp    = ckpt["no_imp"]
        start_ep  = ckpt["epoch"] + 1
        print(f"[RESUME] epoch {start_ep}  best_val={best_val:.4f}")

    for epoch in range(start_ep, CFG["epochs"] + 1):
        # -- train
        model.train()
        tr_losses = []
        for imgs, boxes, labels in tr_loader:
            imgs, boxes, labels = (imgs.to(device), boxes.to(device),
                                   labels.to(device))
            optimizer.zero_grad()
            box_pred, cls_pred = model(imgs)
            loss_box = bbox_loss_fn(box_pred, boxes)
            loss_cls = cls_loss_fn(cls_pred, labels) if NUM_CLASSES > 1 else torch.tensor(0.0)
            loss     = loss_box + loss_cls
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())
        scheduler.step()

        # -- val
        model.eval()
        vl_losses = []
        with torch.no_grad():
            for imgs, boxes, labels in vl_loader:
                imgs, boxes, labels = (imgs.to(device), boxes.to(device),
                                       labels.to(device))
                box_pred, cls_pred = model(imgs)
                loss_box = bbox_loss_fn(box_pred, boxes)
                loss_cls = cls_loss_fn(cls_pred, labels) if NUM_CLASSES > 1 else torch.tensor(0.0)
                vl_losses.append((loss_box + loss_cls).item())

        tr_l = np.mean(tr_losses)
        vl_l = np.mean(vl_losses)
        print(f"Epoch {epoch:03d} | train_loss={tr_l:.4f}  val_loss={vl_l:.4f}")

        if vl_l < best_val:
            best_val = vl_l;  no_imp = 0
            torch.save(model.state_dict(), best_path)
            print(f"  [best] val_loss={best_val:.4f}")
        else:
            no_imp += 1

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val":  best_val,
            "no_imp":    no_imp,
        }, last_path)

        if no_imp >= CFG["patience"]:
            print(f"[EarlyStop] epoch={epoch}")
            break

    print(f"\nDone. Best val_loss={best_val:.4f}  weights -> {best_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    train(resume=args.resume)
