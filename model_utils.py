"""
Shared model components, dataset, training loop, and helpers.
Imported by pipeline_enface.py, pipeline_octa.py, pipeline_structured.py.
"""

import os
import sys as _sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

warnings.filterwarnings("ignore")

SEED        = 42
VIT_B16_DIM = 768
PCA_COMPS   = 15
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._file   = open(log_path, "w", encoding="utf-8")
        self._stdout = _sys.__stdout__

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self.flush()
        self._file.close()
        _sys.stdout = self._stdout


# ---------------------------------------------------------------------------
# Folder map
# ---------------------------------------------------------------------------

def build_subject_folder_map(image_root: str) -> Dict[str, Path]:
    image_root = Path(image_root)
    folder_map: Dict[str, Path] = {}
    for d in image_root.iterdir():
        if d.is_dir():
            folder_map[d.name[-13:]] = d
    print(f"[INFO] Found {len(folder_map)} patient folders under {image_root}")
    return folder_map


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def load_features_by_threshold(feature_score_csv, feature_name_col,
                                score_col, threshold, data_columns):
    scores_df    = pd.read_csv(feature_score_csv)
    passing      = scores_df[scores_df[score_col] >= threshold]
    selected     = passing[feature_name_col].astype(str).tolist()
    data_col_set = set(data_columns)
    matched      = [f for f in selected if f in data_col_set]
    unmatched    = [f for f in selected if f not in data_col_set]

    print(f"\n[FeatureSelection] threshold >= {threshold}")
    print(f"  Passing: {len(selected)}  |  Matched in data: {len(matched)}")
    if unmatched:
        print(f"  [WARN] {len(unmatched)} unmatched: {unmatched[:5]}")
    if not matched:
        raise ValueError("No features matched threshold.")

    score_lookup = dict(zip(scores_df[feature_name_col].astype(str),
                            scores_df[score_col]))
    matched.sort(key=lambda f: score_lookup.get(f, 0), reverse=True)
    return matched


# ---------------------------------------------------------------------------
# OCT biomarker preprocessing
# ---------------------------------------------------------------------------

def prepare_biomarker_features(df, labels, selected_features,
                                num_pipe=None, pca=None):
    fit_mode = num_pipe is None
    df_sel   = df.reindex(columns=selected_features).copy()
    for col in df_sel.columns:
        df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")

    if fit_mode:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler",  StandardScaler())])
        X        = num_pipe.fit_transform(df_sel.values)
        n_pca    = min(PCA_COMPS, X.shape[1], X.shape[0] - 1)
        pca      = PCA(n_components=n_pca, random_state=SEED)
        X_pca    = pca.fit_transform(X)
    else:
        X     = num_pipe.transform(df_sel.values)
        X_pca = pca.transform(X)

    return X, X_pca, dict(num_pipe=num_pipe, pca=pca,
                           selected_features=selected_features)


# ---------------------------------------------------------------------------
# ViT-B/16 encoder (frozen)
# ---------------------------------------------------------------------------

class ViTB16Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        vit              = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.patch_embed = vit.conv_proj
        self.encoder     = vit.encoder
        self.class_token = vit.class_token
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):          # (B, 3, 224, 224) -> (B, 768)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.encoder(x)[:, 0]


# ---------------------------------------------------------------------------
# Temporal transformer slice aggregator
# ---------------------------------------------------------------------------

class ImageSetTransformer(nn.Module):
    def __init__(self, embed_dim=VIT_B16_DIM, num_heads=4, num_layers=1,
                 ff_dim=1536, dropout=0.1, max_images=32):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.cls_token     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_images + 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ImageSetTransformer] layers={num_layers}, heads={num_heads}, "
              f"trainable={n:,}")

    def forward(self, x):          # (B, N, 768) -> (B, 768)
        B, N, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embedding[:, :N + 1]
        x   = self.transformer(x)
        return self.norm(x)[:, 0]


# ---------------------------------------------------------------------------
# Biomarker encoder
# ---------------------------------------------------------------------------

class BiomarkerEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Fusion model  (image + biomarkers)
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    def __init__(self, oct_dim: int, oct_embed_dim: int = 256,
                 num_classes: int = 2, dropout: float = 0.4,
                 use_temporal: bool = False, max_images: int = 32):
        super().__init__()
        self.use_temporal = use_temporal
        self.img_encoder  = ViTB16Encoder()

        if use_temporal:
            self.image_aggregator = ImageSetTransformer(
                max_images=max_images)
        else:
            self.image_aggregator = None

        self.oct_encoder = BiomarkerEncoder(oct_dim, oct_embed_dim)
        fused_dim = VIT_B16_DIM + oct_embed_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[FusionModel] fused_dim={fused_dim}, trainable={n:,} (ViT frozen)")

    def forward(self, images, oct_tab):   # images: (B, N, 3, H, W)
        B, N, C, H, W = images.shape
        feats = self.img_encoder(images.view(B * N, C, H, W)).view(B, N, -1)
        img_emb = (self.image_aggregator(feats) if self.use_temporal
                   else feats.mean(dim=1))
        oct_emb = self.oct_encoder(oct_tab)
        return self.fusion(torch.cat([img_emb, oct_emb], dim=1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatientDataset(Dataset):
    def __init__(self, subject_ids, oct_features, labels,
                 folder_map, slice_names, img_size=224, augment=False):
        self.subject_ids = subject_ids
        self.oct_tab     = torch.tensor(oct_features, dtype=torch.float32)
        self.labels      = labels
        self.folder_map  = folder_map
        self.slice_names = slice_names

        base = [transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])]
        aug  = ([transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.ColorJitter(brightness=0.15, contrast=0.15),
                 transforms.RandomRotation(10)] if augment else [])
        self.transform = transforms.Compose(aug + base)

    def _load_images(self, subject_id):
        key         = str(subject_id)[-13:]
        patient_dir = self.folder_map.get(key)
        stem_map: Dict[str, Path] = {}
        if patient_dir:
            for p in patient_dir.rglob("*"):
                if (p.suffix.lower() in IMAGE_EXTS
                        and not p.name.startswith(("._", "."))):
                    stem_map[p.stem.lower()] = p
        else:
            print(f"[WARN] No folder for '{subject_id}'")

        frames = []
        for sn in self.slice_names:
            path = stem_map.get(Path(sn).stem.lower())
            if path:
                img = Image.open(path).convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                for c in range(3):
                    ch = arr[..., c]
                    arr[..., c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
                frames.append(self.transform(img))
            else:
                frames.append(torch.zeros(3, 224, 224))
        return torch.stack(frames)

    def __len__(self):  return len(self.subject_ids)

    def __getitem__(self, idx):
        images = self._load_images(self.subject_ids[idx])
        return (images, self.oct_tab[idx],
                torch.tensor(self.labels[idx], dtype=torch.long))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold_model(model, y_tr, epochs, lr, patience,
                     best_path, last_path, train_ds, val_ds, device):
    counts    = np.bincount(y_tr)
    weights   = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    train_loader = DataLoader(train_ds, batch_size=len(train_ds),
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds),
                              shuffle=False, num_workers=0)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_auc = 0
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, oct_tab, labels in train_loader:
            images, oct_tab, labels = (images.to(device), oct_tab.to(device),
                                       labels.to(device))
            optimizer.zero_grad()
            loss = criterion(model(images, oct_tab), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss, val_preds, val_probs, val_labels = 0.0, [], [], []
        with torch.no_grad():
            for images, oct_tab, labels in val_loader:
                images, oct_tab, labels = (images.to(device), oct_tab.to(device),
                                           labels.to(device))
                logits = model(images, oct_tab)
                val_loss += criterion(logits, labels).item()
                p = F.softmax(logits, dim=-1)
                val_preds.extend(p.argmax(1).cpu().numpy())
                val_probs.extend(p[:, 1].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_auc = (roc_auc_score(val_labels, val_probs)
                   if len(set(val_labels)) > 1 else 0.0)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:03d} | loss={loss.item():.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
                  f"val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ best val_auc={best_val_auc:.4f} (epoch {epoch})")
        else:
            no_improve += 1
            torch.save(model.state_dict(), last_path)
            if no_improve >= patience:
                print(f"  [EarlyStop] epoch {epoch}, best_auc={best_val_auc:.4f}")
                break


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion=None,
             subject_ids=None, label_names=None, print_per_patient=False):
    model.eval()
    all_preds, all_probs_full, all_probs, all_labels = [], [], [], []
    total_loss = 0.0

    for images, oct_tab, labels in loader:
        images, oct_tab, labels = (images.to(device), oct_tab.to(device),
                                   labels.to(device))
        logits = model(images, oct_tab)
        if criterion is not None:
            total_loss += criterion(logits, labels).item()
        probs = F.softmax(logits, dim=-1)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_probs_full.extend(probs.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    auc    = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    report = classification_report(all_labels, all_preds,
                                   output_dict=True, zero_division=0)

    if print_per_patient:
        lnames = label_names or {0: "Healthy", 1: "Disease"}
        sep    = "─" * 90
        print(f"\n{sep}")
        print(f"  {'#':>4}  {'Subject ID':>22}  {'Actual':>10}  "
              f"{'Predicted':>10}  {'P(Healthy)':>12}  {'P(Disease)':>12}  {'OK':>4}")
        print(sep)
        ids = subject_ids or [f"P{i}" for i in range(len(all_labels))]
        for i, (sid, gt, pred, prb) in enumerate(
                zip(ids, all_labels, all_preds, all_probs_full)):
            ok = "✓" if int(gt) == int(pred) else "✗"
            print(f"  {i+1:>4}  {str(sid):>22}  "
                  f"{lnames.get(int(gt), str(gt)):>10}  "
                  f"{lnames.get(int(pred), str(pred)):>10}  "
                  f"{prb[0]:>12.4f}  {prb[1]:>12.4f}  {ok:>4}")
        n_ok = sum(1 for g, p in zip(all_labels, all_preds) if g == p)
        print(f"\n  Correct: {n_ok}/{len(all_labels)}  "
              f"Acc={n_ok/len(all_labels):.4f}  AUC={auc:.4f}")
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0.0
            spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0.0
            print(f"  Sensitivity={sens:.4f}  Specificity={spec:.4f}")
            print(f"  Confusion Matrix:  TP={cm[1,1]} FN={cm[1,0]} "
                  f"FP={cm[0,1]} TN={cm[0,0]}")
        print(sep)

    return {"loss": total_loss / max(len(loader), 1),
            "accuracy": report["accuracy"], "auc": auc,
            "report": report, "preds": all_preds,
            "probs": all_probs_full, "labels": all_labels}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_split_counts(name, y_arr, label_names):
    counts = {label_names.get(i, str(i)): int((y_arr == i).sum())
              for i in sorted(set(y_arr))}
    parts  = "  |  ".join(f"{k}: {v}" for k, v in counts.items())
    print(f"    {name:<18}: {len(y_arr):>3} subjects  |  {parts}")


def print_aggregate_results(fold_results, all_test_labels, all_test_preds,
                             label_names):
    print("\n" + "=" * 70)
    print("  AGGREGATE 5-FOLD RESULTS")
    print("=" * 70)
    accs  = [r["test_acc"]    for r in fold_results]
    aucs  = [r["test_auc"]    for r in fold_results]
    senss = [r["sensitivity"] for r in fold_results]
    specs = [r["specificity"] for r in fold_results]
    print(f"\n  {'Fold':>6}  {'Acc':>8}  {'AUC':>8}  {'Sens':>7}  {'Spec':>7}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}")
    for r in fold_results:
        print(f"  {r['fold']:>6}  {r['test_acc']:>8.4f}  {r['test_auc']:>8.4f}  "
              f"{r['sensitivity']:>7.4f}  {r['specificity']:>7.4f}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}")
    print(f"  {'Mean':>6}  {np.mean(accs):>8.4f}  {np.mean(aucs):>8.4f}  "
          f"{np.mean(senss):>7.4f}  {np.mean(specs):>7.4f}")
    print(f"  {'Std':>6}  {np.std(accs):>8.4f}  {np.std(aucs):>8.4f}  "
          f"{np.std(senss):>7.4f}  {np.std(specs):>7.4f}")
    print(f"\n  Pooled Classification Report:")
    print(classification_report(all_test_labels, all_test_preds,
                                target_names=list(label_names.values()),
                                zero_division=0))
    cm = confusion_matrix(all_test_labels, all_test_preds)
    if cm.shape == (2, 2):
        ps = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        pc = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0
        print(f"  Pooled Confusion Matrix:  TP={cm[1,1]} FN={cm[1,0]} "
              f"FP={cm[0,1]} TN={cm[0,0]}")
        print(f"  Pooled Sensitivity={ps:.4f}  Specificity={pc:.4f}")
    print("=" * 70)
