"""
=============================================================================
ViT-B/16 + OCT Biomarkers Fusion Pipeline — Small-Dataset Fixes
=============================================================================

Run with:
    python vitb16_oct_kfold_tm.py t    ← temporal transformer (9 slices, en face)
    python vitb16_oct_kfold_tm.py m    ← mean pooling (9 slices, en face, default)
    python vitb16_oct_kfold_tm.py v    ← mean pooling (32 slices, en face volume)
    python vitb16_oct_kfold_tm.py s    ← mean pooling (32 slices, second OCT modality)

Modes:
    t  : ImageSetTransformer aggregates N per-slice ViT-B/16 embeddings
         (CLS token + 1 transformer encoder layer, 4 heads, ff_dim=1536).
         Uses en face OCT, 9 slices.
    m  : Mean-pool across 9 en face OCT slices.
    v  : Mean-pool across all 32 en face OCT volume slices (slice_1 … slice_32).
    s  : Mean-pool across 32 slices from a SECOND OCT modality
         (different image root folder, same slice naming convention).
         Update IMAGE_ROOT_MODALITY2 below to point to your data.
=============================================================================
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

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
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VIT_B16_DIM = 768
PCA_COMPS   = 15

# Slice lists
ENFACE_9_SLICES   = [f"slice_{i}" for i in range(9, 0, -1)]   # slice_9 … slice_1
VOLUME_32_SLICES  = [f"slice_{i}" for i in range(32, 0, -1)]  # slice_32 … slice_1
MODALITY2_SLICES  = [f"slice_{i}" for i in range(32, 0, -1)]  # 32 slices, second modality

# ---- Path for second OCT modality images (mode 's') ----
# Update this to the folder that contains the second modality patient folders.
IMAGE_ROOT_MODALITY2 = "/home/suhel.khan/Dimentia_Project/Image_data/OCT_modality2_images_b1_b2"


# =============================================================================
# PARSE ARGUMENT
# =============================================================================

def parse_mode() -> str:
    """
    Returns 't', 'm', 'v', or 's'.
      t — temporal transformer, en face, 9 slices
      m — mean pooling,         en face, 9 slices  (default)
      v — mean pooling,         en face, 32 slices (full volume)
      s — mean pooling,         second OCT modality, 32 slices
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", nargs="?", default="m",
                        choices=["t", "m", "v", "s"],
                        help="t=temporal transformer, m=mean pooling (9 slices), "
                             "v=volume 32 slices, s=second OCT modality 32 slices")
    args, _ = parser.parse_known_args()
    mode = args.mode.lower()
    desc = {
        "t": "Temporal Transformer (1 layer, 4 heads) — en face, 9 slices",
        "m": "Mean Average Pooling — en face, 9 slices",
        "v": "Mean Average Pooling — en face volume, 32 slices",
        "s": "Mean Average Pooling — second OCT modality, 32 slices",
    }[mode]
    print(f"[Mode] {desc}")
    return mode


# =============================================================================
# LOGGER
# =============================================================================

import sys as _sys

class Tee:
    def __init__(self, log_path: str, mode: str = "w"):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._file   = open(log_path, mode, encoding="utf-8")
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


# =============================================================================
# 0. FOLDER LOOKUP
# =============================================================================

def build_subject_folder_map(image_root: str) -> Dict[str, Path]:
    image_root = Path(image_root)
    folder_map: Dict[str, Path] = {}
    for d in image_root.iterdir():
        if d.is_dir():
            folder_map[d.name[-13:]] = d
    print(f"[INFO] Found {len(folder_map)} patient folders under {image_root}")
    return folder_map


# =============================================================================
# 1. FEATURE SCORE CSV LOADER
# =============================================================================

def load_features_by_threshold(feature_score_csv, feature_name_col,
                                score_col, threshold, data_columns):
    scores_df = pd.read_csv(feature_score_csv)
    assert feature_name_col in scores_df.columns
    assert score_col in scores_df.columns

    passing      = scores_df[scores_df[score_col] >= threshold]
    selected     = passing[feature_name_col].astype(str).tolist()
    data_col_set = set(data_columns)
    matched      = [f for f in selected if f in data_col_set]
    unmatched    = [f for f in selected if f not in data_col_set]

    print(f"\n[FeatureSelection]  threshold >= {threshold}")
    print(f"  In score CSV : {len(scores_df)}  |  Passing : {len(selected)}"
          f"  |  Matched in data : {len(matched)}")
    if unmatched:
        print(f"  [WARN] {len(unmatched)} unmatched: {unmatched[:5]}")
    if not matched:
        raise ValueError("No features matched.")

    score_lookup = dict(zip(scores_df[feature_name_col].astype(str),
                            scores_df[score_col]))
    matched.sort(key=lambda f: score_lookup.get(f, 0), reverse=True)
    return matched


# =============================================================================
# 2. ViT-B/16 ENCODER  — always fully frozen
# =============================================================================

class ViTB16Encoder(nn.Module):
    """Fully frozen ViT-B/16. Used as a fixed feature extractor."""

    def __init__(self):
        super().__init__()
        vit              = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.patch_embed = vit.conv_proj
        self.encoder     = vit.encoder
        self.class_token = vit.class_token

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):             # x: (B, 3, 224, 224)
        B   = x.shape[0]
        x   = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        return self.encoder(x)[:, 0]  # (B, 768)


# =============================================================================
# 2b. IMAGE SET TRANSFORMER  — used when mode == 't'
# =============================================================================

class ImageSetTransformer(nn.Module):
    """
    Aggregates N per-slice ViT-B/16 embeddings into one patient-level
    embedding via a small transformer with a learnable CLS token.
    """
    def __init__(self, embed_dim: int = VIT_B16_DIM,
                 num_heads: int = 4,
                 num_layers: int = 1,
                 ff_dim: int = 1536,
                 dropout: float = 0.1,
                 max_images: int = 32):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.cls_token     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_images + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)

        n_trainable = sum(p.numel() for p in self.parameters()
                          if p.requires_grad)
        print(f"[ImageSetTransformer] layers={num_layers}, heads={num_heads}  "
              f"Trainable params: {n_trainable:,}")

    def forward(self, x, key_padding_mask=None):   # x: (B, N, 768)
        B, N, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)            # (B, N+1, 768)
        x   = x + self.pos_embedding[:, :N + 1]
        x   = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)[:, 0]                   # (B, 768)


# =============================================================================
# 3. OCT BIOMARKER ENCODER
# =============================================================================

class BiomarkerEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 4. FUSION MODEL
#
#    use_temporal=True  : ViTB16Encoder × N slices → ImageSetTransformer → (B, 768)
#    use_temporal=False : ViTB16Encoder × N slices → mean-pool → (B, 768)
#
#    Works for any N slices and any image modality.
# =============================================================================

class FusionModel(nn.Module):
    def __init__(self, oct_dim: int, oct_embed_dim: int = 64,
                 num_classes: int = 2, dropout: float = 0.4,
                 use_temporal: bool = False, max_images: int = 32):
        super().__init__()
        self.use_temporal = use_temporal
        self.img_encoder = ViTB16Encoder()

        if use_temporal:
            self.image_aggregator = ImageSetTransformer(
                embed_dim  = VIT_B16_DIM,
                num_heads  = 4,
                num_layers = 1,
                ff_dim     = 1536,
                dropout    = 0.1,
                max_images = max_images,
            )
        else:
            self.image_aggregator = None

        self.oct_encoder = BiomarkerEncoder(oct_dim, oct_embed_dim)

        fused_dim = VIT_B16_DIM + oct_embed_dim
        print(f"[FusionModel] Aggregation: "
              f"{'Temporal Transformer (1L-4H)' if use_temporal else 'Mean Pooling'}")
        print(f"[FusionModel] fused_dim={fused_dim}  "
              f"(img=768 | oct={oct_embed_dim})")

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters()
                          if p.requires_grad)
        print(f"[FusionModel] Trainable params: {n_trainable:,}  "
              f"(ViT is frozen)")

    def forward(self, images, oct_tabular):
        B, N, C, H, W = images.shape
        feats = self.img_encoder(images.view(B * N, C, H, W))  # (B*N, 768)
        feats = feats.view(B, N, -1)                           # (B, N, 768)

        if self.use_temporal:
            img_emb = self.image_aggregator(feats)             # (B, 768)
        else:
            img_emb = feats.mean(dim=1)                        # (B, 768)

        oct_emb = self.oct_encoder(oct_tabular)                # (B, oct_embed)
        fused   = torch.cat([img_emb, oct_emb], dim=1)
        return self.fusion(fused)


# =============================================================================
# 5. DATASET
# =============================================================================

class PatientDataset(Dataset):
    def __init__(self, subject_ids, oct_features, labels,
                 folder_map, slice_names, img_size=224, augment=False):
        self.subject_ids = subject_ids
        self.oct_tab     = torch.tensor(oct_features, dtype=torch.float32)
        self.labels      = labels
        self.folder_map  = folder_map
        self.slice_names = slice_names

        base_tf = [transforms.Resize((img_size, img_size)),
                   transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])]
        aug_tf  = ([transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.RandomRotation(10)] if augment else [])
        self.transform = transforms.Compose(aug_tf + base_tf)

    def _load_patient_images(self, subject_id):
        lookup_key  = str(subject_id)[-13:]
        patient_dir = self.folder_map.get(lookup_key, None)
        stem_map: Dict[str, Path] = {}
        if patient_dir is not None:
            for p in patient_dir.rglob("*"):
                if (p.suffix.lower() in IMAGE_EXTS
                        and not p.name.startswith("._")
                        and not p.name.startswith(".")):
                    stem_map[p.stem.lower()] = p
        else:
            print(f"[WARN] No folder for '{subject_id}'")

        frames, mask = [], []
        for sn in self.slice_names:
            stem = Path(sn).stem.lower()
            path = stem_map.get(stem, None)
            if path is not None:
                img = Image.open(path).convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                for c in range(3):
                    ch = arr[..., c]
                    arr[..., c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                img_norm = Image.fromarray(
                    (arr * 255).clip(0, 255).astype(np.uint8))
                frames.append(self.transform(img_norm))
                mask.append(False)
            else:
                frames.append(torch.zeros(3, 224, 224))
                mask.append(True)

        return torch.stack(frames), torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        images, mask = self._load_patient_images(self.subject_ids[idx])
        return (images, self.oct_tab[idx],
                torch.tensor(self.labels[idx], dtype=torch.long))


# =============================================================================
# 6. OCT PREPROCESSING
# =============================================================================

def prepare_biomarker_features(df, labels, selected_features,
                                num_pipe=None, pca=None):
    fit_mode = num_pipe is None
    df_sel   = df.reindex(columns=selected_features).copy()
    for col in df_sel.columns:
        df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")

    if fit_mode:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler",  StandardScaler())])
        X_model  = num_pipe.fit_transform(df_sel.values)
        n_pca    = min(PCA_COMPS, X_model.shape[1], X_model.shape[0] - 1)
        pca      = PCA(n_components=n_pca, random_state=SEED)
        X_xgb    = pca.fit_transform(X_model)
    else:
        X_model = num_pipe.transform(df_sel.values)
        X_xgb   = pca.transform(X_model)

    return X_model, X_xgb, dict(num_pipe=num_pipe, pca=pca,
                                 selected_features=selected_features)


# =============================================================================
# 7. TRAINING
# =============================================================================

def train_fold_model(model, X_tr, y_tr, X_vl, y_vl, epochs, lr,
                     patience, best_path, last_path,
                     train_ds, val_ds, batch_size):

    counts  = np.bincount(y_tr)
    weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    train_loader = DataLoader(train_ds, batch_size=len(train_ds),
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds),
                              shuffle=False, num_workers=0)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_auc  = 0
    best_val_loss = float("inf")
    no_improve    = 0

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
            print(f"  Epoch {epoch:03d} | Train loss: {loss.item():.4f} | "
                  f"Val loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ New best val auc={best_val_auc:.4f}  "
                  f"(epoch {epoch}, val_auc={val_auc:.4f})")
        else:
            no_improve += 1
            torch.save(model.state_dict(), last_path)
            if no_improve >= patience:
                print(f"  [EarlyStop] epoch {epoch}  "
                      f"best val auc: {best_val_auc:.4f}")
                break

    return best_val_loss


@torch.no_grad()
def evaluate(model, loader, criterion=None, subject_ids=None,
             label_names=None, print_per_patient=False):
    model.eval()
    all_preds, all_probs_full, all_probs, all_labels = [], [], [], []
    total_loss = 0.0

    for batch in loader:
        images, oct_tab, labels = batch
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
        sep    = "─" * 92
        print(f"\n{sep}")
        print(f"  {'#':>4}  {'Subject ID':>22}  {'Actual':>10}  "
              f"{'Predicted':>10}  {'P(Healthy)':>12}  {'P(Disease)':>12}  {'OK':>4}")
        print(sep)
        ids = subject_ids or [f"P{i}" for i in range(len(all_labels))]
        for i, (sid, gt, pred, prb) in enumerate(
                zip(ids, all_labels, all_preds, all_probs_full)):
            print(f"  {i+1:>4}  {str(sid):>22}  "
                  f"{lnames.get(int(gt), str(gt)):>10}  "
                  f"{lnames.get(int(pred), str(pred)):>10}  "
                  f"{prb[0]:>12.4f}  {prb[1]:>12.4f}  "
                  f"{'✓' if int(gt)==int(pred) else '✗':>4}")
        print(sep)
        n_ok = sum(1 for g, p in zip(all_labels, all_preds) if g == p)
        print(f"\n  Correct: {n_ok}/{len(all_labels)}  "
              f"Accuracy: {n_ok/len(all_labels):.4f}  AUC: {auc:.4f}")
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0.0
            spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0.0
            print(f"\n  Confusion Matrix:")
            print(f"                  Pred Healthy  Pred Disease")
            print(f"    GT Healthy    {cm[0,0]:>12}  {cm[0,1]:>12}")
            print(f"    GT Disease    {cm[1,0]:>12}  {cm[1,1]:>12}")
            print(f"\n  Sensitivity: {sens:.4f}  |  Specificity: {spec:.4f}")
        print(sep)

    return {"loss": total_loss / max(len(loader), 1),
            "accuracy": report["accuracy"],
            "auc": auc, "report": report,
            "preds": all_preds, "probs": all_probs_full, "labels": all_labels}


# =============================================================================
# 8. XGBoost BASELINE
# =============================================================================

def train_xgboost(X_xgb, y, n_splits=5):
    print(f"\n[XGBoost] PCA({PCA_COMPS}) features, {n_splits}-fold CV...")
    y     = np.array(y, dtype=np.int64)
    n_neg = max(int((y == 0).sum()), 1)
    n_pos = max(int((y == 1).sum()), 1)
    clf   = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", scale_pos_weight=n_neg / n_pos,
        random_state=SEED, verbosity=0)
    scores = cross_val_score(
        clf, X_xgb, y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED),
        scoring="roc_auc")
    print(f"  AUC per fold: {scores.round(3)}")
    print(f"  Mean AUC    : {scores.mean():.3f} ± {scores.std():.3f}")
    clf.fit(X_xgb, y)
    return clf


# =============================================================================
# 9. SPLIT CLASS-COUNT LOGGER
# =============================================================================

def log_split_counts(name, y_arr, label_names):
    counts = {label_names.get(i, str(i)): int((y_arr == i).sum())
              for i in sorted(set(y_arr))}
    parts  = "  |  ".join(f"{k}: {v}" for k, v in counts.items())
    print(f"    {name:<18}: {len(y_arr):>3} subjects  |  {parts}")


# =============================================================================
# 10. 5-FOLD CROSS-VALIDATION
# =============================================================================

def run_kfold(all_subject_ids, df_biomarkers, y, label_names,
              selected_features, folder_map, slice_names, cfg,
              use_temporal: bool):

    n_splits       = cfg["n_splits"]
    inner_val_frac = cfg["inner_val_frac"]
    skf            = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=SEED)

    fold_results    = []
    all_test_preds  = []
    all_test_labels = []

    print("\n" + "=" * 70)
    print(f"  {n_splits}-FOLD CROSS-VALIDATION")
    print(f"  Aggregation: "
          f"{'Temporal Transformer (1L,4H)' if use_temporal else 'Mean Pooling'}")
    print(f"  Slices ({len(slice_names)}): {slice_names}")
    print(f"  Selected features: {len(selected_features)}")
    print("=" * 70)

    for fold, (train_fold_idx, test_idx) in enumerate(
            skf.split(np.arange(len(y)), y), 1):

        print(f"\n{'═' * 70}")
        print(f"  FOLD {fold} / {n_splits}")
        print(f"{'═' * 70}")

        y_train_fold = y[train_fold_idx]
        y_test_fold  = y[test_idx]

        inner_tr_local, inner_vl_local = train_test_split(
            np.arange(len(train_fold_idx)),
            test_size    = inner_val_frac,
            stratify     = y_train_fold,
            random_state = SEED)

        global_tr_idx  = train_fold_idx[inner_tr_local]
        global_vl_idx  = train_fold_idx[inner_vl_local]
        y_inner_tr     = y[global_tr_idx]
        y_inner_vl     = y[global_vl_idx]

        print(f"\n  Split class distribution:")
        log_split_counts("Inner train", y_inner_tr, label_names)
        log_split_counts("Inner val",   y_inner_vl, label_names)
        log_split_counts("Test fold",   y_test_fold, label_names)

        id_inner_tr  = [all_subject_ids[i] for i in global_tr_idx]
        id_inner_vl  = [all_subject_ids[i] for i in global_vl_idx]
        id_test_fold = [all_subject_ids[i] for i in test_idx]

        df_bio_tr = df_biomarkers.iloc[global_tr_idx].reset_index(drop=True)
        df_bio_vl = df_biomarkers.iloc[global_vl_idx].reset_index(drop=True)
        df_bio_ts = df_biomarkers.iloc[test_idx].reset_index(drop=True)

        X_tr_model, X_tr_xgb, bio_prep = prepare_biomarker_features(
            df_bio_tr, y_inner_tr, selected_features)
        X_vl_model, _, _ = prepare_biomarker_features(
            df_bio_vl, y_inner_vl, selected_features,
            num_pipe=bio_prep["num_pipe"], pca=bio_prep["pca"])
        X_ts_model, _, _ = prepare_biomarker_features(
            df_bio_ts, y_test_fold, selected_features,
            num_pipe=bio_prep["num_pipe"], pca=bio_prep["pca"])

        print(f"\n  OCT shapes — train: {X_tr_model.shape}  "
              f"val: {X_vl_model.shape}  test: {X_ts_model.shape}")

        train_ds = PatientDataset(id_inner_tr, X_tr_model, y_inner_tr,
                                  folder_map, slice_names, augment=True)
        val_ds   = PatientDataset(id_inner_vl, X_vl_model, y_inner_vl,
                                  folder_map, slice_names, augment=False)
        test_ds  = PatientDataset(id_test_fold, X_ts_model, y_test_fold,
                                  folder_map, slice_names, augment=False)

        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                                 shuffle=False, num_workers=0)

        model = FusionModel(
            oct_dim       = X_tr_model.shape[1],
            oct_embed_dim = cfg["oct_embed_dim"],
            dropout       = cfg["dropout"],
            use_temporal  = use_temporal,
            max_images    = len(slice_names),
        ).to(device)

        best_path = os.path.join(cfg["weights_dir"], f"fold{fold}_best.pth")
        last_path = os.path.join(cfg["weights_dir"], f"fold{fold}_last.pth")

        print(f"\n  Training fold {fold}...")
        train_fold_model(
            model, X_tr_model, y_inner_tr, X_vl_model, y_inner_vl,
            epochs    = cfg["epochs"],
            lr        = cfg["lr"],
            patience  = cfg["patience"],
            best_path = best_path,
            last_path = last_path,
            train_ds  = train_ds,
            val_ds    = val_ds,
            batch_size = cfg["batch_size"],
        )

        model.load_state_dict(torch.load(best_path, map_location=device))
        counts  = np.bincount(y_inner_tr)
        weights = torch.tensor(1.0 / (counts + 1e-6),
                               dtype=torch.float32).to(device)
        criterion    = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        test_metrics = evaluate(model, test_loader, criterion,
                                subject_ids=id_test_fold,
                                label_names=label_names,
                                print_per_patient=True)

        cm   = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
        sens = (cm[1,1]/(cm[1,1]+cm[1,0])
                if cm.shape == (2,2) and (cm[1,1]+cm[1,0]) > 0 else 0.0)
        spec = (cm[0,0]/(cm[0,0]+cm[0,1])
                if cm.shape == (2,2) and (cm[0,0]+cm[0,1]) > 0 else 0.0)

        print(f"\n  ── Fold {fold} Results ──")
        print(f"  Test Acc     : {test_metrics['accuracy']:.4f}")
        print(f"  Test AUC     : {test_metrics['auc']:.4f}")
        print(f"  Sensitivity  : {sens:.4f}")
        print(f"  Specificity  : {spec:.4f}")
        print(classification_report(test_metrics["labels"],
                                    test_metrics["preds"],
                                    target_names=list(label_names.values()),
                                    zero_division=0))

        fold_results.append(dict(fold=fold,
                                 test_acc=test_metrics["accuracy"],
                                 test_auc=test_metrics["auc"],
                                 sensitivity=sens, specificity=spec))
        all_test_preds.extend(test_metrics["preds"])
        all_test_labels.extend(test_metrics["labels"])

    print("\n" + "=" * 70)
    print("  AGGREGATE 5-FOLD RESULTS")
    print("=" * 70)

    accs  = [r["test_acc"]    for r in fold_results]
    aucs  = [r["test_auc"]    for r in fold_results]
    senss = [r["sensitivity"] for r in fold_results]
    specs = [r["specificity"] for r in fold_results]

    print(f"\n  {'Fold':>6}  {'Test Acc':>9}  {'Test AUC':>9}  "
          f"{'Sens':>7}  {'Spec':>7}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*7}")
    for r in fold_results:
        print(f"  {r['fold']:>6}  {r['test_acc']:>9.4f}  "
              f"{r['test_auc']:>9.4f}  "
              f"{r['sensitivity']:>7.4f}  {r['specificity']:>7.4f}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*7}")
    print(f"  {'Mean':>6}  {np.mean(accs):>9.4f}  {np.mean(aucs):>9.4f}  "
          f"{np.mean(senss):>7.4f}  {np.mean(specs):>7.4f}")
    print(f"  {'Std':>6}  {np.std(accs):>9.4f}  {np.std(aucs):>9.4f}  "
          f"{np.std(senss):>7.4f}  {np.std(specs):>7.4f}")

    print(f"\n  Aggregate Classification Report (all folds pooled):")
    print(classification_report(all_test_labels, all_test_preds,
                                target_names=list(label_names.values()),
                                zero_division=0))

    cm_all = confusion_matrix(all_test_labels, all_test_preds)
    if cm_all.shape == (2, 2):
        print(f"  Pooled Confusion Matrix:")
        print(f"                  Pred Healthy  Pred Disease")
        print(f"    GT Healthy    {cm_all[0,0]:>12}  {cm_all[0,1]:>12}")
        print(f"    GT Disease    {cm_all[1,0]:>12}  {cm_all[1,1]:>12}")
        ps = cm_all[1,1]/(cm_all[1,1]+cm_all[1,0]) if (cm_all[1,1]+cm_all[1,0])>0 else 0
        pc = cm_all[0,0]/(cm_all[0,0]+cm_all[0,1]) if (cm_all[0,0]+cm_all[0,1])>0 else 0
        print(f"\n  Pooled Sensitivity : {ps:.4f}")
        print(f"  Pooled Specificity : {pc:.4f}")
    print("=" * 70)
    return fold_results


# =============================================================================
# 11. MAIN
# =============================================================================

def main(mode: str):
    use_temporal = (mode == "t")

    if mode == "s":
        # Second OCT modality — different image folder, 32 slices, mean pooling
        slice_names = MODALITY2_SLICES
        image_root  = IMAGE_ROOT_MODALITY2
        mode_label  = "oct_modality2_slices_32"
    elif mode == "v":
        # En face OCT full volume — same folder as 'm'/'t', 32 slices
        slice_names = VOLUME_32_SLICES
        image_root  = "/home/suhel.khan/Dimentia_Project/Image_data/OCT_enface_images_all_b1_b2"
        mode_label  = "oct_volume_slices_32"
    elif mode == "t":
        slice_names = ENFACE_9_SLICES
        image_root  = "/home/suhel.khan/Dimentia_Project/Image_data/OCT_enface_images_all_b1_b2"
        mode_label  = "temporal_transformer"
    else:  # 'm'
        slice_names = ENFACE_9_SLICES
        image_root  = "/home/suhel.khan/Dimentia_Project/Image_data/OCT_enface_images_all_b1_b2"
        mode_label  = "mean_pooling"

    print("=" * 70)
    print(f"  ViT-B/16 + OCT Fusion — 5-Fold CV  [{mode_label}]")
    print(f"  Slices   : {len(slice_names)}")
    print(f"  ImageRoot: {image_root}")
    print("=" * 70)

    CSV_PATH            = "/home/suhel.khan/Dimentia_Project/LKC_Dimentia_structured_data/merged_output_b1_b2_CNvsCI_Refined.csv"
    TARGET_COL          = "Dia"
    SUBJECT_ID_COL      = "Subj_ID"
    WEIGHTS_DIR         = (
        f"/home/suhel.khan/Dimentia_Project/Weights/"
        f"5fold_2_thresh_CN_CI_{mode_label}"
    )

    FEATURE_SCORE_CSV   = "/home/suhel.khan/Dimentia_Project/Feature_Engineering/b1andb2_OCT_biomarkers_feature_scores_CN_CI_real.csv"
    FEATURE_NAME_COL    = "feature"
    FEATURE_SCORE_COL   = "score"
    SCORE_THRESHOLD     = 2.0
    BIOMARKER_START_COL = 1

    cfg = dict(
        n_splits        = 5,
        inner_val_frac  = 0.35,
        batch_size      = 8,
        epochs          = 200,
        lr              = 3e-4,
        patience        = 10,
        oct_embed_dim   = 256,
        dropout         = 0.4,
        weights_dir     = WEIGHTS_DIR,
    )

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"\n  Device     : {device}")
    print(f"  Aggregation: "
          f"{'Temporal Transformer (1L, 4H)' if use_temporal else 'Mean Average Pooling'}")
    print(f"  Slice count: {len(slice_names)}")

    df           = pd.read_csv(CSV_PATH)
    y_raw        = df.pop(TARGET_COL).values
    subject_ids  = df.pop(SUBJECT_ID_COL).astype(str).tolist()
    df_bio       = df.iloc[:, BIOMARKER_START_COL:].copy()
    all_bio_cols = df_bio.columns.tolist()

    unique_labels = sorted(set(y_raw))
    if list(unique_labels) != [0, 1]:
        label_map = {v: i for i, v in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_raw], dtype=np.int64)
        print(f"  Labels remapped: {label_map}")
    else:
        y = y_raw.astype(np.int64)

    label_names = {0: "Healthy", 1: "Disease"}
    print(f"  Total: {len(y)}  |  Healthy: {int((y==0).sum())}  "
          f"|  Disease: {int((y==1).sum())}")

    selected_features = load_features_by_threshold(
        FEATURE_SCORE_CSV, FEATURE_NAME_COL,
        FEATURE_SCORE_COL, SCORE_THRESHOLD, all_bio_cols)
    print(f"  Features selected: {len(selected_features)}")

    folder_map = build_subject_folder_map(image_root)

    run_kfold(subject_ids, df_bio, y, label_names,
              selected_features, folder_map, slice_names, cfg,
              use_temporal=use_temporal)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mode       = parse_mode()
    mode_label = {
        "t": "temporal_transformer",
        "m": "mean_pooling",
        "v": "oct_volume_slices_32",
        "s": "oct_modality2_slices_32",
    }[mode]
    LOG_PATH = (
        f"/home/suhel.khan/Dimentia_Project/Results/CN_CI_after_B2/"
        f"results_vitb16_5fold_fixed_{mode_label}.txt"
    )
    sys.stdout = Tee(LOG_PATH)
    try:
        main(mode)
    finally:
        sys.stdout.close()
