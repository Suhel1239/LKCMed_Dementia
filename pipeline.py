"""
=============================================================================
ViT-B/16 + OCT Biomarkers Fusion Pipeline
=============================================================================

Run:
    python pipeline.py enface              # en face OCT, mean pooling (default)
    python pipeline.py enface t            # en face OCT, temporal transformer
    python pipeline.py octa                # OCTA images, 32 slices, mean pooling
    python pipeline.py structured          # structured biomarkers only (XGBoost + MLP)

Branches:
    enface     : 9 en face OCT slices + biomarkers -> ViT-B/16 fusion
    octa       : 32 OCTA slices       + biomarkers -> ViT-B/16 fusion
    structured : OCT biomarkers only  (no images)  -> XGBoost + MLP
=============================================================================
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# PATHS  (edit here)
# =============================================================================

CSV_PATH            = "/home/suhel.khan/Dimentia_Project/LKC_Dimentia_structured_data/merged_output_b1_b2_CNvsCI_Refined.csv"
TARGET_COL          = "Dia"
SUBJECT_ID_COL      = "Subj_ID"
BIOMARKER_START_COL = 1

FEATURE_SCORE_CSV   = "/home/suhel.khan/Dimentia_Project/Feature_Engineering/b1andb2_OCT_biomarkers_feature_scores_CN_CI_real.csv"
FEATURE_NAME_COL    = "feature"
FEATURE_SCORE_COL   = "score"
SCORE_THRESHOLD     = 2.0

IMAGE_ROOT_ENFACE   = "/home/suhel.khan/Dimentia_Project/Image_data/OCT_enface_images_all_b1_b2"
IMAGE_ROOT_OCTA     = "/home/suhel.khan/Dimentia_Project/Image_data/OCTA_images_b1_b2"  # <-- update

WEIGHT_BASE         = "/home/suhel.khan/Dimentia_Project/Weights"
LOG_DIR             = "/home/suhel.khan/Dimentia_Project/Results/CN_CI_after_B2"

SLICES_ENFACE       = [f"slice_{i}" for i in range(9, 0, -1)]   # 9 slices
SLICES_OCTA         = [f"slice_{i}" for i in range(32, 0, -1)]  # 32 slices

# training hyperparameters (shared across image branches)
CFG = dict(
    n_splits       = 5,
    inner_val_frac = 0.35,
    epochs         = 200,
    lr             = 3e-4,
    patience       = 10,
    oct_embed_dim  = 256,
    dropout        = 0.4,
    batch_size     = 8,
)

# structured-only branch extras
MLP_CFG = dict(
    epochs   = 300,
    lr       = 3e-4,
    patience = 20,
    hidden   = 128,
    dropout  = 0.3,
)


# =============================================================================
# LOGGER
# =============================================================================

import sys as _sys

class Tee:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._file   = open(log_path, "w", encoding="utf-8")
        self._stdout = _sys.__stdout__
    def write(self, msg):
        self._stdout.write(msg); self._file.write(msg)
    def flush(self):
        self._stdout.flush(); self._file.flush()
    def close(self):
        self.flush(); self._file.close(); _sys.stdout = self._stdout


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def build_subject_folder_map(image_root: str) -> Dict[str, Path]:
    root = Path(image_root)
    m    = {d.name[-13:]: d for d in root.iterdir() if d.is_dir()}
    print(f"[INFO] {len(m)} patient folders under {root}")
    return m


def load_features_by_threshold(score_csv, name_col, score_col,
                                threshold, data_columns):
    df       = pd.read_csv(score_csv)
    passing  = df[df[score_col] >= threshold]
    selected = passing[name_col].astype(str).tolist()
    col_set  = set(data_columns)
    matched  = [f for f in selected if f in col_set]
    print(f"[Features] threshold>={threshold}: {len(matched)} matched")
    if not matched:
        raise ValueError("No features matched threshold.")
    lookup = dict(zip(df[name_col].astype(str), df[score_col]))
    matched.sort(key=lambda f: lookup.get(f, 0), reverse=True)
    return matched


def prepare_biomarker_features(df, labels, selected, num_pipe=None, pca=None):
    df_sel = df.reindex(columns=selected).copy()
    for c in df_sel.columns:
        df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")
    if num_pipe is None:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("scl", StandardScaler())])
        X      = num_pipe.fit_transform(df_sel.values)
        n_pca  = min(PCA_COMPS, X.shape[1], X.shape[0] - 1)
        pca    = PCA(n_components=n_pca, random_state=SEED)
        X_pca  = pca.fit_transform(X)
    else:
        X     = num_pipe.transform(df_sel.values)
        X_pca = pca.transform(X)
    return X, X_pca, dict(num_pipe=num_pipe, pca=pca)


def log_split_counts(name, y, lnames):
    parts = "  |  ".join(f"{lnames.get(i,i)}: {int((y==i).sum())}"
                         for i in sorted(set(y)))
    print(f"    {name:<18}: {len(y):>3}  |  {parts}")


def print_aggregate(fold_results, all_labels, all_preds, label_names):
    print("\n" + "=" * 70 + "\n  AGGREGATE RESULTS\n" + "=" * 70)
    accs  = [r["acc"]  for r in fold_results]
    aucs  = [r["auc"]  for r in fold_results]
    senss = [r["sens"] for r in fold_results]
    specs = [r["spec"] for r in fold_results]
    print(f"  {'Fold':>5}  {'Acc':>7}  {'AUC':>7}  {'Sens':>7}  {'Spec':>7}")
    for r in fold_results:
        print(f"  {r['fold']:>5}  {r['acc']:>7.4f}  {r['auc']:>7.4f}  "
              f"{r['sens']:>7.4f}  {r['spec']:>7.4f}")
    print(f"  {'Mean':>5}  {np.mean(accs):>7.4f}  {np.mean(aucs):>7.4f}  "
          f"{np.mean(senss):>7.4f}  {np.mean(specs):>7.4f}")
    print(f"  {'Std':>5}  {np.std(accs):>7.4f}  {np.std(aucs):>7.4f}  "
          f"{np.std(senss):>7.4f}  {np.std(specs):>7.4f}")
    print("\n" + classification_report(
        all_labels, all_preds,
        target_names=list(label_names.values()), zero_division=0))
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        ps = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        pc = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0
        print(f"  Pooled  TP={cm[1,1]} FN={cm[1,0]} FP={cm[0,1]} TN={cm[0,0]}")
        print(f"  Pooled Sensitivity={ps:.4f}  Specificity={pc:.4f}")
    print("=" * 70)


# =============================================================================
# IMAGE BRANCH: MODEL COMPONENTS
# =============================================================================

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
    def forward(self, x):            # (B, 3, 224, 224) -> (B, 768)
        B   = x.shape[0]
        x   = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        return self.encoder(torch.cat([cls, x], dim=1))[:, 0]


class ImageSetTransformer(nn.Module):
    def __init__(self, embed_dim=VIT_B16_DIM, num_heads=4,
                 num_layers=1, ff_dim=1536, dropout=0.1, max_slices=32):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_emb   = nn.Parameter(
            torch.randn(1, max_slices + 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):           # (B, N, 768) -> (B, 768)
        B, N, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_emb[:, :N + 1]
        return self.norm(self.transformer(x))[:, 0]


class BiomarkerEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, oct_dim, oct_embed_dim=256, num_classes=2,
                 dropout=0.4, use_temporal=False, max_slices=32):
        super().__init__()
        self.use_temporal = use_temporal
        self.img_encoder  = ViTB16Encoder()
        self.image_agg    = (ImageSetTransformer(max_slices=max_slices)
                             if use_temporal else None)
        self.oct_encoder  = BiomarkerEncoder(oct_dim, oct_embed_dim)
        fused_dim = VIT_B16_DIM + oct_embed_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),                  nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[FusionModel] fused={fused_dim}  trainable={n:,}  "
              f"agg={'transformer' if use_temporal else 'mean_pool'}  "
              f"(ViT frozen)")

    def forward(self, images, oct_tab):  # images: (B, N, 3, H, W)
        B, N, C, H, W = images.shape
        feats   = self.img_encoder(images.view(B*N, C, H, W)).view(B, N, -1)
        img_emb = (self.image_agg(feats) if self.use_temporal
                   else feats.mean(dim=1))
        oct_emb = self.oct_encoder(oct_tab)
        return self.fusion(torch.cat([img_emb, oct_emb], dim=1))


# =============================================================================
# IMAGE BRANCH: DATASET
# =============================================================================

class PatientDataset(Dataset):
    def __init__(self, subject_ids, oct_features, labels,
                 folder_map, slice_names, augment=False):
        self.subject_ids = subject_ids
        self.oct_tab     = torch.tensor(oct_features, dtype=torch.float32)
        self.labels      = labels
        self.folder_map  = folder_map
        self.slice_names = slice_names
        base = [transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        aug  = ([transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.ColorJitter(brightness=0.15, contrast=0.15),
                 transforms.RandomRotation(10)] if augment else [])
        self.transform = transforms.Compose(aug + base)

    def _load(self, sid):
        patient_dir = self.folder_map.get(str(sid)[-13:])
        stem_map: Dict[str, Path] = {}
        if patient_dir:
            for p in patient_dir.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(("._",".")):
                    stem_map[p.stem.lower()] = p
        else:
            print(f"[WARN] no folder for '{sid}'")
        frames = []
        for sn in self.slice_names:
            path = stem_map.get(Path(sn).stem.lower())
            if path:
                img = Image.open(path).convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                for c in range(3):
                    ch = arr[..., c]
                    arr[..., c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                img = Image.fromarray((arr*255).clip(0,255).astype(np.uint8))
                frames.append(self.transform(img))
            else:
                frames.append(torch.zeros(3, 224, 224))
        return torch.stack(frames)

    def __len__(self): return len(self.subject_ids)
    def __getitem__(self, idx):
        return (self._load(self.subject_ids[idx]),
                self.oct_tab[idx],
                torch.tensor(self.labels[idx], dtype=torch.long))


# =============================================================================
# IMAGE BRANCH: TRAINING & EVALUATION
# =============================================================================

def train_image_fold(model, y_tr, epochs, lr, patience,
                     best_path, last_path, train_ds, val_ds):
    counts    = np.bincount(y_tr)
    weights   = torch.tensor(1.0/(counts+1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    tr_loader = DataLoader(train_ds, batch_size=len(train_ds),
                           shuffle=True, num_workers=0)
    vl_loader = DataLoader(val_ds, batch_size=len(val_ds),
                           shuffle=False, num_workers=0)
    best_auc = 0; no_imp = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, oct, lbl in tr_loader:
            imgs, oct, lbl = imgs.to(device), oct.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, oct), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vl_preds, vl_probs, vl_lbls = [], [], []
        with torch.no_grad():
            for imgs, oct, lbl in vl_loader:
                imgs, oct, lbl = imgs.to(device), oct.to(device), lbl.to(device)
                p = F.softmax(model(imgs, oct), dim=-1)
                vl_preds.extend(p.argmax(1).cpu().numpy())
                vl_probs.extend(p[:, 1].cpu().numpy())
                vl_lbls.extend(lbl.cpu().numpy())

        vl_acc = np.mean(np.array(vl_preds) == np.array(vl_lbls))
        vl_auc = (roc_auc_score(vl_lbls, vl_probs)
                  if len(set(vl_lbls)) > 1 else 0.0)
        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:03d} | loss={loss.item():.4f} | "
                  f"val_acc={vl_acc:.4f} | val_auc={vl_auc:.4f}")
        if vl_auc > best_auc:
            best_auc = vl_auc; no_imp = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ best val_auc={best_auc:.4f} (epoch {epoch})")
        else:
            no_imp += 1
            torch.save(model.state_dict(), last_path)
            if no_imp >= patience:
                print(f"  [EarlyStop] epoch={epoch} best_auc={best_auc:.4f}")
                break


@torch.no_grad()
def eval_image(model, loader, label_names, subject_ids=None):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for imgs, oct, lbl in loader:
        imgs, oct, lbl = imgs.to(device), oct.to(device), lbl.to(device)
        p = F.softmax(model(imgs, oct), dim=-1)
        all_preds.extend(p.argmax(1).cpu().numpy())
        all_probs.extend(p.cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

    auc = roc_auc_score(all_labels, [p[1] for p in all_probs]) \
          if len(set(all_labels)) > 1 else 0.0
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    lnames = label_names
    sep    = "─" * 90
    print(f"\n{sep}")
    print(f"  {'#':>4}  {'Subject ID':>22}  {'Actual':>10}  {'Predicted':>10}  "
          f"{'P(H)':>8}  {'P(D)':>8}  {'OK':>4}")
    print(sep)
    ids = subject_ids or [f"P{i}" for i in range(len(all_labels))]
    for i, (sid, gt, pred, prb) in enumerate(
            zip(ids, all_labels, all_preds, all_probs)):
        ok = "✓" if int(gt) == int(pred) else "✗"
        print(f"  {i+1:>4}  {str(sid):>22}  "
              f"{lnames.get(int(gt),'?'):>10}  "
              f"{lnames.get(int(pred),'?'):>10}  "
              f"{prb[0]:>8.4f}  {prb[1]:>8.4f}  {ok:>4}")
    n_ok = sum(1 for g, p in zip(all_labels, all_preds) if g == p)
    print(f"\n  Correct={n_ok}/{len(all_labels)}  Acc={acc:.4f}  AUC={auc:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0
        print(f"  Sens={sens:.4f}  Spec={spec:.4f}  "
              f"TP={cm[1,1]} FN={cm[1,0]} FP={cm[0,1]} TN={cm[0,0]}")
    print(sep)
    cm = confusion_matrix(all_labels, all_preds)
    sens = (cm[1,1]/(cm[1,1]+cm[1,0]) if cm.shape==(2,2) and (cm[1,1]+cm[1,0])>0 else 0)
    spec = (cm[0,0]/(cm[0,0]+cm[0,1]) if cm.shape==(2,2) and (cm[0,0]+cm[0,1])>0 else 0)
    return dict(acc=acc, auc=auc, sens=sens, spec=spec,
                preds=all_preds, labels=all_labels)


# =============================================================================
# IMAGE BRANCH: 5-FOLD LOOP  (shared by enface + octa)
# =============================================================================

def run_image_kfold(subject_ids, df_bio, y, label_names,
                    selected, folder_map, slice_names,
                    use_temporal, weights_dir):
    skf = StratifiedKFold(n_splits=CFG["n_splits"], shuffle=True,
                          random_state=SEED)
    fold_results, all_preds, all_labels = [], [], []

    for fold, (tr_idx, ts_idx) in enumerate(
            skf.split(np.arange(len(y)), y), 1):
        print(f"\n{'=' * 70}\n  FOLD {fold}/{CFG['n_splits']}\n{'=' * 70}")
        y_trf = y[tr_idx]
        loc_tr, loc_vl = train_test_split(
            np.arange(len(tr_idx)), test_size=CFG["inner_val_frac"],
            stratify=y_trf, random_state=SEED)
        g_tr = tr_idx[loc_tr]; g_vl = tr_idx[loc_vl]
        y_tr, y_vl, y_ts = y[g_tr], y[g_vl], y[ts_idx]

        log_split_counts("Inner train", y_tr, label_names)
        log_split_counts("Inner val",   y_vl, label_names)
        log_split_counts("Test",        y_ts, label_names)

        ids_tr = [subject_ids[i] for i in g_tr]
        ids_vl = [subject_ids[i] for i in g_vl]
        ids_ts = [subject_ids[i] for i in ts_idx]

        X_tr, _, prep = prepare_biomarker_features(
            df_bio.iloc[g_tr].reset_index(drop=True), y_tr, selected)
        X_vl, _, _ = prepare_biomarker_features(
            df_bio.iloc[g_vl].reset_index(drop=True), y_vl, selected,
            num_pipe=prep["num_pipe"], pca=prep["pca"])
        X_ts, _, _ = prepare_biomarker_features(
            df_bio.iloc[ts_idx].reset_index(drop=True), y_ts, selected,
            num_pipe=prep["num_pipe"], pca=prep["pca"])

        train_ds = PatientDataset(ids_tr, X_tr, y_tr, folder_map,
                                  slice_names, augment=True)
        val_ds   = PatientDataset(ids_vl, X_vl, y_vl, folder_map,
                                  slice_names, augment=False)
        test_ds  = PatientDataset(ids_ts, X_ts, y_ts, folder_map,
                                  slice_names, augment=False)
        test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                                 shuffle=False, num_workers=0)

        model = FusionModel(
            oct_dim=X_tr.shape[1], oct_embed_dim=CFG["oct_embed_dim"],
            dropout=CFG["dropout"], use_temporal=use_temporal,
            max_slices=len(slice_names),
        ).to(device)

        best_p = os.path.join(weights_dir, f"fold{fold}_best.pth")
        last_p = os.path.join(weights_dir, f"fold{fold}_last.pth")
        print(f"\n  Training fold {fold}...")
        train_image_fold(model, y_tr, CFG["epochs"], CFG["lr"],
                         CFG["patience"], best_p, last_p, train_ds, val_ds)

        model.load_state_dict(torch.load(best_p, map_location=device))
        m = eval_image(model, test_loader, label_names, subject_ids=ids_ts)
        print(f"  Fold {fold}: Acc={m['acc']:.4f} AUC={m['auc']:.4f} "
              f"Sens={m['sens']:.4f} Spec={m['spec']:.4f}")
        fold_results.append(dict(fold=fold, **m))
        all_preds.extend(m["preds"])
        all_labels.extend(m["labels"])

    print_aggregate(fold_results, all_labels, all_preds, label_names)


# =============================================================================
# STRUCTURED BRANCH: MLP
# =============================================================================

class BiomarkerMLP(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.3, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.LayerNorm(64),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        n = sum(p.numel() for p in self.parameters())
        print(f"[BiomarkerMLP] input={input_dim} hidden={hidden} params={n:,}")
    def forward(self, x): return self.net(x)


def train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, best_path):
    counts    = np.bincount(y_tr)
    weights   = torch.tensor(1.0/(counts+1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=MLP_CFG["lr"],
                            weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(device)
    yvl_np = y_vl
    best_auc = 0; no_imp = 0

    for epoch in range(1, MLP_CFG["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr), ytr)
        loss.backward(); optimizer.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            p    = F.softmax(model(Xvl), dim=-1).cpu().numpy()
            pred = p.argmax(1)
        auc = roc_auc_score(yvl_np, p[:,1]) if len(set(yvl_np)) > 1 else 0.0
        acc = np.mean(pred == yvl_np)
        if epoch % 20 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:03d} | loss={loss.item():.4f} | "
                  f"val_acc={acc:.4f} | val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc; no_imp = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ best val_auc={best_auc:.4f} (epoch {epoch})")
        else:
            no_imp += 1
            if no_imp >= MLP_CFG["patience"]:
                print(f"  [EarlyStop] epoch={epoch} best_auc={best_auc:.4f}")
                break


def run_structured_kfold(subject_ids, df_bio, y, label_names,
                         selected, weights_dir):
    skf = StratifiedKFold(n_splits=CFG["n_splits"], shuffle=True,
                          random_state=SEED)
    fold_results, all_preds, all_labels = [], [], []

    for fold, (tr_idx, ts_idx) in enumerate(
            skf.split(np.arange(len(y)), y), 1):
        print(f"\n{'=' * 70}\n  FOLD {fold}/{CFG['n_splits']}\n{'=' * 70}")
        y_trf = y[tr_idx]
        loc_tr, loc_vl = train_test_split(
            np.arange(len(tr_idx)), test_size=0.2,
            stratify=y_trf, random_state=SEED)
        g_tr = tr_idx[loc_tr]; g_vl = tr_idx[loc_vl]
        y_tr, y_vl, y_ts = y[g_tr], y[g_vl], y[ts_idx]

        log_split_counts("Inner train", y_tr, label_names)
        log_split_counts("Inner val",   y_vl, label_names)
        log_split_counts("Test",        y_ts, label_names)

        X_tr, _, prep = prepare_biomarker_features(
            df_bio.iloc[g_tr].reset_index(drop=True), y_tr, selected)
        X_vl, _, _ = prepare_biomarker_features(
            df_bio.iloc[g_vl].reset_index(drop=True), y_vl, selected,
            num_pipe=prep["num_pipe"], pca=prep["pca"])
        X_ts, _, _ = prepare_biomarker_features(
            df_bio.iloc[ts_idx].reset_index(drop=True), y_ts, selected,
            num_pipe=prep["num_pipe"], pca=prep["pca"])

        model    = BiomarkerMLP(X_tr.shape[1], MLP_CFG["hidden"],
                                MLP_CFG["dropout"]).to(device)
        best_p   = os.path.join(weights_dir, f"mlp_fold{fold}_best.pth")
        train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl, best_p)

        model.load_state_dict(torch.load(best_p, map_location=device))
        model.eval()
        with torch.no_grad():
            Xts   = torch.tensor(X_ts, dtype=torch.float32).to(device)
            p     = F.softmax(model(Xts), dim=-1).cpu().numpy()
            preds = p.argmax(1)
        auc  = roc_auc_score(y_ts, p[:,1]) if len(set(y_ts)) > 1 else 0.0
        acc  = np.mean(preds == y_ts)
        cm   = confusion_matrix(y_ts, preds)
        sens = (cm[1,1]/(cm[1,1]+cm[1,0]) if cm.shape==(2,2) and (cm[1,1]+cm[1,0])>0 else 0)
        spec = (cm[0,0]/(cm[0,0]+cm[0,1]) if cm.shape==(2,2) and (cm[0,0]+cm[0,1])>0 else 0)
        print(f"\n  Fold {fold}: Acc={acc:.4f} AUC={auc:.4f} "
              f"Sens={sens:.4f} Spec={spec:.4f}")
        print(classification_report(y_ts, preds,
                                    target_names=list(label_names.values()),
                                    zero_division=0))
        fold_results.append(dict(fold=fold, acc=acc, auc=auc,
                                 sens=sens, spec=spec))
        all_preds.extend(preds.tolist())
        all_labels.extend(y_ts.tolist())

    print_aggregate(fold_results, all_labels, all_preds, label_names)


# =============================================================================
# LOAD DATA  (shared)
# =============================================================================

def load_data():
    df          = pd.read_csv(CSV_PATH)
    y_raw       = df.pop(TARGET_COL).values
    subject_ids = df.pop(SUBJECT_ID_COL).astype(str).tolist()
    df_bio      = df.iloc[:, BIOMARKER_START_COL:].copy()
    unique      = sorted(set(y_raw))
    if list(unique) != [0, 1]:
        lmap = {v: i for i, v in enumerate(unique)}
        y    = np.array([lmap[v] for v in y_raw], dtype=np.int64)
        print(f"  Labels remapped: {lmap}")
    else:
        y = y_raw.astype(np.int64)
    label_names = {0: "Healthy", 1: "Disease"}
    print(f"  N={len(y)}  Healthy={int((y==0).sum())}  "
          f"Disease={int((y==1).sum())}")
    return subject_ids, df_bio, y, label_names


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=["enface", "octa", "structured"],
                        help="Which pipeline branch to run")
    parser.add_argument("agg", nargs="?", default="m", choices=["m", "t"],
                        help="Aggregation for enface: m=mean pool, t=transformer")
    return parser.parse_args()


def main():
    args   = parse_args()
    branch = args.branch
    agg    = args.agg

    print("=" * 70)
    print(f"  Branch : {branch}" +
          (f"  [{agg}]" if branch == "enface" else ""))
    print(f"  Device : {device}")
    print("=" * 70)

    subject_ids, df_bio, y, label_names = load_data()
    selected = load_features_by_threshold(
        FEATURE_SCORE_CSV, FEATURE_NAME_COL, FEATURE_SCORE_COL,
        SCORE_THRESHOLD, df_bio.columns.tolist())

    # ------------------------------------------------------------------
    if branch == "enface":
        use_temporal = (agg == "t")
        label        = "enface_transformer" if use_temporal else "enface_meanpool"
        weights_dir  = os.path.join(WEIGHT_BASE, label)
        os.makedirs(weights_dir, exist_ok=True)
        folder_map = build_subject_folder_map(IMAGE_ROOT_ENFACE)
        print(f"  Slices : {len(SLICES_ENFACE)}  |  "
              f"Agg: {'transformer' if use_temporal else 'mean pool'}")
        run_image_kfold(subject_ids, df_bio, y, label_names,
                        selected, folder_map, SLICES_ENFACE,
                        use_temporal, weights_dir)

    # ------------------------------------------------------------------
    elif branch == "octa":
        weights_dir = os.path.join(WEIGHT_BASE, "octa_meanpool_32slices")
        os.makedirs(weights_dir, exist_ok=True)
        folder_map = build_subject_folder_map(IMAGE_ROOT_OCTA)
        print(f"  Slices : {len(SLICES_OCTA)}  |  Agg: mean pool")
        run_image_kfold(subject_ids, df_bio, y, label_names,
                        selected, folder_map, SLICES_OCTA,
                        use_temporal=False, weights_dir=weights_dir)

    # ------------------------------------------------------------------
    elif branch == "structured":
        weights_dir = os.path.join(WEIGHT_BASE, "structured_only")
        os.makedirs(weights_dir, exist_ok=True)
        print(f"  Features: {len(selected)}  |  XGBoost + MLP")

        # XGBoost on PCA features
        X_full, X_pca, _ = prepare_biomarker_features(df_bio, y, selected)
        n_neg = max(int((y==0).sum()), 1)
        n_pos = max(int((y==1).sum()), 1)
        clf   = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", scale_pos_weight=n_neg/n_pos,
            random_state=SEED, verbosity=0)
        cv     = StratifiedKFold(n_splits=CFG["n_splits"], shuffle=True,
                                 random_state=SEED)
        scores = cross_val_score(clf, X_pca, y, cv=cv, scoring="roc_auc")
        print(f"\n[XGBoost] AUC per fold: {scores.round(3)}")
        print(f"[XGBoost] Mean AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

        # MLP on scaled features
        print("\n[MLP] Starting 5-fold CV...")
        run_structured_kfold(subject_ids, df_bio, y, label_names,
                             selected, weights_dir)


if __name__ == "__main__":
    args   = parse_args()
    branch = args.branch
    agg    = args.agg if branch == "enface" else ""
    label  = f"{branch}_{agg}" if agg else branch
    sys.stdout = Tee(os.path.join(LOG_DIR, f"results_{label}.txt"))
    try:
        main()
    finally:
        sys.stdout.close()
