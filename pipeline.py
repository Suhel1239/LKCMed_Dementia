"""
=============================================================================
Tri-Modal Fusion Pipeline
=============================================================================

  Branch 1 (En face OCT)  : ViT-B/16 -> mean pool over 9 slices  -> (B, 768)
  Branch 2 (OCTA)         : ViT-B/16 -> mean pool over 32 slices -> (B, 768)
  Branch 3 (Structured)   : BiomarkerEncoder                      -> (B, 256)

  Fusion: concat [768 + 768 + 256] -> LayerNorm -> MLP -> 2 classes

Run:
    python pipeline.py
=============================================================================
"""

import os, sys, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image, UnidentifiedImageError

warnings.filterwarnings("ignore")
SEED        = 42
VIT_DIM     = 768
PCA_COMPS   = 15
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# PATHS  -- update before running
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
IMAGE_ROOT_OCTA     = "/home/suhel.khan/Dimentia_Project/Image_data/only_OCT_images_b1_b2"  # <-- update if needed

WEIGHT_DIR          = "/home/suhel.khan/Dimentia_Project/Weights/trimodal_fusion"
LOG_PATH            = "/home/suhel.khan/Dimentia_Project/Results/CN_CI_after_B2/results_trimodal_fusion.txt"

SLICES_ENFACE       = [f"slice_{i}" for i in range(9, 0, -1)]   # 9 slices
SLICES_OCTA         = [f"slice_{i}" for i in range(32, 0, -1)]  # 32 slices

CFG = dict(
    n_splits       = 5,
    inner_val_frac = 0.35,
    epochs         = 200,
    lr             = 3e-4,
    patience       = 10,
    bio_embed_dim  = 256,
    dropout        = 0.4,
    batch_size     = 8,
)


# =============================================================================
# LOGGER
# =============================================================================

class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")
        self._s = sys.__stdout__
    def write(self, m):  self._s.write(m);  self._f.write(m)
    def flush(self):     self._s.flush();   self._f.flush()
    def close(self):     self.flush();      self._f.close();  sys.stdout = self._s


# =============================================================================
# DATA HELPERS
# =============================================================================

def build_folder_map(root: str) -> Dict[str, Path]:
    r = Path(root)
    m = {d.name[-13:]: d for d in r.iterdir() if d.is_dir()}
    print(f"[INFO] {len(m)} patient folders in {r}")
    return m


def load_features_by_threshold(score_csv, name_col, score_col, thresh, cols):
    df      = pd.read_csv(score_csv)
    passing = df[df[score_col] >= thresh][name_col].astype(str).tolist()
    matched = [f for f in passing if f in set(cols)]
    if not matched:
        raise ValueError("No features matched threshold.")
    lookup  = dict(zip(df[name_col].astype(str), df[score_col]))
    matched.sort(key=lambda f: lookup.get(f, 0), reverse=True)
    print(f"[Features] {len(matched)} selected (threshold>={thresh})")
    return matched


def prepare_biomarker(df, selected, num_pipe=None, pca=None):
    df_s = df.reindex(columns=selected).copy()
    for c in df_s.columns:
        df_s[c] = pd.to_numeric(df_s[c], errors="coerce")
    if num_pipe is None:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("scl", StandardScaler())])
        X     = num_pipe.fit_transform(df_s.values)
        n_pca = min(PCA_COMPS, X.shape[1], X.shape[0] - 1)
        pca   = PCA(n_components=n_pca, random_state=SEED)
        pca.fit(X)
    else:
        X = num_pipe.transform(df_s.values)
    return X, dict(num_pipe=num_pipe, pca=pca)


# =============================================================================
# DATASET  -- loads enface slices, OCTA slices, and biomarkers per patient
# =============================================================================

def make_transform(augment=False):
    base = [transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    aug  = ([transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.ColorJitter(brightness=0.15, contrast=0.15),
             transforms.RandomRotation(10)] if augment else [])
    return transforms.Compose(aug + base)


def load_slices(subject_id, folder_map, slice_names, transform):
    patient_dir = folder_map.get(str(subject_id)[-13:])
    stem_map: Dict[str, Path] = {}
    if patient_dir:
        for p in patient_dir.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(("._", ".")):
                stem_map[p.stem.lower()] = p
    else:
        print(f"[WARN] no folder for '{subject_id}'")
    frames = []
    for sn in slice_names:
        path = stem_map.get(Path(sn).stem.lower())
        tensor = torch.zeros(3, 224, 224)
        if path:
            try:
                img = Image.open(path).convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                for c in range(3):
                    ch = arr[..., c]
                    arr[..., c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
                tensor = transform(img)
            except (UnidentifiedImageError, OSError, Exception) as e:
                print(f"[WARN] skipping corrupt slice {path}: {e}")
        frames.append(tensor)
    return torch.stack(frames)   # (N, 3, 224, 224)


class TriModalDataset(Dataset):
    """
    Returns (enface_slices, octa_slices, biomarkers, label) per patient.
    """
    def __init__(self, subject_ids, biomarkers, labels,
                 enface_map, octa_map, augment=False):
        self.subject_ids = subject_ids
        self.bio         = torch.tensor(biomarkers, dtype=torch.float32)
        self.labels      = labels
        self.enface_map  = enface_map
        self.octa_map    = octa_map
        self.tf          = make_transform(augment)

    def __len__(self): return len(self.subject_ids)

    def __getitem__(self, idx):
        sid     = self.subject_ids[idx]
        enface  = load_slices(sid, self.enface_map, SLICES_ENFACE, self.tf)
        octa    = load_slices(sid, self.octa_map,   SLICES_OCTA,   self.tf)
        return (enface, octa, self.bio[idx],
                torch.tensor(self.labels[idx], dtype=torch.long))


# =============================================================================
# MODEL
# =============================================================================

class ViTEncoder(nn.Module):
    """Frozen ViT-B/16. Shared weights instance used independently per modality."""
    def __init__(self):
        super().__init__()
        vit              = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.patch_embed = vit.conv_proj
        self.encoder     = vit.encoder
        self.class_token = vit.class_token
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):             # (B, 3, 224, 224) -> (B, 768)
        B   = x.shape[0]
        x   = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        return self.encoder(torch.cat([cls, x], dim=1))[:, 0]


class TriModalFusionModel(nn.Module):
    """
    Three parallel branches fused into one classifier.

    Branch 1 - En face : ViTEncoder x N_enface slices -> mean pool -> (B, 768)
    Branch 2 - OCTA    : ViTEncoder x N_octa   slices -> mean pool -> (B, 768)
    Branch 3 - Struct  : Linear -> LayerNorm -> ReLU               -> (B, bio_embed)

    Fusion: concat -> LayerNorm -> 512 -> 256 -> 128 -> num_classes
    """
    def __init__(self, bio_dim: int, bio_embed: int = 256,
                 num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        # each modality gets its own ViT instance (both frozen)
        self.enface_vit = ViTEncoder()
        self.octa_vit   = ViTEncoder()

        # structured branch
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_dim, bio_embed),
            nn.LayerNorm(bio_embed),
            nn.ReLU(),
        )

        fused_dim = VIT_DIM + VIT_DIM + bio_embed   # 768 + 768 + 256 = 1792

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),                  nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TriModalFusion] fused_dim={fused_dim}  "
              f"trainable={n:,}  (both ViTs frozen)")

    def forward(self, enface, octa, bio):
        # enface : (B, N_e, 3, H, W)
        # octa   : (B, N_o, 3, H, W)
        # bio    : (B, bio_dim)
        B, Ne, C, H, W = enface.shape
        _, No, *_      = octa.shape

        enface_emb = (self.enface_vit(enface.view(B*Ne, C, H, W))
                          .view(B, Ne, -1).mean(dim=1))          # (B, 768)

        octa_emb   = (self.octa_vit(octa.view(B*No, C, H, W))
                          .view(B, No, -1).mean(dim=1))           # (B, 768)

        bio_emb    = self.bio_encoder(bio)                        # (B, bio_embed)

        fused = torch.cat([enface_emb, octa_emb, bio_emb], dim=1)
        return self.fusion(fused)


# =============================================================================
# TRAINING
# =============================================================================

def train_fold(model, y_tr, train_ds, val_ds, best_path, last_path):
    counts    = np.bincount(y_tr)
    weights   = torch.tensor(1.0 / (counts + 1e-6),
                             dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=CFG["lr"], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    tr_loader = DataLoader(train_ds, batch_size=len(train_ds),
                           shuffle=True,  num_workers=0)
    vl_loader = DataLoader(val_ds,   batch_size=len(val_ds),
                           shuffle=False, num_workers=0)

    best_auc = 0;  no_imp = 0

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        for enface, octa, bio, lbl in tr_loader:
            enface, octa, bio, lbl = (enface.to(device), octa.to(device),
                                      bio.to(device), lbl.to(device))
            optimizer.zero_grad()
            loss = criterion(model(enface, octa, bio), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vl_probs, vl_preds, vl_lbls = [], [], []
        with torch.no_grad():
            for enface, octa, bio, lbl in vl_loader:
                enface, octa, bio, lbl = (enface.to(device), octa.to(device),
                                          bio.to(device), lbl.to(device))
                p = F.softmax(model(enface, octa, bio), dim=-1)
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
            best_auc = vl_auc;  no_imp = 0
            torch.save(model.state_dict(), best_path)
            print(f"    checkmark best val_auc={best_auc:.4f} (epoch {epoch})")
        else:
            no_imp += 1
            torch.save(model.state_dict(), last_path)
            if no_imp >= CFG["patience"]:
                print(f"  [EarlyStop] epoch={epoch}  best_auc={best_auc:.4f}")
                break


@torch.no_grad()
def evaluate(model, loader, label_names, subject_ids=None):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for enface, octa, bio, lbl in loader:
        enface, octa, bio, lbl = (enface.to(device), octa.to(device),
                                  bio.to(device), lbl.to(device))
        p = F.softmax(model(enface, octa, bio), dim=-1)
        all_preds.extend(p.argmax(1).cpu().numpy())
        all_probs.extend(p.cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

    auc = (roc_auc_score(all_labels, [p[1] for p in all_probs])
           if len(set(all_labels)) > 1 else 0.0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    # per-patient table
    lnames = label_names
    sep    = "-" * 90
    print(f"\n{sep}")
    print(f"  {'#':>4}  {'Subject ID':>22}  {'Actual':>10}  {'Predicted':>10}  "
          f"{'P(H)':>8}  {'P(D)':>8}  {'OK':>4}")
    print(sep)
    ids = subject_ids or [f"P{i}" for i in range(len(all_labels))]
    for i, (sid, gt, pred, prb) in enumerate(
            zip(ids, all_labels, all_preds, all_probs)):
        ok = "OK" if int(gt) == int(pred) else "X"
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

    cm   = confusion_matrix(all_labels, all_preds)
    sens = (cm[1,1]/(cm[1,1]+cm[1,0]) if cm.shape==(2,2) and (cm[1,1]+cm[1,0])>0 else 0)
    spec = (cm[0,0]/(cm[0,0]+cm[0,1]) if cm.shape==(2,2) and (cm[0,0]+cm[0,1])>0 else 0)
    return dict(acc=acc, auc=auc, sens=sens, spec=spec,
                preds=all_preds, labels=all_labels)


# =============================================================================
# 5-FOLD CROSS-VALIDATION
# =============================================================================

def run_kfold(subject_ids, df_bio, y, label_names, selected,
              enface_map, octa_map):

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    skf = StratifiedKFold(n_splits=CFG["n_splits"], shuffle=True,
                          random_state=SEED)
    fold_results, all_preds, all_labels = [], [], []

    print("\n" + "=" * 70)
    print(f"  TRI-MODAL FUSION  |  {CFG['n_splits']}-Fold CV")
    print(f"  En face slices : {len(SLICES_ENFACE)}")
    print(f"  OCTA slices    : {len(SLICES_OCTA)}")
    print(f"  Biomarkers     : {len(selected)} features")
    print("=" * 70)

    for fold, (tr_idx, ts_idx) in enumerate(
            skf.split(np.arange(len(y)), y), 1):

        print(f"\n{'=' * 70}\n  FOLD {fold}/{CFG['n_splits']}\n{'=' * 70}")

        y_trf = y[tr_idx]
        loc_tr, loc_vl = train_test_split(
            np.arange(len(tr_idx)), test_size=CFG["inner_val_frac"],
            stratify=y_trf, random_state=SEED)
        g_tr = tr_idx[loc_tr];  g_vl = tr_idx[loc_vl]
        y_tr, y_vl, y_ts = y[g_tr], y[g_vl], y[ts_idx]

        ids_tr = [subject_ids[i] for i in g_tr]
        ids_vl = [subject_ids[i] for i in g_vl]
        ids_ts = [subject_ids[i] for i in ts_idx]

        print(f"  Inner train: {len(y_tr)}  Inner val: {len(y_vl)}  "
              f"Test: {len(y_ts)}")

        # biomarker preprocessing: fit on inner-train only
        X_tr, prep = prepare_biomarker(df_bio.iloc[g_tr].reset_index(drop=True),
                                       selected)
        X_vl, _    = prepare_biomarker(df_bio.iloc[g_vl].reset_index(drop=True),
                                       selected, **prep)
        X_ts, _    = prepare_biomarker(df_bio.iloc[ts_idx].reset_index(drop=True),
                                       selected, **prep)

        train_ds    = TriModalDataset(ids_tr, X_tr, y_tr,
                                      enface_map, octa_map, augment=True)
        val_ds      = TriModalDataset(ids_vl, X_vl, y_vl,
                                      enface_map, octa_map, augment=False)
        test_ds     = TriModalDataset(ids_ts, X_ts, y_ts,
                                      enface_map, octa_map, augment=False)
        test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                                 shuffle=False, num_workers=0)

        model = TriModalFusionModel(
            bio_dim   = X_tr.shape[1],
            bio_embed = CFG["bio_embed_dim"],
            dropout   = CFG["dropout"],
        ).to(device)

        best_p = os.path.join(WEIGHT_DIR, f"fold{fold}_best.pth")
        last_p = os.path.join(WEIGHT_DIR, f"fold{fold}_last.pth")

        print(f"\n  Training fold {fold}...")
        train_fold(model, y_tr, train_ds, val_ds, best_p, last_p)

        model.load_state_dict(torch.load(best_p, map_location=device))
        m = evaluate(model, test_loader, label_names, subject_ids=ids_ts)
        print(f"  Fold {fold}: Acc={m['acc']:.4f}  AUC={m['auc']:.4f}  "
              f"Sens={m['sens']:.4f}  Spec={m['spec']:.4f}")
        fold_results.append(dict(fold=fold, **m))
        all_preds.extend(m["preds"])
        all_labels.extend(m["labels"])

    # aggregate
    print("\n" + "=" * 70 + "\n  AGGREGATE RESULTS\n" + "=" * 70)
    accs  = [r["acc"]  for r in fold_results]
    aucs  = [r["auc"]  for r in fold_results]
    senss = [r["sens"] for r in fold_results]
    specs = [r["spec"] for r in fold_results]
    print(f"\n  {'Fold':>5}  {'Acc':>7}  {'AUC':>7}  {'Sens':>7}  {'Spec':>7}")
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
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Tri-Modal Fusion: En face + OCTA + Structured")
    print(f"  Device: {device}")
    print("=" * 70)

    df          = pd.read_csv(CSV_PATH)
    y_raw       = df.pop(TARGET_COL).values
    subject_ids = df.pop(SUBJECT_ID_COL).astype(str).tolist()
    df_bio      = df.iloc[:, BIOMARKER_START_COL:].copy()

    unique = sorted(set(y_raw))
    if list(unique) != [0, 1]:
        lmap = {v: i for i, v in enumerate(unique)}
        y    = np.array([lmap[v] for v in y_raw], dtype=np.int64)
        print(f"  Labels remapped: {lmap}")
    else:
        y = y_raw.astype(np.int64)

    label_names = {0: "Healthy", 1: "Disease"}
    print(f"  N={len(y)}  Healthy={int((y==0).sum())}  "
          f"Disease={int((y==1).sum())}")

    selected   = load_features_by_threshold(
        FEATURE_SCORE_CSV, FEATURE_NAME_COL, FEATURE_SCORE_COL,
        SCORE_THRESHOLD, df_bio.columns.tolist())

    enface_map = build_folder_map(IMAGE_ROOT_ENFACE)
    octa_map   = build_folder_map(IMAGE_ROOT_OCTA)

    run_kfold(subject_ids, df_bio, y, label_names,
              selected, enface_map, octa_map)


if __name__ == "__main__":
    sys.stdout = Tee(LOG_PATH)
    try:
        main()
    finally:
        sys.stdout.close()
