"""
Pipeline: Structured OCT biomarker data only (no images) -> 5-Fold CV

Models:
  1. XGBoost  (PCA-reduced features, cross-validated)
  2. MLP      (raw scaled features, trained with early stopping)

Run:
    python pipeline_structured.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model_utils import (
    Tee, SEED, load_features_by_threshold,
    prepare_biomarker_features, log_split_counts, PCA_COMPS,
)

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Config ----------------------------------------------------------------

CSV_PATH            = "/home/suhel.khan/Dimentia_Project/LKC_Dimentia_structured_data/merged_output_b1_b2_CNvsCI_Refined.csv"
TARGET_COL          = "Dia"
SUBJECT_ID_COL      = "Subj_ID"
FEATURE_SCORE_CSV   = "/home/suhel.khan/Dimentia_Project/Feature_Engineering/b1andb2_OCT_biomarkers_feature_scores_CN_CI_real.csv"
FEATURE_NAME_COL    = "feature"
FEATURE_SCORE_COL   = "score"
SCORE_THRESHOLD     = 2.0
BIOMARKER_START_COL = 1

CFG = dict(
    n_splits       = 5,
    inner_val_frac = 0.2,
    mlp_epochs     = 300,
    mlp_lr         = 3e-4,
    mlp_patience   = 20,
    mlp_dropout    = 0.3,
    mlp_hidden     = 128,
)

LOG_DIR = "/home/suhel.khan/Dimentia_Project/Results/CN_CI_after_B2"

# ---------------------------------------------------------------------------
# Small MLP for tabular biomarker features
# ---------------------------------------------------------------------------

class BiomarkerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128,
                 dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        n = sum(p.numel() for p in self.parameters())
        print(f"[BiomarkerMLP] input={input_dim}  hidden={hidden}  "
              f"params={n:,}")

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# XGBoost 5-fold CV
# ---------------------------------------------------------------------------

def run_xgboost(X_pca, y, n_splits):
    print(f"\n[XGBoost] PCA({PCA_COMPS}) features  |  {n_splits}-fold CV")
    n_neg = max(int((y == 0).sum()), 1)
    n_pos = max(int((y == 1).sum()), 1)
    clf   = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", scale_pos_weight=n_neg / n_pos,
        random_state=SEED, verbosity=0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X_pca, y, cv=cv, scoring="roc_auc")
    print(f"  AUC per fold : {scores.round(3)}")
    print(f"  Mean AUC     : {scores.mean():.3f} +/- {scores.std():.3f}")
    clf.fit(X_pca, y)
    return clf


# ---------------------------------------------------------------------------
# MLP 5-fold CV
# ---------------------------------------------------------------------------

def train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl,
                   epochs, lr, patience, best_path):
    counts    = np.bincount(y_tr)
    weights   = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xvl = torch.tensor(X_vl, dtype=torch.float32).to(device)
    yvl = torch.tensor(y_vl, dtype=torch.long).to(device)

    best_auc   = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr), ytr)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xvl)
            probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
        labels_np = yvl.cpu().numpy()
        val_auc = (roc_auc_score(labels_np, probs)
                   if len(set(labels_np)) > 1 else 0.0)
        val_acc = np.mean(preds == labels_np)

        if epoch % 20 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:03d} | loss={loss.item():.4f} | "
                  f"val_acc={val_acc:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc   = val_auc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ best val_auc={best_auc:.4f} (epoch {epoch})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [EarlyStop] epoch {epoch}, best_auc={best_auc:.4f}")
                break

    return best_auc


def run_mlp_kfold(subject_ids, df_bio, y, label_names,
                  selected_features, weights_dir):
    skf = StratifiedKFold(n_splits=CFG["n_splits"], shuffle=True,
                          random_state=SEED)
    fold_results, all_preds, all_labels = [], [], []

    print("\n" + "=" * 70)
    print(f"  MLP  |  {CFG['n_splits']}-Fold CV  |  Structured biomarkers only")
    print("=" * 70)

    for fold, (tr_idx, ts_idx) in enumerate(
            skf.split(np.arange(len(y)), y), 1):

        print(f"\n{'=' * 70}\n  FOLD {fold}/{CFG['n_splits']}\n{'=' * 70}")

        y_tr_fold = y[tr_idx]
        loc_tr, loc_vl = train_test_split(
            np.arange(len(tr_idx)),
            test_size=CFG["inner_val_frac"],
            stratify=y_tr_fold, random_state=SEED)

        g_tr = tr_idx[loc_tr]
        g_vl = tr_idx[loc_vl]
        y_tr, y_vl, y_ts = y[g_tr], y[g_vl], y[ts_idx]

        log_split_counts("Inner train", y_tr, label_names)
        log_split_counts("Inner val",   y_vl, label_names)
        log_split_counts("Test",        y_ts, label_names)

        X_tr, _, prep = prepare_biomarker_features(
            df_bio.iloc[g_tr].reset_index(drop=True), y_tr, selected_features)
        X_vl, _, _   = prepare_biomarker_features(
            df_bio.iloc[g_vl].reset_index(drop=True), y_vl, selected_features,
            num_pipe=prep["num_pipe"], pca=prep["pca"])
        X_ts, _, _   = prepare_biomarker_features(
            df_bio.iloc[ts_idx].reset_index(drop=True), y_ts, selected_features,
            num_pipe=prep["num_pipe"], pca=prep["pca"])

        model = BiomarkerMLP(
            input_dim = X_tr.shape[1],
            hidden    = CFG["mlp_hidden"],
            dropout   = CFG["mlp_dropout"],
        ).to(device)

        best_path = os.path.join(weights_dir, f"mlp_fold{fold}_best.pth")

        train_mlp_fold(model, X_tr, y_tr, X_vl, y_vl,
                       CFG["mlp_epochs"], CFG["mlp_lr"],
                       CFG["mlp_patience"], best_path)

        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        with torch.no_grad():
            Xts    = torch.tensor(X_ts, dtype=torch.float32).to(device)
            logits = model(Xts)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()

        auc  = (roc_auc_score(y_ts, probs[:, 1])
                if len(set(y_ts)) > 1 else 0.0)
        acc  = np.mean(preds == y_ts)
        cm   = confusion_matrix(y_ts, preds)
        sens = (cm[1,1]/(cm[1,1]+cm[1,0])
                if cm.shape==(2,2) and (cm[1,1]+cm[1,0])>0 else 0.0)
        spec = (cm[0,0]/(cm[0,0]+cm[0,1])
                if cm.shape==(2,2) and (cm[0,0]+cm[0,1])>0 else 0.0)

        print(f"\n  Fold {fold}: Acc={acc:.4f}  AUC={auc:.4f}  "
              f"Sens={sens:.4f}  Spec={spec:.4f}")
        print(classification_report(y_ts, preds,
                                    target_names=list(label_names.values()),
                                    zero_division=0))
        fold_results.append(dict(fold=fold, test_acc=acc, test_auc=auc,
                                 sensitivity=sens, specificity=spec))
        all_preds.extend(preds.tolist())
        all_labels.extend(y_ts.tolist())

    from model_utils import print_aggregate_results
    print_aggregate_results(fold_results, all_labels, all_preds, label_names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    weights_dir = "/home/suhel.khan/Dimentia_Project/Weights/structured_only"
    os.makedirs(weights_dir, exist_ok=True)
    print(f"[pipeline_structured]  device={device}")

    df          = pd.read_csv(CSV_PATH)
    y_raw       = df.pop(TARGET_COL).values
    subject_ids = df.pop(SUBJECT_ID_COL).astype(str).tolist()
    df_bio      = df.iloc[:, BIOMARKER_START_COL:].copy()

    unique = sorted(set(y_raw))
    if list(unique) != [0, 1]:
        lmap = {v: i for i, v in enumerate(unique)}
        y    = np.array([lmap[v] for v in y_raw], dtype=np.int64)
    else:
        y = y_raw.astype(np.int64)

    label_names = {0: "Healthy", 1: "Disease"}
    print(f"  N={len(y)}  Healthy={int((y==0).sum())}  Disease={int((y==1).sum())}")

    selected = load_features_by_threshold(
        FEATURE_SCORE_CSV, FEATURE_NAME_COL, FEATURE_SCORE_COL,
        SCORE_THRESHOLD, df_bio.columns.tolist())
    print(f"  Features selected: {len(selected)}")

    # Fit preprocessing on full data for XGBoost CV (sklearn handles splits)
    X_full, X_pca, _ = prepare_biomarker_features(
        df_bio, y, selected)

    # 1. XGBoost
    run_xgboost(X_pca, y, CFG["n_splits"])

    # 2. MLP
    run_mlp_kfold(subject_ids, df_bio, y, label_names, selected, weights_dir)


if __name__ == "__main__":
    sys.stdout = Tee(os.path.join(LOG_DIR, "results_structured_only.txt"))
    try:
        main()
    finally:
        sys.stdout.close()
