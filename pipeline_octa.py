"""
Pipeline: OCTA images + OCT biomarkers -> ViT-B/16 Fusion -> 5-Fold CV

Run:
    python pipeline_octa.py

32 OCTA slices (slice_1 ... slice_32), mean pooling, separate image folder.
Update IMAGE_ROOT below to point to your OCTA data directory.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model_utils import (
    Tee, SEED, build_subject_folder_map, load_features_by_threshold,
    prepare_biomarker_features, PatientDataset, FusionModel,
    train_fold_model, evaluate, log_split_counts, print_aggregate_results,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Config ----------------------------------------------------------------

CSV_PATH            = "/home/suhel.khan/Dimentia_Project/LKC_Dimentia_structured_data/merged_output_b1_b2_CNvsCI_Refined.csv"
TARGET_COL          = "Dia"
SUBJECT_ID_COL      = "Subj_ID"
IMAGE_ROOT          = "/home/suhel.khan/Dimentia_Project/Image_data/OCTA_images_b1_b2"  # <-- update this
FEATURE_SCORE_CSV   = "/home/suhel.khan/Dimentia_Project/Feature_Engineering/b1andb2_OCT_biomarkers_feature_scores_CN_CI_real.csv"
FEATURE_NAME_COL    = "feature"
FEATURE_SCORE_COL   = "score"
SCORE_THRESHOLD     = 2.0
BIOMARKER_START_COL = 1

SLICE_NAMES = [f"slice_{i}" for i in range(32, 0, -1)]  # slice_32 ... slice_1

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

LOG_DIR     = "/home/suhel.khan/Dimentia_Project/Results/CN_CI_after_B2"
WEIGHT_DIR  = "/home/suhel.khan/Dimentia_Project/Weights/octa_mean_pooling_32slices"

# ---------------------------------------------------------------------------

def run_kfold(subject_ids, df_bio, y, label_names, selected_features, folder_map):
    skf = StratifiedKFold(n_splits=CFG["n_splits"], shuffle=True,
                          random_state=SEED)
    fold_results, all_preds, all_labels = [], [], []

    print("\n" + "=" * 70)
    print(f"  OCTA  |  {CFG['n_splits']}-Fold CV  |  Mean Pooling  |  "
          f"{len(SLICE_NAMES)} slices")
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

        ids_tr = [subject_ids[i] for i in g_tr]
        ids_vl = [subject_ids[i] for i in g_vl]
        ids_ts = [subject_ids[i] for i in ts_idx]

        X_tr, _, prep = prepare_biomarker_features(
            df_bio.iloc[g_tr].reset_index(drop=True), y_tr, selected_features)
        X_vl, _, _   = prepare_biomarker_features(
            df_bio.iloc[g_vl].reset_index(drop=True), y_vl, selected_features,
            num_pipe=prep["num_pipe"], pca=prep["pca"])
        X_ts, _, _   = prepare_biomarker_features(
            df_bio.iloc[ts_idx].reset_index(drop=True), y_ts, selected_features,
            num_pipe=prep["num_pipe"], pca=prep["pca"])

        train_ds = PatientDataset(ids_tr, X_tr, y_tr, folder_map,
                                  SLICE_NAMES, augment=True)
        val_ds   = PatientDataset(ids_vl, X_vl, y_vl, folder_map,
                                  SLICE_NAMES, augment=False)
        test_ds  = PatientDataset(ids_ts, X_ts, y_ts, folder_map,
                                  SLICE_NAMES, augment=False)

        model = FusionModel(
            oct_dim=X_tr.shape[1], oct_embed_dim=CFG["oct_embed_dim"],
            dropout=CFG["dropout"], use_temporal=False,
            max_images=len(SLICE_NAMES),
        ).to(device)

        best_path = os.path.join(WEIGHT_DIR, f"fold{fold}_best.pth")
        last_path = os.path.join(WEIGHT_DIR, f"fold{fold}_last.pth")

        train_fold_model(model, y_tr, CFG["epochs"], CFG["lr"],
                         CFG["patience"], best_path, last_path,
                         train_ds, val_ds, device)

        model.load_state_dict(torch.load(best_path, map_location=device))
        counts  = np.bincount(y_tr)
        weights = torch.tensor(1.0 / (counts + 1e-6),
                               dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                                 shuffle=False, num_workers=0)
        m = evaluate(model, test_loader, device, crit,
                     subject_ids=ids_ts, label_names=label_names,
                     print_per_patient=True)

        cm   = confusion_matrix(m["labels"], m["preds"])
        sens = (cm[1,1]/(cm[1,1]+cm[1,0])
                if cm.shape==(2,2) and (cm[1,1]+cm[1,0])>0 else 0.0)
        spec = (cm[0,0]/(cm[0,0]+cm[0,1])
                if cm.shape==(2,2) and (cm[0,0]+cm[0,1])>0 else 0.0)

        print(f"\n  Fold {fold}: Acc={m['accuracy']:.4f}  "
              f"AUC={m['auc']:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")
        fold_results.append(dict(fold=fold, test_acc=m["accuracy"],
                                 test_auc=m["auc"],
                                 sensitivity=sens, specificity=spec))
        all_preds.extend(m["preds"])
        all_labels.extend(m["labels"])

    print_aggregate_results(fold_results, all_labels, all_preds, label_names)


def main():
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    print(f"[pipeline_octa]  device={device}  slices={len(SLICE_NAMES)}")
    print(f"  IMAGE_ROOT: {IMAGE_ROOT}")

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

    folder_map = build_subject_folder_map(IMAGE_ROOT)

    run_kfold(subject_ids, df_bio, y, label_names, selected, folder_map)


if __name__ == "__main__":
    sys.stdout = Tee(os.path.join(LOG_DIR, "results_octa_mean_pooling_32slices.txt"))
    try:
        main()
    finally:
        sys.stdout.close()
