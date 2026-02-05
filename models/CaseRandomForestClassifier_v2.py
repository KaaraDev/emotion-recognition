# train_rf_from_case_combined_plain_oof_preds.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json
import logging
import sys
import time
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import joblib


# ------------------- Konfiguration -------------------

@dataclass
class TrainCfg:
    csv_path: Path
    out_dir: Path
    random_state: int = 42
    log_level: str = "INFO"

    # welche Videos raus?
    exclude_pauses: bool = True
    # GroupKFold folds
    n_folds: int = 5
    # bored (3/4) etwas runterstutzen
    downsample_bored: bool = False
    bored_target: str = "median"  # "min" oder "median"
    classes_to_keep: List[str] | None = None


# Mapping wie in deinem alten Skript
VIDEO_TO_LABEL = {
    1: "amused",
    2: "amused",
    3: "bored",
    4: "bored",
    5: "relaxed",
    6: "relaxed",
    7: "scary",
    8: "scary",
}
EXCLUDE_VIDEO_IDS = {10, 11, 12}
EMOTIONS = ["amused", "bored", "relaxed", "scary"]


# ------------------- Logging -------------------

def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rf_case")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(fh)

    logger.info("Logger ready.")
    return logger


# ------------------- Daten laden -------------------

META_COLS = {"subject", "start_s", "end_s", "video"}
LABEL_COLS = {"label_valence", "label_arousal"}


def load_case_combined(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Lade Features aus %s ...", path)
    if path.suffix.endswith("gz") or path.suffix.endswith("csv"):
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unbekanntes Format: {path}")
    logger.info("Gelesen: %d Zeilen, %d Spalten", len(df), df.shape[1])
    return df


def attach_labels(df: pd.DataFrame, cfg: TrainCfg, logger: logging.Logger) -> pd.DataFrame:
    if "video" not in df.columns:
        raise ValueError("Spalte 'video' wird für das Mapping benötigt (steht in deinem combined drin).")
    df = df.copy()

    # Pausen raus
    if cfg.exclude_pauses:
        before = len(df)
        df = df.loc[~df["video"].isin(EXCLUDE_VIDEO_IDS)]
        logger.info("Pausen (10/11/12) entfernt: %d -> %d", before, len(df))

    # Video -> Label
    df["label"] = df["video"].map(VIDEO_TO_LABEL)
    before_map = len(df)
    df = df[df["label"].notna()].copy()
    logger.info("Video->Label Mapping angewendet. Unmapped gedroppt: %d -> %d", before_map, len(df))

    # sanity
    if "subject" not in df.columns:
        raise ValueError("Spalte 'subject' fehlt – wird für GroupKFold gebraucht.")
    return df


def downsample_class(df: pd.DataFrame, cls: str, cfg: TrainCfg, logger: logging.Logger) -> pd.DataFrame:
    counts = df["label"].value_counts().to_dict()
    logger.info("Vor Downsampling: %s", counts)
    if cls not in counts:
        return df
    other_counts = [v for k, v in counts.items() if k != cls]
    if not other_counts:
        return df

    if cfg.bored_target == "min":
        target_n = int(min(other_counts))
    else:
        target_n = int(np.median(other_counts))

    cur_n = counts[cls]
    if cur_n <= target_n:
        logger.info("Klasse %s bereits <= target (%d <= %d)", cls, cur_n, target_n)
        return df

    df_major = df[df["label"] != cls]
    df_minor = df[df["label"] == cls].sample(n=target_n, random_state=cfg.random_state)
    out = pd.concat([df_major, df_minor], axis=0).sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
    logger.info("Nach Downsampling: %s", out["label"].value_counts().to_dict())
    return out


# ------------------- RF-Builder -------------------

def build_rf(cfg: TrainCfg) -> RandomForestClassifier:
    """Plain RF mit festen Hyperparametern."""
    return RandomForestClassifier(
        n_estimators=800,
        max_depth=40,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=cfg.random_state,
        class_weight="balanced_subsample",
    )


# ------------------- CV & Training -------------------

def run_group_cv(df: pd.DataFrame, cfg: TrainCfg, logger: logging.Logger) -> Dict[str, Any]:
    # Feature-Spalten = alles, was nicht meta + nicht label
    drop_cols = META_COLS | LABEL_COLS | {"label"}
    candidate_feats = [c for c in df.columns if c not in drop_cols]

    # komplett leere Spalten rausfiltern
    non_empty_feats = []
    empty_feats = []
    for c in candidate_feats:
        col = df[c]
        if col.notna().any():
            non_empty_feats.append(c)
        else:
            empty_feats.append(c)

    if empty_feats:
        logger.info(
            "Ignoriere %d komplett leere Feature-Spalten: %s",
            len(empty_feats),
            empty_feats,
        )

    feat_cols = non_empty_feats
    logger.info("Verwende %d Feature-Spalten (ohne komplett leere).", len(feat_cols))

    X = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    groups = df["subject"].to_numpy()

    gkf = GroupKFold(n_splits=cfg.n_folds)

    all_true = []
    all_pred = []
    fold_rows = []
    pred_rows = []  # out-of-fold Predictions für spätere Plots

    best_fold = None
    best_f1m = -np.inf

    t0_all = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        t0 = time.perf_counter()

        # --- Subjekte für Train/Val dokumentieren ---
        train_subjects = [int(s) for s in sorted(np.unique(groups[tr_idx]))]
        val_subjects = [int(s) for s in sorted(np.unique(groups[va_idx]))]
        logger.info("Fold %d Train-Subjects: %s", fold, train_subjects)
        logger.info("Fold %d Val-Subjects:   %s", fold, val_subjects)

        # Plain Random Forest für diesen Fold
        rf = build_rf(cfg)
        rf.fit(X[tr_idx], y[tr_idx])

        # Vorhersage auf Val-Set
        y_hat = rf.predict(X[va_idx])
        if hasattr(rf, "predict_proba"):
            y_proba = rf.predict_proba(X[va_idx])
        else:
            y_proba = None

        # Metriken
        acc = accuracy_score(y[va_idx], y_hat)
        bacc = balanced_accuracy_score(y[va_idx], y_hat)
        f1m = f1_score(y[va_idx], y_hat, average="macro")

        logger.info(
            "Fold %d/%d: acc=%.4f | bAcc=%.4f | f1_macro=%.4f",
            fold, cfg.n_folds, acc, bacc, f1m
        )

        # Confusion Matrix & Report pro Fold
        cm_labels = sorted(np.unique(np.concatenate([y[va_idx], y_hat])))
        cm = confusion_matrix(y[va_idx], y_hat, labels=cm_labels)
        cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
        cm_df.to_csv(cfg.out_dir / f"cm_fold{fold}.csv", index=True)

        with open(cfg.out_dir / f"report_fold{fold}.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y[va_idx], y_hat, digits=3))

        # --- out-of-fold Predictions speichern ---
        for i, idx in enumerate(va_idx):
            row = {
                "fold": fold,
                "subject": df.iloc[idx]["subject"],
                "video": df.iloc[idx]["video"],
                "start_s": df.iloc[idx].get("start_s", np.nan),
                "end_s": df.iloc[idx].get("end_s", np.nan),
                "true_label": y[va_idx][i],
                "pred_label": y_hat[i],
            }
            if y_proba is not None:
                for class_idx, cls_name in enumerate(rf.classes_):
                    row[f"proba_{cls_name}"] = float(y_proba[i, class_idx])
            pred_rows.append(row)
        # -----------------------------------------

        # Zeile für Gesamt-CSV
        fold_rows.append({
            "fold": fold,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1_macro": f1m,
            "n_train_samples": int(len(tr_idx)),
            "n_val_samples": int(len(va_idx)),
            "train_subjects": ",".join(map(str, train_subjects)),
            "val_subjects": ",".join(map(str, val_subjects)),
        })

        all_true.append(y[va_idx])
        all_pred.append(y_hat)

        # Bestes Fold tracken
        if f1m > best_f1m:
            best_f1m = f1m
            best_fold = fold

        elapsed = time.perf_counter() - t0
        logger.info("Fold %d done in %.1fs", fold, elapsed)

    # Pooled Ergebnisse
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    acc_all = accuracy_score(all_true, all_pred)
    bacc_all = balanced_accuracy_score(all_true, all_pred)
    f1m_all = f1_score(all_true, all_pred, average="macro")

    logger.info("== Pooled results over all folds ==")
    logger.info("acc=%.4f | bAcc=%.4f | f1_macro=%.4f", acc_all, bacc_all, f1m_all)

    labels_sorted = sorted(np.unique(np.concatenate([all_true, all_pred])))
    cm_all = confusion_matrix(all_true, all_pred, labels=labels_sorted)
    pd.DataFrame(cm_all, index=labels_sorted, columns=labels_sorted) \
        .to_csv(cfg.out_dir / "confusion_matrix_pooled.csv", index=True)

    with open(cfg.out_dir / "classification_report_pooled.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(all_true, all_pred, digits=3))

    # --- Zusammenfassungs-CSV mit allem drin ---
    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) > 0:
        best_idx = fold_df["f1_macro"].idxmax()
        fold_df["is_best"] = False
        fold_df.loc[best_idx, "is_best"] = True
    else:
        fold_df["is_best"] = []

    fold_df.to_csv(cfg.out_dir / "cv_folds_summary.csv", index=False)
    logger.info("CV-Zusammenfassung gespeichert nach %s", cfg.out_dir / "cv_folds_summary.csv")
    # --------------------------------------------

    # --- out-of-fold Predictions speichern ---
    pred_df = pd.DataFrame(pred_rows)
    pred_path = cfg.out_dir / "cv_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("CV-Predictions gespeichert nach %s", pred_path)
    # -----------------------------------------

    summary = {
        "pooled_accuracy": float(acc_all),
        "pooled_balanced_accuracy": float(bacc_all),
        "pooled_f1_macro": float(f1m_all),
        "feature_cols": feat_cols,
        "best_fold": int(best_fold) if best_fold is not None else None,
        "best_fold_f1_macro": float(best_f1m) if best_fold is not None else None,
    }
    with open(cfg.out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("CV fertig in %.1fs", time.perf_counter() - t0_all)
    if best_fold is not None:
        logger.info("Bester Fold: %d mit f1_macro=%.4f", best_fold, best_f1m)

    return summary


# ------------------- main -------------------

if __name__ == "__main__":
    base_csv = Path("features_case_vA_20w10s/combined.parquet")

    experiments = [
        ("scary_vs_amused", ["scary", "amused"]),
        ("bored_vs_relaxed", ["bored", "relaxed"]),
        ("scary_vs_bored", ["scary", "bored"]),
        ("amused_vs_bored", ["amused", "bored"]),
    ]

    for exp_name, classes in experiments:
        print(f"\n=== Starte Experiment: {exp_name} ({classes}) ===")

        cfg = TrainCfg(
            csv_path=base_csv,
            out_dir=Path(f"outputs_20w10s_plain_vA/{exp_name}"),
            random_state=42,
            classes_to_keep=classes,
        )

        np.random.seed(cfg.random_state)
        random.seed(cfg.random_state)

        logger = setup_logger(cfg.out_dir, cfg.log_level)

        df = load_case_combined(cfg.csv_path, logger)
        df = attach_labels(df, cfg, logger)

        # Nur die Klassen für dieses Experiment behalten
        if cfg.classes_to_keep is not None:
            before = len(df)
            df = df[df["label"].isin(cfg.classes_to_keep)].copy()
            logger.info(
                "Filter auf Klassen %s: %d -> %d Zeilen",
                cfg.classes_to_keep, before, len(df)
            )

        # Downsampling bored nur, wenn bored überhaupt vorkommt
        if cfg.downsample_bored and "bored" in (cfg.classes_to_keep or EMOTIONS):
            if "bored" in df["label"].unique():
                df = downsample_class(df, "bored", cfg, logger)

        # GroupKFold-CV nur zur Performance-Schätzung
        summary = run_group_cv(df, cfg, logger)
        feat_cols = summary.get("feature_cols", [])

        # Finales Modell auf ALLEN Daten trainieren (so macht man es typisch)
        if len(feat_cols) > 0:
            X_all = df[feat_cols].to_numpy(dtype=float)
            y_all = df["label"].to_numpy()

            rf_final = build_rf(cfg)
            rf_final.fit(X_all, y_all)

            final_model_path = cfg.out_dir / "final_model.joblib"
            joblib.dump(rf_final, final_model_path)
            logger.info("Finales Modell gespeichert nach %s", final_model_path)

            # verwendete Features speichern
            with open(cfg.out_dir / "used_features.json", "w", encoding="utf-8") as f:
                json.dump(feat_cols, f, indent=2)

        logger.info("Experiment %s fertig.", exp_name)
