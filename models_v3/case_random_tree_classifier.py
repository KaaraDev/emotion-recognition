# train_rf_from_case_features.py
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE
from imblearn.base import BaseSampler
from sklearn.base import clone
from numbers import Integral
from sklearn.utils._param_validation import Interval

import joblib


# ------------------- Konfiguration -------------------

@dataclass
class TrainCfg:
    csv_path: Path
    out_dir: Path
    random_state: int = 42
    log_level: str = "INFO"

    # Pausen-Videos raus?
    exclude_pauses: bool = True
    # SMOTE Variante
    smote_variant: str = "svm"  # "svm" oder "borderline"
    # QuantileTransformer aus?
    no_quantile: bool = False
    # class_weight im RF aus?
    no_class_weight: bool = False
    # GroupKFold folds
    n_folds: int = 5
    # bored (3/4) etwas runterstutzen
    downsample_bored: bool = True
    bored_target: str = "median"  # "min" oder "median"
    # optional: auf bestimmte Klassen beschränken (z.B. ["scary", "amused"])
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


# ------------------- Adaptive SMOTE -------------------

class AdaptiveSMOTE(BaseSampler):
    """
    SMOTE/BorderlineSMOTE/SVMSMOTE mit adaptivem k und sicheren Fallbacks:
    - Wenn im Train-Fold < 2 Klassen vorhanden: no-op.
    - Wenn kleinste Klasse <= 1: no-op.
    - k_neighbors wird an die kleinste Klasse angepasst.
    - Wenn der interne Sampler trotzdem wirft: no-op.
    """
    _parameter_constraints = {
        "base_smote": [object],
        "min_k": [Interval(Integral, 1, None, closed="left")],
        "sampling_strategy": [object],
    }
    _sampling_type = "over-sampling"

    def __init__(self, base_smote, min_k=3, sampling_strategy="auto"):
        self.base_smote = base_smote
        self.min_k = min_k
        self.sampling_strategy = sampling_strategy
        self._effective_smote_ = None
        self._disabled_ = False

    def _fit_resample(self, X, y):
        import numpy as np
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        if classes.size < 2:
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y

        min_count = int(counts.min())
        if min_count <= 1:
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y

        max_k = max(1, min_count - 1)
        k = max(1, min(self.min_k, max_k))

        smote = clone(self.base_smote)
        set_params = {
            "k_neighbors": k,
            "sampling_strategy": self.sampling_strategy,
        }
        if hasattr(smote, "m_neighbors"):
            set_params["m_neighbors"] = min(getattr(smote, "m_neighbors", 10), k, max_k)
        if hasattr(smote, "n_neighbors"):
            set_params["n_neighbors"] = min(getattr(smote, "n_neighbors", 5), k, max_k)

        smote.set_params(**set_params)

        try:
            X_res, y_res = smote.fit_resample(X, y)
            self._effective_smote_ = smote
            self._disabled_ = False
            return X_res, y_res
        except Exception as e:
            import logging
            logging.getLogger("rf_case").warning(
                "AdaptiveSMOTE: fallback to no-op (k=%d, classes=%s, counts=%s): %r",
                k, classes.tolist(), counts.tolist(), e
            )
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y


def build_pipeline(cfg: TrainCfg) -> ImbPipeline:
    if cfg.smote_variant.lower() == "borderline":
        base = BorderlineSMOTE(random_state=cfg.random_state, k_neighbors=3)
    else:
        base = SVMSMOTE(random_state=cfg.random_state, k_neighbors=3)
    smote = AdaptiveSMOTE(base_smote=base, min_k=3)

    steps = [("imp", SimpleImputer(strategy="median"))]
    if not cfg.no_quantile:
        steps.append(("qt", QuantileTransformer(
            n_quantiles=300,
            output_distribution="normal",
            subsample=200000,
            random_state=cfg.random_state
        )))
    steps.append(("smote", smote))
    steps.append(("rf", RandomForestClassifier(
        n_estimators=800,
        max_depth=40,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=cfg.random_state,
        class_weight=None if cfg.no_class_weight else "balanced_subsample"
    )))
    return ImbPipeline(steps=steps)


# ------------------- Daten laden -------------------

# Meta-Spalten in DEINER Feature-CSV
META_COLS = {"subject", "window_start_ms", "window_end_ms", "video_id"}
LABEL_COLS = {"arousal", "valence"}  # kontinuierlich, hier nicht für Klassifikation genutzt


def load_features(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Lade Feature-CSV aus %s ...", path)
    if path.suffix.endswith("gz") or path.suffix.endswith("csv"):
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unbekanntes Format: {path}")
    logger.info("Gelesen: %d Zeilen, %d Spalten", len(df), df.shape[1])
    return df


def attach_labels(df: pd.DataFrame, cfg: TrainCfg, logger: logging.Logger) -> pd.DataFrame:
    if "video_id" not in df.columns:
        raise ValueError("Spalte 'video_id' wird für das Mapping benötigt (kommt aus feature_extraction.py).")
    if "subject" not in df.columns:
        raise ValueError("Spalte 'subject' fehlt – wird für GroupKFold gebraucht.")

    df = df.copy()

    # Pausen rauswerfen
    if cfg.exclude_pauses:
        before = len(df)
        df = df.loc[~df["video_id"].isin(EXCLUDE_VIDEO_IDS)]
        logger.info("Pausen (10/11/12) entfernt: %d -> %d", before, len(df))

    # Video-ID (numerisch) -> Label
    df["label"] = df["video_id"].map(VIDEO_TO_LABEL)
    before_map = len(df)
    df = df[df["label"].notna()].copy()
    logger.info("Video->Label Mapping angewendet. Unmapped gedroppt: %d -> %d", before_map, len(df))

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
    out = pd.concat([df_major, df_minor], axis=0).sample(
        frac=1.0, random_state=cfg.random_state
    ).reset_index(drop=True)
    logger.info("Nach Downsampling: %s", out["label"].value_counts().to_dict())
    return out


# ------------------- CV & Training -------------------

def run_group_cv(df: pd.DataFrame, cfg: TrainCfg, logger: logging.Logger) -> Dict[str, Any]:
    # Feature-Spalten = alles, was nicht meta + nicht label
    drop_cols = META_COLS | LABEL_COLS | {"label"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    logger.info("Verwende %d Feature-Spalten.", len(feat_cols))

    X = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    groups = df["subject"].to_numpy()

    gkf = GroupKFold(n_splits=cfg.n_folds)

    all_true = []
    all_pred = []
    fold_rows = []

    t0_all = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        t0 = time.perf_counter()
        pipe = build_pipeline(cfg)
        pipe.fit(X[tr_idx], y[tr_idx])
        y_hat = pipe.predict(X[va_idx])

        acc = accuracy_score(y[va_idx], y_hat)
        bacc = balanced_accuracy_score(y[va_idx], y_hat)
        f1m = f1_score(y[va_idx], y_hat, average="macro")

        logger.info("Fold %d/%d: acc=%.4f | bAcc=%.4f | f1_macro=%.4f",
                    fold, cfg.n_folds, acc, bacc, f1m)

        cm_labels = sorted(np.unique(np.concatenate([y[va_idx], y_hat])))
        cm = confusion_matrix(y[va_idx], y_hat, labels=cm_labels)
        cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
        cm_df.to_csv(cfg.out_dir / f"cm_fold{fold}.csv", index=True)

        with open(cfg.out_dir / f"report_fold{fold}.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y[va_idx], y_hat, digits=3))

        fold_rows.append({
            "fold": fold,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1_macro": f1m
        })

        all_true.append(y[va_idx])
        all_pred.append(y_hat)

        elapsed = time.perf_counter() - t0
        logger.info("Fold %d done in %.1fs", fold, elapsed)

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

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(cfg.out_dir / "metrics_per_fold.csv", index=False)

    summary = {
        "pooled_accuracy": float(acc_all),
        "pooled_balanced_accuracy": float(bacc_all),
        "pooled_f1_macro": float(f1m_all),
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "f1_macro_mean": float(fold_df["f1_macro"].mean()),
        "f1_macro_std": float(fold_df["f1_macro"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "feature_cols": feat_cols,
    }
    with open(cfg.out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("CV fertig in %.1fs", time.perf_counter() - t0_all)
    return summary


def refit_final(df: pd.DataFrame, cfg: TrainCfg, logger: logging.Logger, feat_cols: List[str]):
    # X und y vorbereiten
    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df["label"].to_numpy()

    pipe = build_pipeline(cfg)
    pipe.fit(X_all, y_all)

    rf = pipe.named_steps["rf"]

    importances = rf.feature_importances_
    n_feat = len(feat_cols)
    n_imp = len(importances)

    if n_feat != n_imp:
        logger.warning(
            "Längen-Mismatch bei Feature Importances: len(feat_cols)=%d, len(importances)=%d. "
            "Schneide auf das Minimum zu.",
            n_feat, n_imp
        )
        n = min(n_feat, n_imp)
        feat_cols_use = feat_cols[:n]
        importances_use = importances[:n]
    else:
        feat_cols_use = feat_cols
        importances_use = importances

    imp_df = pd.DataFrame({
        "feature": feat_cols_use,
        "importance": importances_use
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(cfg.out_dir / "feature_importance.csv", index=False)
    joblib.dump(pipe, cfg.out_dir / "final_model_pipeline.joblib")
    logger.info("Finales Modell gespeichert nach %s", cfg.out_dir / "final_model_pipeline.joblib")


# ------------------- main -------------------

if __name__ == "__main__":
    # HIER deinen Pfad zur Feature-CSV aus feature_extraction.py eintragen:
    base_csv = Path("outputs/features_case_bvp_gsr_skt.csv")

    # Beispiel: mehrere 2-Klassen-Experimente wie vorher
    experiments = [
        ("scary_vs_amused", ["scary", "amused"]),
        ("bored_vs_relaxed", ["bored", "relaxed"]),
        ("scary_vs_bored", ["scary", "bored"]),
        ("amused_vs_bored", ["amused", "bored"]),
        # Oder alle 4 Klassen:
        ("all_four", ["scary", "amused", "bored", "relaxed"]),
    ]

    for exp_name, classes in experiments:
        print(f"\n=== Starte Experiment: {exp_name} ({classes}) ===")

        cfg = TrainCfg(
            csv_path=base_csv,
            out_dir=Path(f"rf_outputs/{exp_name}"),
            random_state=42,
            classes_to_keep=classes,
        )

        np.random.seed(cfg.random_state)
        random.seed(cfg.random_state)

        logger = setup_logger(cfg.out_dir, cfg.log_level)

        df = load_features(cfg.csv_path, logger)
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

        summary = run_group_cv(df, cfg, logger)
        feat_cols = summary["feature_cols"]
        refit_final(df, cfg, logger, feat_cols)

        with open(cfg.out_dir / "used_features.json", "w", encoding="utf-8") as f:
            json.dump(feat_cols, f, indent=2)

        logger.info("Experiment %s fertig.", exp_name)
