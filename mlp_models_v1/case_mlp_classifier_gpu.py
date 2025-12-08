# train_mlp_whitelist_hyperopt_gpu.py
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

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# --- PyTorch + skorch für GPU-MLP ---
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

# ------------------- Feature-Whitelist -------------------
# Hier deine finalen Features eintragen (alle müssen Spaltennamen im CSV sein).
FEATURE_WHITELIST: List[str] | None = [
    "gsr_mean",
    "gsr_std",
    "gsr_min",
    "gsr_max",
    "gsr_slope_per_s",
    "eda_SCR_Peaks_N",
    "eda_SCR_Peaks_Amplitude_Mean",
    "eda_EDA_Tonic_SD",
    "eda_EDA_Autocorrelation",
    "eda_tonic_mean",
    "eda_phasic_mean",
    "gsr_diff_mean",
    "gsr_diff_std",
    "gsr_pos_diff_ratio",
]


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

    # Hyperparameter-Suche
    use_hyperopt: bool = True
    n_iter_search: int = 30  # Anzahl Zufalls-Kombinationen in RandomizedSearchCV


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
    logger = logging.getLogger("mlp_case_gpu")
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
    df
