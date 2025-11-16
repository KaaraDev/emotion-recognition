# train_cnn_raw_case.py
#
# 1D-CNN auf NeuroKit2-vorverarbeiteten physiologischen DAQ-Signalen.
# Labels/Segmente kommen aus den interpolierten CSVs (mit video-Spalte).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import json
import logging
import sys
import time
import random
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import neurokit2 as nk  # NeuroKit2


# ---------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------


@dataclass
class TrainCfg:
    raw_dir: Path  # z.B. CASE_dataset/data/raw/physiological
    interp_dir: Path  # z.B. CASE_dataset/data/interpolated/physiological
    out_dir: Path

    random_state: int = 42
    log_level: str = "INFO"

    # Fenster
    window_sec: float = 8.0
    step_sec: float = 2.0

    # Ziel-Samplingrate NACH NeuroKit (fürs CNN)
    target_fs: float = 50.0  # 50 Hz

    # Pausen-Videos raus?
    exclude_pauses: bool = True

    # GroupKFold folds
    n_folds: int = 5

    # Experiment: nur bestimmte Klassen
    classes_to_keep: List[str] | None = None

    # CNN Hyperparameter
    n_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    conv_channels1: int = 32
    conv_channels2: int = 64
    conv_channels3: int = 128
    linear_hidden: int = 64
    dropout: float = 0.4
    early_stopping_patience: int = 7


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

# Rohspalten im DAQ-File
RAW_COLS = [
    "daqtime",
    "ecg",
    "bvp",
    "gsr",
    "rsp",
    "skt",
    "emg_zygo",
    "emg_coru",
    "emg_trap",
]

# Feature-Kanäle, die ins CNN gehen (alle aus NeuroKit oder leicht abgeleitet)
NK_FEATURE_COLS = [
    "ecg_clean",
    "ecg_rate",
    "ppg_clean",
    "ppg_rate",
    "eda_tonic",
    "eda_phasic",
    "rsp_clean",
    "rsp_amplitude",
    "skt",
    "emg_zygo_envelope",
    "emg_coru_envelope",
    "emg_trap_envelope",
]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cnn_raw_case")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

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


# ---------------------------------------------------------------------
# Hilfsfunktionen: Sampling, Segmente
# ---------------------------------------------------------------------


def estimate_fs(t_sec: np.ndarray) -> float:
    """Schätze Samplingrate aus Zeitvektor (in Sekunden)."""
    if len(t_sec) < 2:
        return 1000.0
    dt = np.diff(t_sec)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 1000.0
    median_dt = np.median(dt)
    if median_dt <= 0:
        return 1000.0
    return 1.0 / median_dt


def compute_downsample_factor(daqtime_sec: np.ndarray, target_fs: float) -> int:
    orig_fs = estimate_fs(daqtime_sec)
    factor = int(round(orig_fs / target_fs))
    if factor < 1:
        factor = 1
    return factor


def parse_subject_id_from_raw_name(path: Path) -> int | None:
    # erwartet z.B. sub1_DAQ.txt, sub10_DAQ.txt
    m = re.search(r"sub(\d+)_?DAQ", path.stem)
    if not m:
        return None
    return int(m.group(1))


# ---------------------------------------------------------------------
# NeuroKit-Processing + Fensterbau
# ---------------------------------------------------------------------


def neurokit_process_subject(df_raw: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Nimmt Roh-DAQ-Daten (Original-Sampling ~1000 Hz) und erzeugt NeuroKit-Features
    mit gleicher Länge. Es wird NICHT heruntergesampelt – das passiert später.
    """
    t_sec = df_raw["daqtime"].to_numpy(dtype=float)
    fs = estimate_fs(t_sec)
    fs_int = int(round(fs))
    logger.info(f"NeuroKit: geschätzte Samplingrate ~{fs:.2f} Hz (fs_int={fs_int}).")

    signals = pd.DataFrame({"daqtime": t_sec})

    # ---------------- ECG ----------------
    try:
        ecg_sig = df_raw["ecg"].to_numpy(dtype=float)
        ecg_signals, _ = nk.ecg_process(ecg_sig, sampling_rate=fs_int)
        cols = ecg_signals.columns
        if "ECG_Clean" in cols and "ECG_Rate" in cols:
            signals["ecg_clean"] = ecg_signals["ECG_Clean"]
            signals["ecg_rate"] = ecg_signals["ECG_Rate"]
        else:
            raise KeyError(f"Erwartete Spalten ECG_Clean/ECG_Rate nicht gefunden, habe: {list(cols)}")
    except Exception as e:
        logger.warning(f"NeuroKit ECG-Processing fehlgeschlagen: {e!r}, verwende Roh-ECG.")
        ecg = df_raw["ecg"].to_numpy(dtype=float)
        signals["ecg_clean"] = ecg
        # sehr grobe „Rate“: Ableitung + Betrag (nur Fallback)
        signals["ecg_rate"] = np.abs(np.gradient(ecg))

    # ---------------- BVP → PPG ----------------
    try:
        ppg_sig = df_raw["bvp"].to_numpy(dtype=float)
        ppg_signals, _ = nk.ppg_process(ppg_sig, sampling_rate=fs_int)
        cols = ppg_signals.columns
        if "PPG_Clean" in cols and "PPG_Rate" in cols:
            signals["ppg_clean"] = ppg_signals["PPG_Clean"]
            signals["ppg_rate"] = ppg_signals["PPG_Rate"]
        else:
            raise KeyError(f"Erwartete Spalten PPG_Clean/PPG_Rate nicht gefunden, habe: {list(cols)}")
    except Exception as e:
        logger.warning(f"NeuroKit PPG-Processing fehlgeschlagen: {e!r}, verwende Roh-BVP.")
        bvp = df_raw["bvp"].to_numpy(dtype=float)
        signals["ppg_clean"] = bvp
        signals["ppg_rate"] = np.abs(np.gradient(bvp))

    # ---------------- GSR → EDA ----------------
    try:
        eda_sig = df_raw["gsr"].to_numpy(dtype=float)
        eda_signals, _ = nk.eda_process(eda_sig, sampling_rate=fs_int)
        cols = eda_signals.columns
        if "EDA_Tonic" in cols and "EDA_Phasic" in cols:
            signals["eda_tonic"] = eda_signals["EDA_Tonic"]
            signals["eda_phasic"] = eda_signals["EDA_Phasic"]
        else:
            raise KeyError(f"Erwartete Spalten EDA_Tonic/EDA_Phasic nicht gefunden, habe: {list(cols)}")
    except Exception as e:
        logger.warning(f"NeuroKit EDA-Processing fehlgeschlagen: {e!r}, verwende Roh-GSR.")
        gsr = df_raw["gsr"].to_numpy(dtype=float)
        signals["eda_tonic"] = gsr
        signals["eda_phasic"] = np.zeros_like(gsr)

    # ---------------- RSP ----------------
    try:
        rsp_sig = df_raw["rsp"].to_numpy(dtype=float)
        rsp_signals, _ = nk.rsp_process(rsp_sig, sampling_rate=fs_int)
        cols = rsp_signals.columns
        if "RSP_Clean" in cols and "RSP_Amplitude" in cols:
            signals["rsp_clean"] = rsp_signals["RSP_Clean"]
            signals["rsp_amplitude"] = rsp_signals["RSP_Amplitude"]
        else:
            raise KeyError(f"Erwartete Spalten RSP_Clean/RSP_Amplitude nicht gefunden, habe: {list(cols)}")
    except Exception as e:
        logger.warning(f"NeuroKit RSP-Processing fehlgeschlagen: {e!r}, verwende Roh-RSP.")
        rsp = df_raw["rsp"].to_numpy(dtype=float)
        signals["rsp_clean"] = rsp
        signals["rsp_amplitude"] = np.abs(rsp - np.mean(rsp))

    # ---------------- SKT ----------------
    skt = df_raw["skt"].to_numpy(dtype=float)
    signals["skt"] = skt

    # ---------------- EMG (3 Kanäle) ----------------
    # Wichtig: nur bei ausreichend hoher Fs NeuroKit verwenden, sonst direkt Fallback.
    for emg_col in ["emg_zygo", "emg_coru", "emg_trap"]:
        try:
            emg_sig = df_raw[emg_col].to_numpy(dtype=float)

            if fs_int < 200:
                # Zu niedrige Fs für NeuroKit-EMG-Filter → lieber direkt Fallback
                raise ValueError(
                    f"Samplingrate {fs_int} Hz zu niedrig für EMG-Processing, "
                    f"verwende Roh-EMG für {emg_col}."
                )

            emg_signals, _ = nk.emg_process(emg_sig, sampling_rate=fs_int)
            cols = emg_signals.columns
            if "EMG_Envelope" in cols:
                env = emg_signals["EMG_Envelope"]
                signals[f"{emg_col}_envelope"] = env
            else:
                raise KeyError(
                    f"EMG_Envelope nicht gefunden für {emg_col}, habe Spalten: {list(cols)}"
                )
        except Exception as e:
            logger.warning(
                f"NeuroKit EMG-Processing ({emg_col}) fehlgeschlagen: {e!r}, verwende Roh-EMG."
            )
            emg = df_raw[emg_col].to_numpy(dtype=float)
            # einfache Hüllkurve: zentrieren und Betrag
            signals[f"{emg_col}_envelope"] = np.abs(emg - np.mean(emg))

    return signals


def build_windows_from_raw_and_interp(cfg: TrainCfg, logger: logging.Logger) -> Dict[str, np.ndarray]:
    """
    Lädt für alle Subjekte die rohen DAQ-Daten, verarbeitet sie mit NeuroKit2
    (auf Original-Sampling), sampelt dann auf target_fs herunter und schneidet
    daraus Fenster anhand der interpolierten 'video'-Segmente.

    Rückgabe:
      X: (N, C, T)  float32
      y: (N,)       String-Labels ("scary", "bored", ...)
      subjects: (N,) int Subject-ID
    """
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    subj_list: List[int] = []

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.interp_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(cfg.raw_dir.glob("sub*_DAQ.txt"))
    if not raw_files:
        logger.warning("Keine sub*_DAQ.txt in %s gefunden.", cfg.raw_dir)

    window_samples = int(cfg.window_sec * cfg.target_fs)
    step_samples = int(cfg.step_sec * cfg.target_fs)

    logger.info(
        "Fenster: window_sec=%.1f, step_sec=%.1f, target_fs=%.1f -> window_samples=%d, step_samples=%d",
        cfg.window_sec,
        cfg.step_sec,
        cfg.target_fs,
        window_samples,
        step_samples,
    )

    for raw_path in raw_files:
        subj_id = parse_subject_id_from_raw_name(raw_path)
        if subj_id is None:
            logger.warning("Konnte Subject-ID aus %s nicht parsen, überspringe.", raw_path.name)
            continue

        interp_path = cfg.interp_dir / f"sub_{subj_id}.csv"
        if not interp_path.exists():
            logger.warning("Interpolierte Datei %s fehlt, überspringe Subject %d.", interp_path.name, subj_id)
            continue

        logger.info("Lade RAW %s und INTERP %s ...", raw_path.name, interp_path.name)

        # Rohdaten (ohne Header)
        try:
            df_raw = pd.read_csv(
                raw_path,
                header=None,
                sep=r"\s+|,|\t",
                engine="python",
            )
        except Exception as e:
            logger.warning("Fehler beim Lesen von %s: %r", raw_path.name, e)
            continue

        if df_raw.shape[1] != 9:
            logger.warning("%s hat %d Spalten, erwarte 9. Überspringe.", raw_path.name, df_raw.shape[1])
            continue

        df_raw.columns = RAW_COLS

        df_raw = df_raw.sort_values("daqtime").reset_index(drop=True)
        t_raw_sec = df_raw["daqtime"].to_numpy(dtype=float)

        # Interpoliert (mit video-Spalte, daqtime in Millisekunden)
        try:
            df_interp = pd.read_csv(interp_path)
        except Exception as e:
            logger.warning("Fehler beim Lesen von %s: %r", interp_path.name, e)
            continue

        if "daqtime" not in df_interp.columns or "video" not in df_interp.columns:
            logger.warning("%s hat keine Spalten 'daqtime' und 'video'. Überspringe.", interp_path.name)
            continue

        t_interp_sec = df_interp["daqtime"].to_numpy(dtype=float) / 1000.0
        video_ids_interp = df_interp["video"].to_numpy(dtype=int)

        # 1) NeuroKit auf Original-Sampling
        df_nk = neurokit_process_subject(df_raw, logger)

        # 2) Jetzt erst Downsampling auf target_fs
        ds_factor = compute_downsample_factor(t_raw_sec, cfg.target_fs)
        idx_raw_ds = np.arange(0, len(df_nk), ds_factor, dtype=int)

        t_raw_ds = t_raw_sec[idx_raw_ds]
        sig_raw_ds = df_nk[NK_FEATURE_COLS].to_numpy(dtype=float)[idx_raw_ds].T  # (C, T_total)

        if sig_raw_ds.shape[1] < window_samples:
            logger.info(
                "Sub %d: zu wenig Samples nach Downsampling (%d), brauche >= %d. Überspringe.",
                subj_id,
                sig_raw_ds.shape[1],
                window_samples,
            )
            continue

        # 3) Segmente aus interpolierten Daten ermitteln (video konstant)
        changes = np.where(video_ids_interp[1:] != video_ids_interp[:-1])[0] + 1
        boundaries = np.concatenate(([0], changes, [len(video_ids_interp)]))

        for i in range(len(boundaries) - 1):
            start_i = boundaries[i]
            end_i = boundaries[i + 1]  # exklusiv
            video_id = int(video_ids_interp[start_i])

            if cfg.exclude_pauses and video_id in EXCLUDE_VIDEO_IDS:
                continue

            label = VIDEO_TO_LABEL.get(video_id, None)
            if label is None:
                # uninteressante Videos ignorieren
                continue

            seg_start_time = t_interp_sec[start_i]
            seg_end_time = t_interp_sec[end_i - 1]

            # Indexbereich im downsampled RAW, der in [seg_start_time, seg_end_time] liegt
            mask_raw = (t_raw_ds >= seg_start_time) & (t_raw_ds <= seg_end_time)
            if not np.any(mask_raw):
                continue

            idx_seg = np.where(mask_raw)[0]
            start_raw = idx_seg[0]
            end_raw = idx_seg[-1] + 1  # exklusiv

            seg_len = end_raw - start_raw
            if seg_len < window_samples:
                # Segment zu kurz für ein Fenster
                continue

            # Sliding Windows in diesem Segment
            for s in range(start_raw, end_raw - window_samples + 1, step_samples):
                e = s + window_samples
                if e > end_raw:
                    break
                X_win = sig_raw_ds[:, s:e]  # (C, window_samples)
                X_list.append(X_win.astype(np.float32))
                y_list.append(label)
                subj_list.append(subj_id)

        logger.info(
            "Sub %d: aktuelle Gesamtzahl Fenster: %d",
            subj_id,
            len(X_list),
        )

    if not X_list:
        raise RuntimeError("Keine Fenster erzeugt – prüfe Pfade/Konfiguration.")

    X = np.stack(X_list, axis=0)  # (N, C, T)
    y = np.array(y_list, dtype=object)
    subjects = np.array(subj_list, dtype=int)

    uniq, cnt = np.unique(y, return_counts=True)
    logger.info("Gesamt-Fenster: %d", X.shape[0])
    logger.info("Label-Verteilung: %s", dict(zip(uniq, cnt)))

    return {"X": X, "y": y, "subjects": subjects}


# ---------------------------------------------------------------------
# Normalisierung
# ---------------------------------------------------------------------


def compute_channel_norm_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # X: (N, C, T)
    means = X.mean(axis=(0, 2))
    stds = X.std(axis=(0, 2))
    stds = np.where(stds < 1e-8, 1.0, stds)
    return means.astype(np.float32), stds.astype(np.float32)


def apply_channel_norm(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return ((X - means[None, :, None]) / stds[None, :, None]).astype(np.float32)


# ---------------------------------------------------------------------
# Torch Dataset & CNN
# ---------------------------------------------------------------------


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is None:
            return x
        return x, self.y[idx]


class Cnn1D(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, cfg: TrainCfg):
        super().__init__()

        # Deutlich größeres zeitliches Rezeptives Feld + Pooling
        self.conv1 = nn.Conv1d(
            in_channels=n_channels,
            out_channels=cfg.conv_channels1,
            kernel_size=51,
            padding=25,
        )
        self.bn1 = nn.BatchNorm1d(cfg.conv_channels1)
        self.pool1 = nn.MaxPool1d(kernel_size=4)  # reduziert T

        self.conv2 = nn.Conv1d(
            in_channels=cfg.conv_channels1,
            out_channels=cfg.conv_channels2,
            kernel_size=25,
            padding=12,
        )
        self.bn2 = nn.BatchNorm1d(cfg.conv_channels2)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(
            in_channels=cfg.conv_channels2,
            out_channels=cfg.conv_channels3,
            kernel_size=9,
            padding=4,
        )
        self.bn3 = nn.BatchNorm1d(cfg.conv_channels3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(cfg.conv_channels3, cfg.linear_hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc_out = nn.Linear(cfg.linear_hidden, n_classes)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.act(self.bn3(self.conv3(x)))

        x = self.global_pool(x)  # (B, C3, 1)
        x = x.squeeze(-1)        # (B, C3)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


def train_cnn_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray | None,
    y_va: np.ndarray | None,
    n_channels: int,
    n_classes: int,
    cfg: TrainCfg,
    logger: logging.Logger,
) -> Tuple[Cnn1D, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Device: {device}")

    model = Cnn1D(n_channels, n_classes, cfg).to(device)

    # Klassengewichte gegen Imbalance
    class_counts = np.bincount(y_tr, minlength=n_classes)
    class_weights = class_counts.sum() / np.maximum(class_counts * n_classes, 1)
    class_weights = torch.from_numpy(class_weights.astype(np.float32)).to(device)
    logger.info(f"Klassenzählung: {class_counts}, Gewichte: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_ds = SequenceDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    if X_va is not None and y_va is not None:
        val_ds = SequenceDataset(X_va, y_va)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    else:
        val_loader = None

    best_state = None
    best_val_loss = float("inf")
    history: Dict[str, float] = {}
    epochs_no_improve = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        n_samples = 0
        train_loss = 0.0

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            train_loss += loss.item() * bs
            n_samples += bs

        train_loss /= max(1, n_samples)
        msg = f"Epoch {epoch}/{cfg.n_epochs} | train_loss={train_loss:.4f}"

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    bs = y.size(0)
                    val_loss += loss.item() * bs
                    val_samples += bs
            val_loss /= max(1, val_samples)
            msg += f" | val_loss={val_loss:.4f}"

            if val_loss < best_val_loss - 1e-4:  # kleiner Toleranzschwellenwert
                best_val_loss = val_loss
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= cfg.early_stopping_patience:
                logger.info(
                    f"Early Stopping nach {epoch} Epochen (keine Verbesserung der Val-Loss seit {cfg.early_stopping_patience} Epochen)."
                )
                logger.info(msg)
                break

        logger.info(msg)

    if best_state is not None and val_loader is not None:
        model.load_state_dict(best_state)
        history["best_val_loss"] = float(best_val_loss)
    else:
        history["best_val_loss"] = float("nan")

    return model, history


def predict_cnn(model: Cnn1D, X: np.ndarray, cfg: TrainCfg) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    ds = SequenceDataset(X, None)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            logits = model(x)
            y_hat = torch.argmax(logits, dim=1)
            preds.append(y_hat.cpu().numpy())

    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------
# CV & final Fit
# ---------------------------------------------------------------------


def run_group_cv_cnn(
    X: np.ndarray,
    y_str: np.ndarray,
    subjects: np.ndarray,
    cfg: TrainCfg,
    logger: logging.Logger,
) -> Dict[str, Any]:
    if cfg.classes_to_keep is None:
        raise ValueError("cfg.classes_to_keep muss gesetzt sein.")

    mask = np.isin(y_str, cfg.classes_to_keep)
    X = X[mask]
    y_str = y_str[mask]
    subjects = subjects[mask]

    if X.shape[0] == 0:
        raise RuntimeError(f"Keine Samples für Klassen {cfg.classes_to_keep}.")

    logger.info("Nach Klassenfilter %s: %d Fenster", cfg.classes_to_keep, X.shape[0])

    unique_labels = sorted(cfg.classes_to_keep)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    y = np.array([label_to_idx[lab] for lab in y_str], dtype=np.int64)
    n_channels = X.shape[1]

    gkf = GroupKFold(n_splits=cfg.n_folds)
    all_true = []
    all_pred = []
    fold_rows = []

    t0_all = time.perf_counter()
    fold = 0

    for tr_idx, va_idx in gkf.split(X, y, groups=subjects):
        fold += 1
        t0 = time.perf_counter()
        X_tr_raw = X[tr_idx]
        y_tr = y[tr_idx]
        X_va_raw = X[va_idx]
        y_va = y[va_idx]

        logger.info("Fold %d: Train=%d, Val=%d", fold, X_tr_raw.shape[0], X_va_raw.shape[0])

        means, stds = compute_channel_norm_stats(X_tr_raw)
        X_tr = apply_channel_norm(X_tr_raw, means, stds)
        X_va = apply_channel_norm(X_va_raw, means, stds)

        model, hist = train_cnn_model(
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            n_channels=n_channels,
            n_classes=len(unique_labels),
            cfg=cfg,
            logger=logger,
        )

        y_hat_idx = predict_cnn(model, X_va, cfg)
        y_va_str = np.array([idx_to_label[i] for i in y_va], dtype=object)
        y_hat_str = np.array([idx_to_label[i] for i in y_hat_idx], dtype=object)

        acc = accuracy_score(y_va_str, y_hat_str)
        bacc = balanced_accuracy_score(y_va_str, y_hat_str)
        f1m = f1_score(y_va_str, y_hat_str, average="macro")

        logger.info(
            "Fold %d/%d: acc=%.4f | bAcc=%.4f | f1_macro=%.4f",
            fold, cfg.n_folds, acc, bacc, f1m
        )

        cm_labels = sorted(np.unique(np.concatenate([y_va_str, y_hat_str])))
        cm = confusion_matrix(y_va_str, y_hat_str, labels=cm_labels)
        pd.DataFrame(cm, index=cm_labels, columns=cm_labels).to_csv(
            cfg.out_dir / f"cm_fold{fold}.csv", index=True
        )

        with open(cfg.out_dir / f"report_fold{fold}.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_va_str, y_hat_str, digits=3))

        fold_rows.append(
            {
                "fold": fold,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "f1_macro": f1m,
                "best_val_loss": hist.get("best_val_loss", float("nan")),
            }
        )

        all_true.append(y_va_str)
        all_pred.append(y_hat_str)

        logger.info("Fold %d done in %.1fs", fold, time.perf_counter() - t0)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    acc_all = accuracy_score(all_true, all_pred)
    bacc_all = balanced_accuracy_score(all_true, all_pred)
    f1m_all = f1_score(all_true, all_pred, average="macro")

    logger.info("== Pooled results ==")
    logger.info("acc=%.4f | bAcc=%.4f | f1_macro=%.4f", acc_all, bacc_all, f1m_all)

    labels_sorted = sorted(np.unique(np.concatenate([all_true, all_pred])))
    cm_all = confusion_matrix(all_true, all_pred, labels=labels_sorted)
    pd.DataFrame(cm_all, index=labels_sorted, columns=labels_sorted).to_csv(
        cfg.out_dir / "confusion_matrix_pooled.csv", index=True
    )

    with open(cfg.out_dir / "classification_report_pooled.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(all_true, all_pred, digits=3))

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(cfg.out_dir / "metrics_per_fold.csv", index=False)

    summary = {
        "pooled_accuracy": float(acc_all),
        "pooled_balanced_accuracy": float(bacc_all),
        "pooled_f1_macro": float(f1m_all),
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=1) if len(fold_df) > 1 else 0.0),
        "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=1) if len(fold_df) > 1 else 0.0),
        "f1_macro_mean": float(fold_df["f1_macro"].mean()),
        "f1_macro_std": float(fold_df["f1_macro"].std(ddof=1) if len(fold_df) > 1 else 0.0),
        "labels": unique_labels,
        "channels": NK_FEATURE_COLS,
    }

    with open(cfg.out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("CV fertig in %.1fs", time.perf_counter() - t0_all)
    return summary


def refit_final_cnn(
    X: np.ndarray,
    y_str: np.ndarray,
    subjects: np.ndarray,
    cfg: TrainCfg,
    logger: logging.Logger,
) -> None:
    if cfg.classes_to_keep is None:
        raise ValueError("cfg.classes_to_keep muss gesetzt sein.")

    mask = np.isin(y_str, cfg.classes_to_keep)
    X = X[mask]
    y_str = y_str[mask]

    if X.shape[0] == 0:
        raise RuntimeError(f"Keine Samples für Klassen {cfg.classes_to_keep}.")

    unique_labels = sorted(cfg.classes_to_keep)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    y = np.array([label_to_idx[lab] for lab in y_str], dtype=np.int64)

    means, stds = compute_channel_norm_stats(X)
    X_norm = apply_channel_norm(X, means, stds)

    n_channels = X_norm.shape[1]
    logger.info("Finaler Fit auf %d Samples, Klassen=%s", X_norm.shape[0], unique_labels)

    # finaler Fit: kein Val-Set → trotzdem train_cnn_model, aber val= None
    model, hist = train_cnn_model(
        X_tr=X_norm,
        y_tr=y,
        X_va=None,
        y_va=None,
        n_channels=n_channels,
        n_classes=len(unique_labels),
        cfg=cfg,
        logger=logger,
    )

    out_path = cfg.out_dir / "final_cnn_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
            "channels": NK_FEATURE_COLS,
            "labels": unique_labels,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "means": means,
            "stds": stds,
            "window_sec": cfg.window_sec,
            "step_sec": cfg.step_sec,
            "target_fs": cfg.target_fs,
        },
        out_path,
    )
    logger.info("Finales CNN-Modell gespeichert nach %s", out_path)


# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------


def cfg_to_serializable(cfg: TrainCfg):
    out = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


if __name__ == "__main__":
    raw_dir = Path(
        r"C:\Users\metin\OneDrive\Desktop\Informatik\10.Semester\thesis\emotion-recognition\case_dataset-master\data\raw\physiological"
    )
    interp_dir = Path(
        r"C:\Users\metin\OneDrive\Desktop\Informatik\10.Semester\thesis\emotion-recognition\case_dataset-master\data\interpolated\physiological"
    )

    # Basisdaten einmal bauen
    tmp_cfg = TrainCfg(
        raw_dir=raw_dir,
        interp_dir=interp_dir,
        out_dir=Path("outputs_cnn_raw_dummy"),
    )
    tmp_logger = setup_logger(tmp_cfg.out_dir, "INFO")
    data = build_windows_from_raw_and_interp(tmp_cfg, tmp_logger)
    X_all = data["X"]
    y_all = data["y"]
    subjects_all = data["subjects"]

    # 2-Klassen-Experimente: (Name, Klassen, window_sec, step_sec)
    experiments = [
        ("scary_vs_amused", ["scary", "amused"], 8.0, 2.0),
        ("bored_vs_relaxed", ["bored", "relaxed"], 12.0, 3.0),
        ("scary_vs_bored", ["scary", "bored"], 10.0, 2.0),
        ("amused_vs_bored", ["amused", "bored"], 8.0, 2.0),
    ]

    for exp_name, classes, window_sec, step_sec in experiments:
        print(f"\n=== Starte CNN-Raw-Experiment: {exp_name} ({classes}) ===")
        cfg = TrainCfg(
            raw_dir=raw_dir,
            interp_dir=interp_dir,
            out_dir=Path(f"outputs_cnn_raw/{exp_name}"),
            random_state=42,
            classes_to_keep=classes,
            window_sec=window_sec,
            step_sec=step_sec,
            target_fs=50.0,
            n_folds=5,
            n_epochs=20,
            batch_size=128,
            learning_rate=1e-4,
            weight_decay=1e-4,
            conv_channels1=32,
            conv_channels2=64,
            conv_channels3=128,
            linear_hidden=64,
            dropout=0.4,
            early_stopping_patience=7,
        )

        np.random.seed(cfg.random_state)
        random.seed(cfg.random_state)
        torch.manual_seed(cfg.random_state)

        logger = setup_logger(cfg.out_dir, cfg.log_level)

        summary = run_group_cv_cnn(
            X=X_all,
            y_str=y_all,
            subjects=subjects_all,
            cfg=cfg,
            logger=logger,
        )

        refit_final_cnn(
            X=X_all,
            y_str=y_all,
            subjects=subjects_all,
            cfg=cfg,
            logger=logger,
        )

        with open(cfg.out_dir / "summary_used_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "cfg": cfg_to_serializable(cfg),
                "cv_summary": summary
            }, f, indent=2)

        logger.info("Experiment %s fertig.", exp_name)
