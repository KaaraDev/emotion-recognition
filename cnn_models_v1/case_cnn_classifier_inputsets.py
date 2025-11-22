# train_cnn_video_type_noninterp_binary_inputsets.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


# ----------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------

@dataclass
class TrainCfg:
    noninterp_phys_dir: Path
    out_dir: Path

    # Fenster / Sampling
    window_size_s: int = 60
    step_size_s: int = 30
    target_fs: int = 100
    min_samples_in_window: int = 50

    # Training
    batch_size: int = 64
    num_epochs: int = 60
    lr: float = 5e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    early_stopping_patience: int = 3  # Epochen ohne Verbesserung

    # CV
    random_state: int = 42
    n_splits: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ----------------------------------------------------------
# Loader für sub_*.csv
# ----------------------------------------------------------

def load_subject_phys_files(phys_dir: Path) -> Dict[int, pd.DataFrame]:
    """
    Lädt alle sub_*.csv der non-interpolated physiologischen Daten.
    Fügt eine Spalte 'subject_id' hinzu und gibt ein Dict {sid: df}.
    """
    if not any(phys_dir.glob("sub_*.csv")) and (phys_dir / "physiological").exists():
        phys_dir = phys_dir / "physiological"

    csv_files = list(phys_dir.glob("sub_*.csv"))
    if not csv_files:
        raise RuntimeError(f"Keine sub_*.csv-Dateien in {phys_dir} gefunden.")

    subject_dfs: Dict[int, pd.DataFrame] = {}

    for f in csv_files:
        name = f.stem.lower()  # z.B. "sub_1"

        # subject_id aus "sub_1" extrahieren
        if name.startswith("sub__"):
            sid = int(name.split("_")[1])
        else:
            digits = "".join(ch for ch in name if ch.isdigit())
            sid = int(digits)

        df = pd.read_csv(f)

        if "daqtime" not in df.columns:
            raise ValueError(f"Spalte 'daqtime' fehlt in {f.name}")
        if "video" not in df.columns:
            raise ValueError(f"Spalte 'video' fehlt in {f.name}")

        df["subject_id"] = sid
        subject_dfs[sid] = df

    print(f"{len(subject_dfs)} Subjects geladen aus {phys_dir}")
    return subject_dfs


# ----------------------------------------------------------
# Fenster erstellen (multiclass, aber ohne blu/start/end)
# + Normalisierung & robustere Behandlung von NaNs
# ----------------------------------------------------------

def build_windows_from_noninterp(
        subject_dfs: Dict[int, pd.DataFrame],
        cfg: TrainCfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Baut Fenster und gibt:
      X      : [N, C, T]
      y_str  : [N]      (String-Labels: 'amusing','boring','relaxed','scary')
      groups : [N]      (subject_ids)
      channel_cols : Kanäle (nach Ausschluss von daqtime, video, subject_id)
    zurück.

    Verbesserungen:
    - Z-Score Normalisierung pro Subject & Kanal
    - Clipping von Ausreißern (±5 Std-Abw.)
    - NaN-Prüfung pro Fenster / Kanal
    """

    video_type_map = {
        1: "amusing",
        2: "amusing",
        3: "boring",
        4: "boring",
        5: "relaxed",
        6: "relaxed",
        7: "scary",
        8: "scary",
        10: "start",
        11: "blu",
        12: "end",
    }

    excluded_types = {"blu", "start", "end"}

    all_X: List[np.ndarray] = []
    all_y_str: List[str] = []
    all_groups: List[int] = []

    example_df = next(iter(subject_dfs.values()))
    excluded_cols = {"daqtime", "video", "subject_id"}
    channel_cols = [c for c in example_df.columns if c not in excluded_cols]

    window_ms = cfg.window_size_s * 1000
    step_ms = cfg.step_size_s * 1000
    target_len = cfg.window_size_s * cfg.target_fs

    for sid, df in subject_dfs.items():
        df = df.sort_values("daqtime").reset_index(drop=True)

        # --- Normalisierung pro Subject & Kanal ---
        chan_stats: Dict[str, Tuple[float, float]] = {}
        for col in channel_cols:
            vals = df[col].values.astype(np.float64)
            # Falls NaNs vorhanden → mit finite Werten Statistik berechnen
            finite_mask = np.isfinite(vals)
            if finite_mask.sum() == 0:
                mean = 0.0
                std = 1.0
            else:
                mean = float(np.mean(vals[finite_mask]))
                std = float(np.std(vals[finite_mask]))
                if std < 1e-8:
                    std = 1.0
            chan_stats[col] = (mean, std)

        videos = df["video"].unique()

        for vid in videos:
            if vid not in video_type_map:
                continue

            video_type = video_type_map[vid]
            if video_type in excluded_types:
                continue

            df_vid = df[df["video"] == vid]
            if df_vid.empty:
                continue

            t_min = df_vid["daqtime"].min()
            t_max = df_vid["daqtime"].max()
            if t_max - t_min < window_ms:
                continue

            starts = np.arange(t_min, t_max - window_ms + 1, step_ms, dtype=np.int64)

            for t_start in starts:
                t_end = t_start + window_ms
                seg = df_vid[(df_vid["daqtime"] >= t_start) & (df_vid["daqtime"] < t_end)]

                if len(seg) < cfg.min_samples_in_window:
                    continue

                t_seg = seg["daqtime"].values.astype(np.float64)
                if len(np.unique(t_seg)) < 2:
                    # Für Interpolation brauchen wir mind. 2 unterschiedliche Zeitpunkte
                    continue

                t_uniform = np.linspace(
                    t_start,
                    t_end,
                    target_len,
                    endpoint=False,
                    dtype=np.float64
                )

                x_resampled = np.zeros((len(channel_cols), target_len), dtype=np.float32)
                window_ok = True

                for ci, col in enumerate(channel_cols):
                    vals = seg[col].values.astype(np.float64)

                    # NaNs pro Kanal entfernen
                    finite_mask = np.isfinite(vals) & np.isfinite(t_seg)
                    if finite_mask.sum() < 2:
                        window_ok = False
                        break

                    vals = vals[finite_mask]
                    t_seg_chan = t_seg[finite_mask]

                    mean, std = chan_stats[col]
                    vals = (vals - mean) / std
                    # Clipping zur Robustheit ggü. Ausreißern
                    vals = np.clip(vals, -5.0, 5.0)

                    # Interpolation
                    try:
                        x_resampled[ci] = np.interp(
                            t_uniform,
                            t_seg_chan,
                            vals
                        ).astype(np.float32)
                    except Exception:
                        window_ok = False
                        break

                if not window_ok:
                    continue

                if np.isnan(x_resampled).any():
                    continue

                all_X.append(x_resampled)
                all_y_str.append(video_type)
                all_groups.append(sid)

    if not all_X:
        raise RuntimeError("Keine Trainingsfenster erzeugt – überprüfe Parameter / Mapping / Normalisierung.")

    X = np.stack(all_X, axis=0)
    y_str = np.array(all_y_str, dtype=object)
    groups = np.array(all_groups, dtype=np.int64)

    print("Verfügbare Klassen (nach Filter):", sorted(set(y_str.tolist())))
    return X, y_str, groups, channel_cols


# ----------------------------------------------------------
# CNN-Modell (größere Kapazität + mehr zeitliche Info)
# ----------------------------------------------------------

class CNN1DVideoType(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # statt alles auf 1 herunterzukochen, behalten wir 4 "zeitliche Slots"
            nn.AdaptiveAvgPool1d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 128 * 4 = 512
            nn.Dropout(0.2),
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------
# Training / Evaluation
# ----------------------------------------------------------

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(Xb)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        criterion: nn.Module,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = 0

    all_y: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        logits = model(Xb)
        loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(Xb)
        total_correct += (preds == yb).sum().item()
        n += len(Xb)

        all_y.append(yb.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    if n == 0:
        return 0.0, 0.0, np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_preds)

    return total_loss / n, total_correct / n, y_true, y_pred


# ----------------------------------------------------------
# Kanal-Auswahl (Input-Kombinationen)
# ----------------------------------------------------------

def select_channels(
        X: np.ndarray,
        channel_cols: List[str],
        selected_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Wählt nur bestimmte Kanäle (per Name) aus X.
    X: [N, C, T]
    channel_cols: Länge C
    selected_names: Kanalnamen, die behalten werden sollen
    """
    name_to_idx = {name: i for i, name in enumerate(channel_cols)}
    idxs = []
    for name in selected_names:
        if name not in name_to_idx:
            raise ValueError(f"Gewünschter Kanal '{name}' ist nicht in channel_cols enthalten.")
        idxs.append(name_to_idx[name])
    idxs = np.array(idxs, dtype=int)
    X_sel = X[:, idxs, :]
    channel_cols_sel = [channel_cols[i] for i in idxs]
    return X_sel, channel_cols_sel


# ----------------------------------------------------------
# Experiment-Runner (Binary-Setups)
# ----------------------------------------------------------

def run_binary_experiment(
        exp_name: str,
        cls_pos: str,
        cls_neg: str,
        X: np.ndarray,
        y_str: np.ndarray,
        groups: np.ndarray,
        channel_cols: List[str],
        cfg: TrainCfg,
):
    """
    cls_pos: Name der Klasse, die Label 1 bekommt
    cls_neg: Name der Klasse, die Label 0 bekommt

    Verbesserungen:
    - Class-Weights für CrossEntropyLoss (bei Imbalance)
    - LR-Scheduler (ReduceLROnPlateau)
    - Early Stopping
    - Fold-Skip, wenn nicht beide Klassen im Train/Val vorkommen
    - Confusion-Matrix + erweiterter Report pro Fold
    """
    print(f"\n========== EXPERIMENT: {exp_name} ({cls_pos} vs {cls_neg}) ==========")

    # Filter: nur Fenster der beiden Klassen
    mask = np.isin(y_str, [cls_pos, cls_neg])
    X_exp = X[mask]
    y_str_exp = y_str[mask]
    groups_exp = groups[mask]

    if X_exp.shape[0] == 0:
        print("  -> Keine Fenster für dieses Klassenpaar. Überspringe.")
        return

    # String -> Binary: pos=1, neg=0
    y_bin = np.where(y_str_exp == cls_pos, 1, 0)

    device = cfg.device
    n_classes = 2
    in_channels = X_exp.shape[1]

    out_dir_exp = cfg.out_dir / exp_name
    out_dir_exp.mkdir(parents=True, exist_ok=True)

    print(f"  Samples: {X_exp.shape[0]}  | Kanäle: {in_channels}  | Klassenverteilung:")
    unique, counts = np.unique(y_bin, return_counts=True)
    for u, c in zip(unique, counts):
        label_name = cls_pos if u == 1 else cls_neg
        print(f"    Label {u} ({label_name}): {c}")

    gkf = GroupKFold(n_splits=cfg.n_splits)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_exp, y_bin, groups_exp), start=1):
        print(f"\n  --- Fold {fold}/{cfg.n_splits} ---")

        Xtr, ytr = X_exp[train_idx], y_bin[train_idx]
        Xva, yva = X_exp[val_idx], y_bin[val_idx]

        # <<< NEU: welche Subjekte sind in Train/Val?
        train_subjects = np.unique(groups_exp[train_idx])
        val_subjects = np.unique(groups_exp[val_idx])

        print(f"    Train-Subjekte (Fold {fold}): {train_subjects}")
        print(f"    Val-Subjekte   (Fold {fold}): {val_subjects}")

        # optional auch in Datei speichern:
        subj_path = out_dir_exp / f"{exp_name}_fold{fold}_subjects.txt"
        with open(subj_path, "w") as f_subj:
            f_subj.write(f"Experiment: {exp_name}\n")
            f_subj.write(f"Fold: {fold}\n\n")
            f_subj.write("Train-Subjekte:\n")
            f_subj.write(", ".join(str(s) for s in train_subjects) + "\n\n")
            f_subj.write("Val-Subjekte:\n")
            f_subj.write(", ".join(str(s) for s in val_subjects) + "\n")

        # Sicherstellen, dass beide Klassen in Train & Val vorkommen
        uniq_tr = np.unique(ytr)
        uniq_va = np.unique(yva)
        if len(uniq_tr) < 2 or len(uniq_va) < 2:
            print("  -> Fold enthält nicht beide Klassen in Train oder Val; überspringe diesen Fold.")
            continue

        train_loader = DataLoader(
            WindowDataset(Xtr, ytr),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            WindowDataset(Xva, yva),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        model = CNN1DVideoType(in_channels=in_channels, n_classes=n_classes).to(device)

        # Class-Weights gegen Imbalance
        class_counts = np.bincount(ytr, minlength=2)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

        best_acc = 0.0
        best_state = None
        no_improve_epochs = 0

        for epoch in range(1, cfg.num_epochs + 1):
            tloss = train_one_epoch(model, train_loader, optimizer, device, criterion)
            vloss, vacc, _, _ = eval_epoch(model, val_loader, device, criterion)

            print(
                f"    Epoch {epoch:02d} | "
                f"train_loss={tloss:.4f} | val_loss={vloss:.4f} | val_acc={vacc:.4f}"
            )

            scheduler.step(vacc)

            if vacc > best_acc + 1e-4:
                best_acc = vacc
                best_state = deepcopy(model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= cfg.early_stopping_patience:
                print(f"    Early Stopping nach {epoch} Epochen (keine Verbesserung seit {no_improve_epochs} Epochen).")
                break

        if best_state is not None:
            # Bestes Modell wiederherstellen und finale Val-Metriken + Confusion-Matrix berechnen
            model.load_state_dict(best_state)
            vloss_best, vacc_best, y_true, y_pred = eval_epoch(model, val_loader, device, criterion)

            # Modell speichern
            model_path = out_dir_exp / f"{exp_name}_fold{fold}.pt"
            torch.save(best_state, model_path)
            print(f"  -> Fold {fold}: best_val_acc={best_acc:.4f}, Modell gespeichert nach {model_path}")

            # Confusion-Matrix speichern
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_path = out_dir_exp / f"{exp_name}_fold{fold}_confusion_matrix.txt"
            np.savetxt(cm_path, cm, fmt="%d")
            print(f"  -> Confusion-Matrix gespeichert als {cm_path}")

            # Lesbaren Report speichern
            report_path = out_dir_exp / f"{exp_name}_fold{fold}.txt"
            with open(report_path, "w") as f:
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Fold: {fold}\n\n")
                f.write(f"Positive class (1): {cls_pos}\n")
                f.write(f"Negative class (0): {cls_neg}\n\n")
                f.write("Config:\n")
                f.write(f"  window_size_s = {cfg.window_size_s}\n")
                f.write(f"  step_size_s   = {cfg.step_size_s}\n")
                f.write(f"  target_fs     = {cfg.target_fs}\n")
                f.write(f"  min_samples_in_window = {cfg.min_samples_in_window}\n")
                f.write(f"  batch_size    = {cfg.batch_size}\n")
                f.write(f"  num_epochs    = {cfg.num_epochs}\n")
                f.write(f"  lr            = {cfg.lr}\n")
                f.write(f"  weight_decay  = {cfg.weight_decay}\n")
                f.write(f"  early_stopping_patience = {cfg.early_stopping_patience}\n")
                f.write(f"  n_splits      = {cfg.n_splits}\n\n")

                f.write(f"Best validation accuracy: {best_acc:.4f}\n")
                f.write(f"Final val_loss (best model re-eval): {vloss_best:.4f}\n")
                f.write(f"Final val_acc  (best model re-eval): {vacc_best:.4f}\n\n")

                f.write(f"Training samples:   {len(Xtr)}\n")
                f.write(f"Validation samples: {len(Xva)}\n\n")

                f.write("Class counts (train):\n")
                for u in np.unique(ytr):
                    cnt = int((ytr == u).sum())
                    name = cls_pos if u == 1 else cls_neg
                    f.write(f"  Label {u} ({name}): {cnt}\n")
                f.write("\n")

                f.write("Class counts (val):\n")
                for u in np.unique(yva):
                    cnt = int((yva == u).sum())
                    name = cls_pos if u == 1 else cls_neg
                    f.write(f"  Label {u} ({name}): {cnt}\n")
                f.write("\n")

                f.write("Confusion matrix (rows=true, cols=pred) [0,1]:\n")
                for row in cm:
                    f.write("  " + " ".join(f"{int(x):4d}" for x in row) + "\n")

            print(f"  -> Lesbarer Report gespeichert als {report_path}")
        else:
            print("  -> Kein bestes Modell für diesen Fold (evtl. alle Folds übersprungen).")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    cfg = TrainCfg(
        noninterp_phys_dir=Path(
            r"C:\Users\Metin\Desktop\Informatik\10.Semester\BA\emotion-recognition\case_dataset-master\data\non-interpolated\physiological"
        ),
        # NEUER OUTPUT-ORDNER, damit alte Ergebnisse nicht überschrieben werden
        out_dir=Path("outputs_cnn_noninterp_binary_inputsets"),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Seeds setzen für Reproduzierbarkeit
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)

    print("Lade physiologische non-interpolated Daten ...")
    subject_dfs = load_subject_phys_files(cfg.noninterp_phys_dir)

    print("Baue Fenster & Labels (multiclass, ohne blu/start/end) ...")
    X, y_str, groups, channel_cols = build_windows_from_noninterp(subject_dfs, cfg)
    print(f"X-Shape: {X.shape} (N, C, T)")
    print(f"y-Shape: {y_str.shape}")
    print(f"Kanäle: {channel_cols}")

    # --------------------------------------------------
    # Verschiedene Input-Kombinationen definieren
    # (wird defensiv gebaut – nur Sets, die wirklich Kanäle enthalten)
    # --------------------------------------------------
    input_sets: Dict[str, List[str]] = {}

    # 1) Alle Kanäle
    input_sets["all_channels"] = channel_cols

    # Hilfsfunktion: prüft, ob ein Pattern in irgendeinem Kanal vorkommt
    lower_cols = [c.lower() for c in channel_cols]

    # 2) Ohne EMG (falls es EMG-Kanäle gibt)
    if any("emg" in c for c in lower_cols):
        no_emg_cols = [c for c in channel_cols if "emg" not in c.lower()]
        if len(no_emg_cols) > 0 and len(no_emg_cols) < len(channel_cols):
            input_sets["no_emg"] = no_emg_cols

    # 3) Nur EDA / GSR (falls vorhanden)
    eda_cols = [c for c in channel_cols if ("eda" in c.lower() or "gsr" in c.lower())]
    if len(eda_cols) > 0:
        input_sets["eda_only"] = eda_cols

    # 4) Nur ECG (falls vorhanden)
    ecg_cols = [c for c in channel_cols if "ecg" in c.lower()]
    if len(ecg_cols) > 0:
        input_sets["ecg_only"] = ecg_cols

    # 5) Ohne RSP (falls vorhanden)
    if any("rsp" in c for c in lower_cols):
        no_rsp_cols = [c for c in channel_cols if "rsp" not in c.lower()]
        if len(no_rsp_cols) > 0 and len(no_rsp_cols) < len(channel_cols):
            input_sets["no_rsp"] = no_rsp_cols

    # 6) Ohne EDA / GSR (falls vorhanden)
    if any("rsp" in c for c in lower_cols):
        no_gsr_cols = [c for c in channel_cols if "gsr" not in c.lower()]
        if len(no_gsr_cols) > 0 and len(no_gsr_cols) < len(channel_cols):
            input_sets["no_gsr"] = no_gsr_cols

    # 7) Ohne ECG (falls vorhanden)
    if any("ecg" in c for c in lower_cols):
        no_ecg_cols = [c for c in channel_cols if ("ecg" in c.lower())]
        if len(no_ecg_cols) > 0:
            input_sets["no_ecg"] = no_ecg_cols

    # 8) Nur ECG, GSR, RSP, SKT (falls vorhanden)
    combined_cols = [
        c for c in channel_cols
        if any(k in c.lower() for k in ("ecg", "gsr", "eda", "rsp", "skt"))
    ]
    if len(combined_cols) > 0:
        input_sets["combined"] = combined_cols

    # 9) Gesichtsmuskeln
    face_cols = [c for c in channel_cols if any(k in c.lower() for k in ("emg_zygo", "emg_coru"))]
    if len(face_cols) > 0:
        input_sets["face"] = face_cols

    # Du kannst hier jederzeit noch manuell weitere Sets ergänzen,
    # z.B. nur bestimmte EMG-Kanäle etc.

    # Sicherheit: nur Sets behalten, die nicht leer sind
    input_sets = {name: cols for name, cols in input_sets.items() if len(cols) > 0}

    print("\nDefinierte Input-Sets:")
    for name, cols in input_sets.items():
        print(f"  {name}: {len(cols)} Kanäle")

    # Definiere deine Binary-Experimente
    experiments = [
        ("scary_vs_boring", "scary", "boring"),
        ("scary_vs_amusing", "scary", "amusing"),
        ("amusing_vs_boring", "amusing", "boring"),
        ("relaxed_vs_boring", "relaxed", "boring"),
    ]

    # Für jedes Input-Set alle Experimente laufen lassen
    for input_name, cols_sel in input_sets.items():
        print(f"\n==================== INPUT-SET: {input_name} ====================")
        X_sel, channel_cols_sel = select_channels(X, channel_cols, cols_sel)
        print(f"  Verwendete Kanäle ({len(channel_cols_sel)}): {channel_cols_sel}")

        for base_exp_name, cls_pos, cls_neg in experiments:
            exp_name = f"{base_exp_name}__inputs={input_name}"
            run_binary_experiment(
                exp_name=exp_name,
                cls_pos=cls_pos,
                cls_neg=cls_neg,
                X=X_sel,
                y_str=y_str,
                groups=groups,
                channel_cols=channel_cols_sel,
                cfg=cfg,
            )


if __name__ == "__main__":
    main()
