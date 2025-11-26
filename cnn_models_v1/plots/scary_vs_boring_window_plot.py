# plot_scary_vs_boring_subject1.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------

@dataclass
class PlotCfg:
    noninterp_phys_dir: Path
    model_path: Path
    out_path: Path

    window_size_s: int = 60
    step_size_s: int = 30
    target_fs: int = 100
    min_samples_in_window: int = 50

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
# CNN-Modell (muss identisch zum Trainingsmodell sein!)
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

            nn.AdaptiveAvgPool1d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------
# Lade sub_*.csv
# ----------------------------------------------------------

def load_subject_phys_files(phys_dir: Path) -> Dict[int, pd.DataFrame]:
    csv_files = list(phys_dir.glob("sub_*.csv"))
    if not csv_files:
        raise RuntimeError(f"Keine sub_*.csv Dateien in {phys_dir}")

    subject_dfs: Dict[int, pd.DataFrame] = {}

    for f in csv_files:
        sid = int("".join(ch for ch in f.stem if ch.isdigit()))
        df = pd.read_csv(f)
        if "daqtime" not in df.columns or "video" not in df.columns:
            raise ValueError(f"{f.name} enthält nicht die Spalten 'daqtime' und 'video'")
        df["subject_id"] = sid
        subject_dfs[sid] = df

    return subject_dfs


# ----------------------------------------------------------
# Windowing: baut Fenster + merkt sich, aus welchem Video
# ----------------------------------------------------------

def build_windows(
    subject_dfs: Dict[int, pd.DataFrame],
    cfg: PlotCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Gibt zurück:
      X          : [N, C, T]
      y_str      : [N] (amusing/boring/relaxed/scary/...)
      groups     : [N] (subject_id)
      channel_cols
      window_vid : [N] (Video-ID pro Fenster, z.B. 3,4,7,...)
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
    excluded = {"blu", "start", "end"}

    example_df = next(iter(subject_dfs.values()))
    excluded_cols = {"daqtime", "video", "subject_id"}
    channel_cols = [c for c in example_df.columns if c not in excluded_cols]

    window_ms = cfg.window_size_s * 1000
    step_ms = cfg.step_size_s * 1000
    target_len = cfg.window_size_s * cfg.target_fs

    all_X: List[np.ndarray] = []
    all_y: List[str] = []
    all_groups: List[int] = []
    all_vids: List[int] = []

    for sid, df in subject_dfs.items():
        df = df.sort_values("daqtime").reset_index(drop=True)

        # Normalisierung pro Subject & Kanal (wie im Trainingsskript)
        means = df[channel_cols].mean(numeric_only=True)
        stds = df[channel_cols].std(numeric_only=True).replace(0, 1)

        for vid in df["video"].unique():
            if vid not in video_type_map:
                continue
            cls = video_type_map[vid]
            if cls in excluded:
                continue

            d = df[df["video"] == vid]
            if d.empty:
                continue

            tmin, tmax = d["daqtime"].min(), d["daqtime"].max()
            if tmax - tmin < window_ms:
                continue

            starts = np.arange(tmin, tmax - window_ms + 1, step_ms, dtype=np.int64)

            for t_start in starts:
                t_end = t_start + window_ms
                seg = d[(d["daqtime"] >= t_start) & (d["daqtime"] < t_end)]

                if len(seg) < cfg.min_samples_in_window:
                    continue

                t_seg = seg["daqtime"].values.astype(np.float64)
                if len(np.unique(t_seg)) < 2:
                    continue

                t_uniform = np.linspace(t_start, t_end, target_len, endpoint=False, dtype=np.float64)

                x_win = np.zeros((len(channel_cols), target_len), dtype=np.float32)
                window_ok = True

                for ci, col in enumerate(channel_cols):
                    vals = seg[col].values.astype(np.float64)
                    finite_mask = np.isfinite(vals) & np.isfinite(t_seg)
                    if finite_mask.sum() < 2:
                        window_ok = False
                        break

                    vals = vals[finite_mask]
                    t_seg_chan = t_seg[finite_mask]

                    vals = (vals - means[col]) / stds[col]
                    vals = np.clip(vals, -5.0, 5.0)

                    try:
                        x_win[ci] = np.interp(t_uniform, t_seg_chan, vals).astype(np.float32)
                    except Exception:
                        window_ok = False
                        break

                if not window_ok or np.isnan(x_win).any():
                    continue

                all_X.append(x_win)
                all_y.append(cls)
                all_groups.append(sid)
                all_vids.append(int(vid))

    if not all_X:
        raise RuntimeError("Keine Fenster erzeugt.")

    X = np.stack(all_X, axis=0)
    y_str = np.array(all_y, dtype=object)
    groups = np.array(all_groups, dtype=np.int64)
    window_vid = np.array(all_vids, dtype=np.int64)

    print("Verfügbare Klassen:", sorted(set(y_str.tolist())))
    return X, y_str, groups, channel_cols, window_vid


# ----------------------------------------------------------
# Plot-Funktion: Subject 1, scary_vs_boring, Fold 3
# ----------------------------------------------------------

def plot_subject1(cfg: PlotCfg, subject_id: int = 1):
    print("Lade Daten...")
    subject_dfs = load_subject_phys_files(cfg.noninterp_phys_dir)

    print("Erstelle Fenster...")
    X, y_str, groups, channel_cols, window_vid = build_windows(subject_dfs, cfg)

    exp_pos = "scary"
    exp_neg = "boring"

    # Nur scary/boring + gewünschtes Subject
    mask_exp = np.isin(y_str, [exp_pos, exp_neg])
    mask_subj = (groups == subject_id)
    mask = mask_exp & mask_subj

    Xs = X[mask]
    ys_str = y_str[mask]
    vids = window_vid[mask]

    print(f"Gefundene Fenster für Subject {subject_id}: {len(Xs)}")

    if len(Xs) == 0:
        print("Keine passenden Fenster vorhanden.")
        return

    # String -> Binary: scary=1, boring=0
    ys = np.where(ys_str == exp_pos, 1, 0)

    # Modell laden (muss mit gleicher Kanalanzahl instanziert werden)
    in_channels = X.shape[1]
    model = CNN1DVideoType(in_channels=in_channels, n_classes=2)
    state = torch.load(cfg.model_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()

    # Vorhersagen + Softmax (Fenster-Ebene)
    with torch.no_grad():
        X_tensor = torch.from_numpy(Xs).to(cfg.device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # [N, 2]
        preds = probs.argmax(axis=1)  # 0=boring, 1=scary

    x = np.arange(len(ys))

    # ------------------------------------------------------
    # Aggregierte Wahrscheinlichkeiten pro Video über Zeit
    # basiert auf den vorhergesagten Klassen (Fenster-Zählung)
    # ------------------------------------------------------
    N = len(preds)
    agg_p_boring = np.zeros(N, dtype=np.float32)
    agg_p_scary = np.zeros(N, dtype=np.float32)

    # pro Video: Zähler
    count_total = {}
    count_boring = {}
    count_scary = {}

    for i, v in enumerate(vids):
        if v not in count_total:
            count_total[v] = 0
            count_boring[v] = 0
            count_scary[v] = 0

        count_total[v] += 1
        if preds[i] == 0:
            count_boring[v] += 1
        else:
            count_scary[v] += 1

        agg_p_boring[i] = count_boring[v] / count_total[v]
        agg_p_scary[i] = count_scary[v] / count_total[v]

    # ------------------------------------------------------
    # Figure mit 2 Subplots: oben Klassen, unten aggregierte Wkeiten
    # ------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 7),
        gridspec_kw={"height_ratios": [2, 2]}
    )

    # --- Oben: True vs Pred ---
    ax1.scatter(x, ys, marker="o", label="True", alpha=0.7)
    ax1.scatter(x, preds, marker="x", label="Pred", alpha=0.7)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels([exp_neg, exp_pos])
    ax1.set_ylabel("Klasse")
    ax1.set_title(f"Subject {subject_id} – scary_vs_boring – Fold 3\nFensterklassifikation")

    # --- Video-Grenzen markieren ---
    unique_vids_in_order = []
    boundaries = []   # x-Position, an der neues Video beginnt

    last_vid = vids[0]
    unique_vids_in_order.append(last_vid)
    for i in range(1, len(vids)):
        if vids[i] != last_vid:
            boundaries.append(i)
            unique_vids_in_order.append(vids[i])
            last_vid = vids[i]

    # Linien zeichnen und Video-Labels oben drüber
    y_min, y_max = -0.3, 1.3
    ax1.set_ylim(y_min, y_max)

    for b in boundaries:
        ax1.axvline(b, linestyle="--", alpha=0.3)

    # Textlabel in der Mitte jedes Video-Blocks
    block_starts = [0] + boundaries
    block_ends = boundaries + [len(x)]
    for v, start, end in zip(unique_vids_in_order, block_starts, block_ends):
        mid = (start + end) / 2.0
        ax1.text(
            mid,
            y_max + 0.05,
            f"Video {v}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    ax1.legend(loc="upper right")

    # --- Unten: aggregierte Wahrscheinlichkeiten pro Video ---
    ax2.plot(x, agg_p_boring, marker=".", linestyle="-", label=f"aggregiert P({exp_neg})")
    ax2.plot(x, agg_p_scary, marker=".", linestyle="-", label=f"aggregiert P({exp_pos})")

    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Aggregierte Wahrscheinlichkeit")
    ax2.set_xlabel("Fenster-Index (chronologische Reihenfolge für Subject 1)")
    ax2.set_title("Aggregierte Klassenwahrscheinlichkeit pro Video (kumulativ über Fenster)")

    # Video-Grenzen auch im unteren Plot
    for b in boundaries:
        ax2.axvline(b, linestyle="--", alpha=0.3)

    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    cfg.out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(cfg.out_path, dpi=150)
    plt.close()

    print(f"Plot gespeichert unter: {cfg.out_path}")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    cfg = PlotCfg(
        noninterp_phys_dir=Path(
            r"C:\Users\metin\OneDrive\Desktop\Informatik\10.Semester\thesis\emotion-recognition\case_dataset-master\data\non-interpolated\physiological"
        ),
        model_path=Path(
            r"..\outputs_cnn_noninterp_binary\scary_vs_boring\scary_vs_boring_fold3.pt"
        ),
        out_path=Path(
            r"figs/scary_vs_boring_subject1_predictions_fold3.png"
        ),
    )

    plot_subject1(cfg, subject_id=1)


if __name__ == "__main__":
    main()
