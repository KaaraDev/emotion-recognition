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
# Configuration
# ----------------------------------------------------------

@dataclass
class PlotCfg:
    noninterp_phys_dir: Path
    model_root: Path  # Ordner, nicht .pt Datei!
    out_root: Path  # Ordner, nicht einzelne PNG!

    window_size_s: int = 60
    step_size_s: int = 30
    target_fs: int = 100
    min_samples_in_window: int = 50

    fold: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_path(self):
        # Modell abhängig von window_size / step_size
        return self.model_root / f"scary_vs_boring_fold{self.fold}.pt"

    @property
    def out_path(self):
        # Ausgabe in eigenen Unterordnern
        out = self.out_root / f"scary_vs_boring_fold{self.fold}_ws{self.window_size_s}_ss{self.step_size_s}"
        out.mkdir(exist_ok=True, parents=True)
        return out


# ----------------------------------------------------------
# CNN model (must be identical to the training model!)
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
# Load sub_*.csv
# ----------------------------------------------------------

def load_subject_phys_files(phys_dir: Path) -> Dict[int, pd.DataFrame]:
    csv_files = list(phys_dir.glob("sub_*.csv"))
    if not csv_files:
        raise RuntimeError(f"No sub_*.csv files found in {phys_dir}")

    subject_dfs: Dict[int, pd.DataFrame] = {}

    for f in csv_files:
        sid = int("".join(ch for ch in f.stem if ch.isdigit()))
        df = pd.read_csv(f)
        if "daqtime" not in df.columns or "video" not in df.columns:
            raise ValueError(f"{f.name} does not contain 'daqtime' and 'video' columns")
        # daqtime is in milliseconds
        df["subject_id"] = sid
        subject_dfs[sid] = df

    return subject_dfs


# ----------------------------------------------------------
# Windowing: build windows + keep track of video + start/end time
# ----------------------------------------------------------

def build_windows(
        subject_dfs: Dict[int, pd.DataFrame],
        cfg: PlotCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X                 : [N, C, T]
      y_str             : [N] (amusing/boring/relaxed/scary/...)
      groups            : [N] (subject_id)
      channel_cols
      window_vid        : [N] (video ID per window, e.g. 3,4,7,...)
      window_t_start_ms : [N] (window start time in ms, daqtime)
      window_t_end_ms   : [N] (window end time in ms, daqtime)
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

    # Window size / step in milliseconds
    window_ms = cfg.window_size_s * 1000
    step_ms = cfg.step_size_s * 1000
    target_len = cfg.window_size_s * cfg.target_fs  # e.g. 60s * 100Hz = 6000

    all_X: List[np.ndarray] = []
    all_y: List[str] = []
    all_groups: List[int] = []
    all_vids: List[int] = []
    all_t_start_ms: List[int] = []
    all_t_end_ms: List[int] = []

    for sid, df in subject_dfs.items():
        df = df.sort_values("daqtime").reset_index(drop=True)

        # Normalization per subject & channel
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

            tmin_ms = d["daqtime"].min()
            tmax_ms = d["daqtime"].max()
            if tmax_ms - tmin_ms < window_ms:
                continue

            starts_ms = np.arange(tmin_ms, tmax_ms - window_ms + 1, step_ms, dtype=np.int64)

            for t_start_ms in starts_ms:
                t_end_ms = t_start_ms + window_ms
                seg = d[(d["daqtime"] >= t_start_ms) & (d["daqtime"] < t_end_ms)]

                if len(seg) < cfg.min_samples_in_window:
                    continue

                t_seg_ms = seg["daqtime"].values.astype(np.float64)
                if len(np.unique(t_seg_ms)) < 2:
                    continue

                # Interpolation time axis in ms
                t_uniform_ms = np.linspace(
                    t_start_ms, t_end_ms, target_len, endpoint=False, dtype=np.float64
                )

                x_win = np.zeros((len(channel_cols), target_len), dtype=np.float32)
                window_ok = True

                for ci, col in enumerate(channel_cols):
                    vals = seg[col].values.astype(np.float64)
                    finite_mask = np.isfinite(vals) & np.isfinite(t_seg_ms)
                    if finite_mask.sum() < 2:
                        window_ok = False
                        break

                    vals = vals[finite_mask]
                    t_seg_chan_ms = t_seg_ms[finite_mask]

                    vals = (vals - means[col]) / stds[col]
                    vals = np.clip(vals, -5.0, 5.0)

                    try:
                        x_win[ci] = np.interp(
                            t_uniform_ms, t_seg_chan_ms, vals
                        ).astype(np.float32)
                    except Exception:
                        window_ok = False
                        break

                if not window_ok or np.isnan(x_win).any():
                    continue

                all_X.append(x_win)
                all_y.append(cls)
                all_groups.append(sid)
                all_vids.append(int(vid))
                all_t_start_ms.append(int(t_start_ms))
                all_t_end_ms.append(int(t_end_ms))

    if not all_X:
        raise RuntimeError("No windows were created.")

    X = np.stack(all_X, axis=0)
    y_str = np.array(all_y, dtype=object)
    groups = np.array(all_groups, dtype=np.int64)
    window_vid = np.array(all_vids, dtype=np.int64)
    window_t_start_ms = np.array(all_t_start_ms, dtype=np.int64)
    window_t_end_ms = np.array(all_t_end_ms, dtype=np.int64)

    print("Available classes:", sorted(set(y_str.tolist())))
    return X, y_str, groups, channel_cols, window_vid, window_t_start_ms, window_t_end_ms


# ----------------------------------------------------------
# Plot function: Subject 1, scary_vs_boring, Fold 3
# ----------------------------------------------------------

def plot_subjects(cfg: PlotCfg, subjects: List[int]):
    print("Loading data...")
    subject_dfs = load_subject_phys_files(cfg.noninterp_phys_dir)

    print("Building windows...")
    (
        X,
        y_str,
        groups,
        channel_cols,
        window_vid,
        window_t_start_ms,
        window_t_end_ms,
    ) = build_windows(subject_dfs, cfg)

    exp_pos = "scary"
    exp_neg = "boring"

    # Load model once
    in_channels = X.shape[1]
    model = CNN1DVideoType(in_channels=in_channels, n_classes=2)
    print(f"Loading model: {cfg.model_path}")
    state = torch.load(cfg.model_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()

    # Loop over subjects
    for subject_id in subjects:
        print(f"\n=== Processing subject {subject_id} ===")

        mask_exp = np.isin(y_str, [exp_pos, exp_neg])
        mask_subj = (groups == subject_id)
        mask = mask_exp & mask_subj

        Xs = X[mask]
        ys_str = y_str[mask]
        vids = window_vid[mask]
        t_start_ms = window_t_start_ms[mask]
        t_end_ms = window_t_end_ms[mask]

        print(f"Number of windows for subject {subject_id}: {len(Xs)}")

        if len(Xs) == 0:
            print(f"No matching windows for subject {subject_id}. Skipping.")
            continue

        # Sort chronologically
        order = np.argsort(t_start_ms)
        Xs = Xs[order]
        ys_str = ys_str[order]
        vids = vids[order]
        t_start_ms = t_start_ms[order]
        t_end_ms = t_end_ms[order]

        ys = np.where(ys_str == exp_pos, 1, 0)

        # === Compute cumulative time axis per video ===
        df_subj = subject_dfs[subject_id]
        used_videos = np.unique(vids)

        video_info = {}
        for v in used_videos:
            d_vid = df_subj[df_subj["video"] == v]
            if d_vid.empty:
                continue
            v_start = int(d_vid["daqtime"].min())
            v_end = int(d_vid["daqtime"].max())
            video_info[v] = (v_start, v_end, v_end - v_start)

        videos_in_order = sorted(video_info.keys(), key=lambda v: video_info[v][0])

        offset_ms = {}
        cum_ms = 0
        video_segments_cum = []
        for v in videos_in_order:
            v_start, v_end, v_dur = video_info[v]
            offset_ms[v] = cum_ms
            video_segments_cum.append((v, cum_ms / 1000, (cum_ms + v_dur) / 1000))
            cum_ms += v_dur

        # Compute x positions
        t_pos_ms = t_end_ms
        x_time_s = np.zeros_like(t_pos_ms, dtype=float)
        for i in range(len(t_pos_ms)):
            v = vids[i]
            v_start, _, _ = video_info[v]
            rel = t_pos_ms[i] - v_start
            x_time_s[i] = (offset_ms[v] + rel) / 1000.0

        # === Model predictions ===
        with torch.no_grad():
            X_tensor = torch.from_numpy(Xs).to(cfg.device)
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

        # === Aggregated probabilities ===
        agg_p_boring = np.zeros(len(ys))
        agg_p_scary = np.zeros(len(ys))

        count_total = {}
        count_boring = {}
        count_scary = {}

        for i, v in enumerate(vids):
            if v not in count_total:
                count_total[v] = count_boring[v] = count_scary[v] = 0

            count_total[v] += 1
            if preds[i] == 0:
                count_boring[v] += 1
            else:
                count_scary[v] += 1

            agg_p_boring[i] = count_boring[v] / count_total[v]
            agg_p_scary[i] = count_scary[v] / count_total[v]

        # === PLOT ===
        fig_path = cfg.out_path / f"subject{subject_id}.png"

        plot_single_subject(
            subject_id,
            cfg,
            vids,
            x_time_s,
            ys,
            preds,
            video_segments_cum,
            agg_p_boring,
            agg_p_scary,
            fig_path
        )

        print(f"Saved: {fig_path}")


def plot_single_subject(subject_id, cfg, vids, x_time_s, ys, preds,
                        video_segments_cum, agg_p_boring, agg_p_scary, fig_path):
    import matplotlib.pyplot as plt
    import numpy as np

    exp_pos = "scary"
    exp_neg = "boring"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 7),
        gridspec_kw={"height_ratios": [2, 2]}
    )

    # TOP PANEL
    ax1.scatter(x_time_s, ys, marker="o", label="True", alpha=0.7)
    ax1.scatter(x_time_s, preds, marker="x", label="Pred", alpha=0.7)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels([exp_neg, exp_pos])
    ax1.set_ylabel("Class")

    for vid, s, e in video_segments_cum:
        ax1.axvline(e, linestyle="--", alpha=0.3)
        mid = (s + e) / 2
        ax1.text(mid, 1.2, f"Video {vid}", ha="center", va="top")

    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    # BOTTOM PANEL
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Aggregated probability")
    ax2.set_xlabel("Time (s)")

    video_start_s_map = {vid: s for vid, s, _ in video_segments_cum}
    vids_arr = np.array(vids)

    first = True
    for vid, start_s, _ in video_segments_cum:
        m = vids_arr == vid
        if not np.any(m): continue

        xs = np.concatenate([[start_s], x_time_s[m]])
        pb = np.concatenate([[0.5], agg_p_boring[m]])
        ps = np.concatenate([[0.5], agg_p_scary[m]])

        ax2.plot(xs, pb, marker=".", color="tab:blue",
                 label="P(boring)" if first else None)
        ax2.plot(xs, ps, marker=".", color="tab:orange",
                 label="P(scary)" if first else None)
        first = False

    for _, _, e in video_segments_cum:
        ax2.axvline(e, linestyle="--", alpha=0.3)

    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    cfg = PlotCfg(
        noninterp_phys_dir=Path("../../case_dataset-master/data/non-interpolated/physiological"),
        model_root=Path("../outputs_cnn_noninterp_binary_20w10s/scary_vs_boring"),
        out_root=Path("figs/scary_vs_boring_plots"),

        window_size_s=20,
        step_size_s=10,

        fold=3,
    )

    plot_subjects(cfg, subjects=[3, 8, 13, 18, 23, 28])


if __name__ == "__main__":
    main()
