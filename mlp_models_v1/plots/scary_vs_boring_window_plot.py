# plot_mlp_scary_vs_bored_subjects.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

@dataclass
class PlotCfg:
    exp_name: str = "scary_vs_bored"
    fold: int = 3

    # Path to MLP outputs (contains cv_predictions.csv)
    mlp_out_dir: Path = Path("outputs_20w10s_whitelist_hyperopt/scary_vs_bored")

    # Where to save figures
    out_root: Path = Path("figs/mlp_scary_vs_bored_plots")

    # Optional: non-interpolated physiological dir (sub_*.csv with daqtime+video)
    # If provided and exists, we use it to determine per-subject video order + durations more robustly.
    noninterp_phys_dir: Optional[Path] = None

    # Labels in your MLP experiment
    exp_pos: str = "scary"
    exp_neg: str = "bored"


# ----------------------------------------------------------
# Optional helper: load non-interpolated phys files
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
            raise ValueError(f"{f.name} missing 'daqtime' and/or 'video'")
        subject_dfs[sid] = df.sort_values("daqtime").reset_index(drop=True)

    return subject_dfs


def build_video_segments_from_phys(df_subj_phys: pd.DataFrame, vids_used: np.ndarray) -> Tuple[List[Tuple[int, float, float]], Dict[int, Tuple[int, int, int]]]:
    """
    Returns:
      video_segments_cum : list of (vid, start_s, end_s) in cumulative time
      video_info         : vid -> (v_start_ms, v_end_ms, v_dur_ms)
    """
    video_info: Dict[int, Tuple[int, int, int]] = {}
    for v in np.unique(vids_used):
        d = df_subj_phys[df_subj_phys["video"] == v]
        if d.empty:
            continue
        v_start = int(d["daqtime"].min())
        v_end = int(d["daqtime"].max())
        video_info[int(v)] = (v_start, v_end, v_end - v_start)

    # order by actual appearance in time
    videos_in_order = sorted(video_info.keys(), key=lambda vv: video_info[vv][0])

    cum_ms = 0
    video_segments_cum: List[Tuple[int, float, float]] = []
    for v in videos_in_order:
        _, _, v_dur = video_info[v]
        s = cum_ms / 1000.0
        e = (cum_ms + v_dur) / 1000.0
        video_segments_cum.append((v, s, e))
        cum_ms += v_dur

    return video_segments_cum, video_info


# ----------------------------------------------------------
# Core: compute cumulative x-axis
# ----------------------------------------------------------

def build_cumulative_time_axis_from_predictions(df_sub: pd.DataFrame) -> Tuple[np.ndarray, List[Tuple[int, float, float]], Dict[int, Tuple[float, float, float]]]:
    """
    Fallback, wenn keine phys dir vorhanden:
    - pro video: v_start = min(start_s), v_end = max(end_s)
    - order videos by v_start
    - cumulative offsets by duration (v_end - v_start)
    - x_time_s per row uses row end_s mapped into cumulative video segment
    Returns:
      x_time_s, video_segments_cum, video_info_s(vid->(v_start_s,v_end_s,v_dur_s))
    """
    if "start_s" not in df_sub.columns or "end_s" not in df_sub.columns:
        # last resort: index-based timeline
        x = np.arange(len(df_sub), dtype=float)
        vids_used = df_sub["video"].to_numpy()
        # Fake segments: one per unique vid in appearance order
        vids_order = []
        for v in vids_used:
            if v not in vids_order:
                vids_order.append(v)
        segs = []
        cur = 0.0
        info = {}
        for v in vids_order:
            n = float(np.sum(vids_used == v))
            segs.append((int(v), cur, cur + n))
            info[int(v)] = (cur, cur + n, n)
            cur += n
        return x, segs, info

    # ensure numeric
    df_sub = df_sub.copy()
    df_sub["start_s"] = pd.to_numeric(df_sub["start_s"], errors="coerce")
    df_sub["end_s"] = pd.to_numeric(df_sub["end_s"], errors="coerce")

    video_info: Dict[int, Tuple[float, float, float]] = {}
    for v, g in df_sub.groupby("video", sort=False):
        v_start = float(np.nanmin(g["start_s"].to_numpy()))
        v_end = float(np.nanmax(g["end_s"].to_numpy()))
        v_dur = max(1e-6, v_end - v_start)
        video_info[int(v)] = (v_start, v_end, v_dur)

    # order by v_start
    videos_in_order = sorted(video_info.keys(), key=lambda vv: video_info[vv][0])

    offset_s: Dict[int, float] = {}
    cum_s = 0.0
    video_segments_cum: List[Tuple[int, float, float]] = []
    for v in videos_in_order:
        v_start, v_end, v_dur = video_info[v]
        offset_s[v] = cum_s
        video_segments_cum.append((v, cum_s, cum_s + v_dur))
        cum_s += v_dur

    # compute x positions using end_s within video
    vids = df_sub["video"].to_numpy(dtype=int)
    t_end = df_sub["end_s"].to_numpy(dtype=float)

    x_time_s = np.zeros(len(df_sub), dtype=float)
    for i in range(len(df_sub)):
        v = int(vids[i])
        v_start, _, _ = video_info[v]
        rel = t_end[i] - v_start
        x_time_s[i] = offset_s[v] + rel

    return x_time_s, video_segments_cum, video_info


def build_cumulative_time_axis_with_phys(
    df_sub_pred: pd.DataFrame,
    df_sub_phys: pd.DataFrame
) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
    """
    Use phys daqtime to:
    - determine true video ordering + durations
    - map prediction end_s into cumulative axis.
    Handles two cases:
      A) end_s is relative-to-video  -> end_ms should be <= video duration
      B) end_s is global/daqtime-sec -> end_ms is around daqtime; we subtract video start
    """
    vids = df_sub_pred["video"].to_numpy(dtype=int)
    video_segments_cum, video_info_ms = build_video_segments_from_phys(df_sub_phys, vids)

    # offsets in ms for each video in cumulative timeline
    offset_ms: Dict[int, int] = {}
    cum_ms = 0
    for v, _, _ in video_segments_cum:
        offset_ms[int(v)] = cum_ms
        cum_ms += video_info_ms[int(v)][2]

    end_s = pd.to_numeric(df_sub_pred.get("end_s", pd.Series([np.nan]*len(df_sub_pred))), errors="coerce").to_numpy(dtype=float)
    end_ms = np.where(np.isfinite(end_s), end_s * 1000.0, np.nan)

    # decide per-video whether end_s is relative or global
    per_video_mode: Dict[int, str] = {}
    for v in np.unique(vids):
        v = int(v)
        m = vids == v
        if not np.any(m):
            continue
        v_dur = float(video_info_ms[v][2])
        # if most end_ms are within duration -> "relative"
        within = np.isfinite(end_ms[m]) & (end_ms[m] <= 1.10 * v_dur)
        ratio = float(within.mean()) if within.size else 0.0
        per_video_mode[v] = "relative" if ratio >= 0.8 else "global"

    x_time_s = np.zeros(len(df_sub_pred), dtype=float)
    for i in range(len(df_sub_pred)):
        v = int(vids[i])
        v_start_ms, _, v_dur_ms = video_info_ms[v]
        if not np.isfinite(end_ms[i]):
            x_time_s[i] = float(i)
            continue

        if per_video_mode.get(v, "relative") == "relative":
            rel_ms = end_ms[i]
        else:
            rel_ms = end_ms[i] - float(v_start_ms)

        # clamp a bit to avoid crazy values
        rel_ms = float(np.clip(rel_ms, 0.0, float(v_dur_ms)))
        x_time_s[i] = (offset_ms[v] + rel_ms) / 1000.0

    return x_time_s, video_segments_cum


# ----------------------------------------------------------
# Plot (same structure as your CNN plot)
# ----------------------------------------------------------

def plot_single_subject(
    subject_id: int,
    cfg: PlotCfg,
    df_sub: pd.DataFrame,
    x_time_s: np.ndarray,
    video_segments_cum: List[Tuple[int, float, float]],
    fig_path: Path
):
    exp_pos = cfg.exp_pos
    exp_neg = cfg.exp_neg

    # True/pred as 0/1
    true_lbl = df_sub["true_label"].astype(str).to_numpy()
    pred_lbl = df_sub["pred_label"].astype(str).to_numpy()

    ys = np.where(true_lbl == exp_pos, 1, 0)
    preds = np.where(pred_lbl == exp_pos, 1, 0)

    vids = df_sub["video"].to_numpy(dtype=int)

    # Aggregated probabilities as running class-frequency per video (exactly like your CNN code)
    agg_p_neg = np.zeros(len(df_sub), dtype=float)
    agg_p_pos = np.zeros(len(df_sub), dtype=float)

    count_total: Dict[int, int] = {}
    count_neg: Dict[int, int] = {}
    count_pos: Dict[int, int] = {}

    for i, v in enumerate(vids):
        if v not in count_total:
            count_total[v] = 0
            count_neg[v] = 0
            count_pos[v] = 0

        count_total[v] += 1
        if preds[i] == 0:
            count_neg[v] += 1
        else:
            count_pos[v] += 1

        agg_p_neg[i] = count_neg[v] / count_total[v]
        agg_p_pos[i] = count_pos[v] / count_total[v]

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

    vids_arr = np.array(vids)
    first = True
    for vid, start_s, _ in video_segments_cum:
        m = vids_arr == int(vid)
        if not np.any(m):
            continue

        xs = np.concatenate([[start_s], x_time_s[m]])
        pneg = np.concatenate([[0.5], agg_p_neg[m]])
        ppos = np.concatenate([[0.5], agg_p_pos[m]])

        ax2.plot(xs, pneg, marker=".", color="tab:blue", label=f"P({exp_neg})" if first else None)
        ax2.plot(xs, ppos, marker=".", color="tab:orange", label=f"P({exp_pos})" if first else None)
        first = False

    for _, _, e in video_segments_cum:
        ax2.axvline(e, linestyle="--", alpha=0.3)

    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


# ----------------------------------------------------------
# Main logic
# ----------------------------------------------------------

def plot_subjects(cfg: PlotCfg, subjects: List[int]):
    pred_path = cfg.mlp_out_dir / "cv_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path}")

    df = pd.read_csv(pred_path)

    # filter fold
    if "fold" in df.columns:
        df = df[df["fold"] == cfg.fold].copy()

    # ensure only the two classes for the experiment (safety)
    df["true_label"] = df["true_label"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str)
    keep = {cfg.exp_pos, cfg.exp_neg}
    df = df[df["true_label"].isin(keep)].copy()

    if df.empty:
        raise RuntimeError(f"No rows left after filtering fold={cfg.fold} and labels={keep}")

    # optional phys loading
    subject_phys = None
    if cfg.noninterp_phys_dir is not None and cfg.noninterp_phys_dir.exists():
        subject_phys = load_subject_phys_files(cfg.noninterp_phys_dir)

    out_dir = cfg.out_root / f"{cfg.exp_name}_fold{cfg.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subjects:
        df_sub = df[df["subject"] == subject_id].copy()
        if df_sub.empty:
            print(f"[skip] subject {subject_id}: no windows in fold {cfg.fold}")
            continue

        # sort chronologically (best effort)
        if "start_s" in df_sub.columns:
            df_sub["start_s"] = pd.to_numeric(df_sub["start_s"], errors="coerce")
            df_sub = df_sub.sort_values(["video", "start_s"], kind="mergesort")
        else:
            df_sub = df_sub.sort_index()

        # build cumulative axis
        if subject_phys is not None and subject_id in subject_phys:
            x_time_s, video_segments_cum = build_cumulative_time_axis_with_phys(df_sub, subject_phys[subject_id])
        else:
            x_time_s, video_segments_cum, _ = build_cumulative_time_axis_from_predictions(df_sub)

            # If ordering by start_s is global, this still looks fine;
            # If start_s resets per video, it will still be correct per video segment (just ordering might be off).
            # video_segments_cum already reflects our best guess.

        # After computing x_time_s, sort by x_time_s to match "timeline"
        order = np.argsort(x_time_s)
        df_sub = df_sub.iloc[order].reset_index(drop=True)
        x_time_s = x_time_s[order]

        fig_path = out_dir / f"subject{subject_id}.png"
        plot_single_subject(subject_id, cfg, df_sub, x_time_s, video_segments_cum, fig_path)
        print(f"Saved: {fig_path}")


def main():
    cfg = PlotCfg(
        exp_name="scary_vs_bored",
        fold=3,
        mlp_out_dir=Path("../outputs_20w10s_whitelist_hyperopt/scary_vs_bored"),
        out_root=Path("figs/mlp_scary_vs_bored_plots"),
        # optional (set this if you want the phys-based ordering like in your CNN plot)
        noninterp_phys_dir=None,  # e.g. Path("../../case_dataset-master/data/non-interpolated/physiological")
        exp_pos="scary",
        exp_neg="bored",
    )

    # choose subjects (ideally val-subjects of that fold)
    plot_subjects(cfg, subjects=[3, 8, 13, 18, 23, 28])


if __name__ == "__main__":
    main()