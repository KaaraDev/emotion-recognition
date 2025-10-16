# -*- coding: utf-8 -*-
"""
CASE Dataset — Valence & Arousal pro Video übereinandergelegt (über alle Subjekte)
Spalten: daqtime,ecg,bvp,gsr,rsp,skt,emg_zygo,emg_coru,emg_trap,video
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import wrap

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
PATH_ANN = Path("../case_dataset-master/data/interpolated/annotations")
SUBJECTS = list(range(1, 30 + 1))  # 1..30
N_POINTS = 600  # Resample-Punkte je Segment
SHOW_INDIVIDUAL_CURVES = False
SAVE_FIG = True
FIG_OUT = Path("va_overlay_by_video.png")
DPI = 200

VIDEO_MAP = {
    1: "Amusement 1",
    2: "Amusement 2",
    3: "Boredom 1",
    4: "Boredom 2",
    5: "Relaxation 1",
    6: "Relaxation 2",
    7: "Scary 1",
    8: "Scary 2",
}
COLOR_BY_CATEGORY = {
    "Amusement": "#1f77b4",
    "Boredom": "#ff7f0e",
    "Relaxation": "#2ca02c",
    "Scary": "#d62728",
}
LINESTYLE_BY_SUFFIX = {"1": "-", "2": "--"}

PLOT_TITLE = "Valence & Arousal — averaged per video across subjects (±1 SD)"
XLABEL = "Normalized time (0–1)"
YLABEL_V = "Valence"
YLABEL_A = "Arousal"


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def category_of(video_name: str) -> str:
    return video_name.rsplit(" ", 1)[0]


def suffix_of(video_name: str) -> str:
    return video_name.rsplit(" ", 1)[1]


def normalize_segment(y: np.ndarray, n_points: int) -> np.ndarray:
    if y.size == 0:
        return np.array([])
    x_old = np.linspace(0.0, 1.0, y.size)
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, y)


# ------------------------------------------------------------
# Daten sammeln: pro Video → Liste aus (Valence-Kurve, Arousal-Kurve)
# ------------------------------------------------------------
data_by_video = {VIDEO_MAP[vid_id]: {"V": [], "A": []} for vid_id in VIDEO_MAP}

n_files = 0
for sid in SUBJECTS:
    f = PATH_ANN / f"sub_{sid}.csv"
    if not f.exists():
        continue
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"[WARN] Konnte sub_{sid}.csv nicht laden: {e}")
        continue

    # Erwartete Spalten exakt:
    missing = [c for c in ["jstime", "valence", "arousal", "video"] if c not in df.columns]
    if missing:
        print(f"[WARN] sub_{sid}.csv: fehlende Spalten {missing} – übersprungen.")
        continue

    n_files += 1
    # Nur relevante Videos 1..8 (Start/Pause/Ende ignorieren)
    df = df[df["video"].isin(VIDEO_MAP.keys())]

    for vid_id, vid_name in VIDEO_MAP.items():
        seg = df[df["video"] == vid_id]
        if seg.empty:
            continue
        v = seg["valence"].to_numpy()
        a = seg["arousal"].to_numpy()

        v_n = normalize_segment(v, N_POINTS)
        a_n = normalize_segment(a, N_POINTS)
        if v_n.size == 0 or a_n.size == 0:
            continue

        data_by_video[vid_name]["V"].append(v_n)
        data_by_video[vid_name]["A"].append(a_n)

print(f"[INFO] Geladene Subjektdateien: {n_files}")
for vname, d in data_by_video.items():
    print(f"[INFO] {vname}: N={len(d['V'])}")

# ------------------------------------------------------------
# Plotten
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
t = np.linspace(0.0, 1.0, N_POINTS)

axes[0].set_title("\n".join(wrap(PLOT_TITLE, 90)), pad=10)
axes[0].set_ylabel(YLABEL_V)
axes[1].set_ylabel(YLABEL_A)
axes[1].set_xlabel(XLABEL)

for vid_name, series in data_by_video.items():
    v_curves = series["V"]
    a_curves = series["A"]
    if len(v_curves) == 0 or len(a_curves) == 0:
        continue

    v_arr = np.vstack(v_curves)  # (n_segments, N_POINTS)
    a_arr = np.vstack(a_curves)

    v_mean, v_std = v_arr.mean(axis=0), v_arr.std(axis=0)
    a_mean, a_std = a_arr.mean(axis=0), a_arr.std(axis=0)

    cat = category_of(vid_name)
    suf = suffix_of(vid_name)
    color = COLOR_BY_CATEGORY.get(cat, None)
    ls = LINESTYLE_BY_SUFFIX.get(suf, "-")

    if SHOW_INDIVIDUAL_CURVES:
        idx = np.arange(v_arr.shape[0])
        if v_arr.shape[0] > 10:
            rng = np.random.default_rng(123)
            idx = np.sort(rng.choice(idx, size=10, replace=False))
        for i in idx:
            axes[0].plot(t, v_arr[i], lw=0.6, alpha=0.25, color=color)
            axes[1].plot(t, a_arr[i], lw=0.6, alpha=0.25, color=color)

    axes[0].plot(t, v_mean, ls=ls, lw=2.2, color=color, label=vid_name)
    axes[0].fill_between(t, v_mean - v_std, v_mean + v_std, alpha=0.15, color=color)
    axes[1].plot(t, a_mean, ls=ls, lw=2.2, color=color, label=vid_name)
    axes[1].fill_between(t, a_mean - a_std, a_mean + a_std, alpha=0.15, color=color)

for ax in axes:
    ax.grid(True, alpha=0.25)
    ax.margins(x=0)

axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), title="Videos", frameon=False)
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Videos", frameon=False)

plt.tight_layout(rect=[0, 0, 0.82, 1])

if SAVE_FIG:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=DPI)
    print(f"[INFO] Abbildung gespeichert unter: {FIG_OUT.resolve()}")

plt.show()
