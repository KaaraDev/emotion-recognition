# -*- coding: utf-8 -*-
"""
CASE Dataset — GSR/EDA pro Video übereinandergelegt (über alle Subjekte)
Erwartete Spalten (physiological/interpolated):
daqtime, ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap, video
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import wrap

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
PATH_PHYS = Path("../case_dataset-master/data/interpolated/physiological")
SUBJECTS = list(range(1, 30 + 1))  # 1..30
N_POINTS = 600  # Resample-Punkte je Segment (normierte Zeitachse 0..1)
SHOW_INDIVIDUAL_CURVES = True
MAX_INDIV_CURVES = 10  # max. Anzahl Einzelkurven pro Video (wenn SHOW_INDIVIDUAL_CURVES=True)
SAVE_FIG = True
FIG_OUT = Path("eda_overlay_by_video.png")
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

PLOT_TITLE = "EDA/GSR — averaged per video across subjects (±1 SD)"
XLABEL = "Normalized time (0–1)"
YLABEL = "EDA (µS)"


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def category_of(video_name: str) -> str:
    return video_name.rsplit(" ", 1)[0]


def suffix_of(video_name: str) -> str:
    return video_name.rsplit(" ", 1)[1]


def normalize_segment(y: np.ndarray, n_points: int) -> np.ndarray:
    """Inter/Extrapoliert ein Segment auf exakt n_points entlang normierter Zeit 0..1."""
    if y.size == 0:
        return np.array([])
    x_old = np.linspace(0.0, 1.0, y.size)
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, y)


# ------------------------------------------------------------
# Daten sammeln: pro Video → Liste aus EDA-Kurven (normierte Länge N_POINTS)
# ------------------------------------------------------------
data_by_video = {VIDEO_MAP[vid_id]: [] for vid_id in VIDEO_MAP}

n_files = 0
for sid in SUBJECTS:
    f = PATH_PHYS / f"sub_{sid}.csv"
    if not f.exists():
        continue
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"[WARN] Konnte sub_{sid}.csv nicht laden: {e}")
        continue

    # Erwartete Spalten:
    missing = [c for c in ["daqtime", "gsr", "video"] if c not in df.columns]
    if missing:
        print(f"[WARN] sub_{sid}.csv: fehlende Spalten {missing} – übersprungen.")
        continue

    n_files += 1

    # nur Videos 1..8 berücksichtigen (Start/Pause/Ende ignorieren)
    df = df[df["video"].isin(VIDEO_MAP.keys())]

    # pro Video -> EDA-Segment einsammeln
    for vid_id, vid_name in VIDEO_MAP.items():
        seg = df[df["video"] == vid_id]["gsr"].to_numpy()
        if seg.size == 0:
            continue
        seg_n = normalize_segment(seg, N_POINTS)
        if seg_n.size == 0:
            continue
        data_by_video[vid_name].append(seg_n)

print(f"[INFO] Geladene Subjektdateien: {n_files}")
for vname, curves in data_by_video.items():
    print(f"[INFO] {vname}: N={len(curves)}")

# ------------------------------------------------------------
# Plotten
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
t = np.linspace(0.0, 1.0, N_POINTS)

ax.set_title("\n".join(wrap(PLOT_TITLE, 90)), pad=10)
ax.set_ylabel(YLABEL)
ax.set_xlabel(XLABEL)

for vid_name, curves in data_by_video.items():
    if len(curves) == 0:
        continue

    arr = np.vstack(curves)  # (n_segments, N_POINTS)
    mean, std = arr.mean(axis=0), arr.std(axis=0)

    cat = category_of(vid_name)
    suf = suffix_of(vid_name)
    color = COLOR_BY_CATEGORY.get(cat, None)
    ls = LINESTYLE_BY_SUFFIX.get(suf, "-")

    # optionale Einzelkurven (zufällige Auswahl, um Plot nicht zu überladen)
    if SHOW_INDIVIDUAL_CURVES:
        idx = np.arange(arr.shape[0])
        if arr.shape[0] > MAX_INDIV_CURVES:
            rng = np.random.default_rng(123)
            idx = np.sort(rng.choice(idx, size=MAX_INDIV_CURVES, replace=False))
        for i in idx:
            ax.plot(t, arr[i], lw=0.6, alpha=0.25, color=color)

    ax.plot(t, mean, ls=ls, lw=2.2, color=color, label=vid_name)
    ax.fill_between(t, mean - std, mean + std, alpha=0.15, color=color)

ax.grid(True, alpha=0.25)
ax.margins(x=0)

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), title="Videos", frameon=False)

plt.tight_layout(rect=[0, 0, 0.82, 1])

if SAVE_FIG:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=DPI)
    print(f"[INFO] Abbildung gespeichert unter: {FIG_OUT.resolve()}")

plt.show()
