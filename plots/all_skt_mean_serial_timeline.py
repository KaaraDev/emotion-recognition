# -*- coding: utf-8 -*-
"""
CASE Dataset — SKT (Skin Temperature)
- Mittelwert ± 1 SD pro Video
- Videos seriell hintereinander auf der x-Achse
- Zeit-normalisiert (0–1 pro Segment), keine Amplituden-Normierung
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import wrap
import matplotlib.patches as mpatches

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
PATH_PHYS = Path("../case_dataset-master/data/interpolated/physiological")
SUBJECTS = list(range(1, 30 + 1))

N_POINTS = 600
GAP_POINTS = 20

SHOW_INDIVIDUAL_CURVES = True
MAX_INDIV_CURVES = 8

SAVE_FIG = True
FIG_OUT = Path("skt_mean_serial_by_video.png")
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
PLOT_ORDER = list(VIDEO_MAP.keys())

COLOR_BY_CATEGORY = {
    "Amusement": "#1f77b4",
    "Boredom": "#ff7f0e",
    "Relaxation": "#2ca02c",
    "Scary": "#d62728",
}

TITLE = "Skin Temperature (SKT) — averaged per video (±1 SD), serial layout"
XLABEL = "Normalized time (concatenated video segments)"
YLABEL = "Skin Temperature (°C)"

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def category_of(video_name: str) -> str:
    return video_name.rsplit(" ", 1)[0]


def normalize_segment(y: np.ndarray, n_points: int) -> np.ndarray:
    if y.size == 0:
        return np.array([])
    x_old = np.linspace(0.0, 1.0, y.size)
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, y)


# ------------------------------------------------------------
# Daten sammeln
# ------------------------------------------------------------
data_by_video = {vid_id: [] for vid_id in VIDEO_MAP}

n_files = 0
for sid in SUBJECTS:
    f = PATH_PHYS / f"sub_{sid}.csv"
    if not f.exists():
        continue

    try:
        df = pd.read_csv(f)
    except Exception:
        continue

    if not {"daqtime", "skt", "video"}.issubset(df.columns):
        continue

    n_files += 1
    df = df[df["video"].isin(VIDEO_MAP.keys())]

    for vid_id in VIDEO_MAP:
        seg = df[df["video"] == vid_id]["skt"].to_numpy()
        if seg.size == 0:
            continue
        seg_n = normalize_segment(seg, N_POINTS)
        if seg_n.size > 0:
            data_by_video[vid_id].append(seg_n)

print(f"[INFO] Geladene Subjekte: {n_files}")
for vid_id in PLOT_ORDER:
    print(f"[INFO] {VIDEO_MAP[vid_id]}: N={len(data_by_video[vid_id])}")

# ------------------------------------------------------------
# X-Achse seriell aufbauen
# ------------------------------------------------------------
n_blocks = len(PLOT_ORDER)
block_len = N_POINTS
gap = GAP_POINTS
total_len = n_blocks * block_len + (n_blocks - 1) * gap

starts = {}
pos = 0
for i, vid_id in enumerate(PLOT_ORDER):
    starts[vid_id] = pos
    pos += block_len
    if i < n_blocks - 1:
        pos += gap

x_full = np.arange(total_len)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))

for vid_id in PLOT_ORDER:
    curves = data_by_video[vid_id]
    if len(curves) == 0:
        continue

    arr = np.vstack(curves)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    vid_name = VIDEO_MAP[vid_id]
    cat = category_of(vid_name)
    color = COLOR_BY_CATEGORY.get(cat, "#333333")

    x0 = starts[vid_id]
    xs = x_full[x0 : x0 + block_len]

    # optionale Einzelkurven
    if SHOW_INDIVIDUAL_CURVES:
        idx = np.arange(arr.shape[0])
        if arr.shape[0] > MAX_INDIV_CURVES:
            rng = np.random.default_rng(123)
            idx = rng.choice(idx, MAX_INDIV_CURVES, replace=False)
        for i in idx:
            ax.plot(xs, arr[i], lw=0.6, alpha=0.25, color=color)

    ax.plot(xs, mean, lw=2.2, color=color)
    ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=color)

# Trenner & Labels
yt = ax.get_ylim()
for i, vid_id in enumerate(PLOT_ORDER):
    x0 = starts[vid_id]
    if i > 0:
        ax.axvline(x0 - gap / 2, color="#888", lw=0.8, alpha=0.6)
    center = x0 + block_len / 2
    ax.text(center, yt[0] + 0.02 * (yt[1] - yt[0]),
            VIDEO_MAP[vid_id], ha="center", va="bottom", fontsize=9)

# Optik
ax.set_title("\n".join(wrap(TITLE, 100)), pad=10)
ax.set_xlabel(XLABEL)
ax.set_ylabel(YLABEL)
ax.grid(True, axis="y", alpha=0.25)
ax.set_xlim(0, total_len - 1)
ax.margins(x=0)

# Legende
handles, cats_seen = [], set()
for vid_id in PLOT_ORDER:
    cat = category_of(VIDEO_MAP[vid_id])
    if cat in cats_seen:
        continue
    cats_seen.add(cat)
    handles.append(mpatches.Patch(color=COLOR_BY_CATEGORY[cat], label=cat))
ax.legend(handles=handles, title="Kategorie",
          loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 0.86, 1])

if SAVE_FIG:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=DPI)
    print(f"[INFO] Gespeichert unter: {FIG_OUT.resolve()}")

plt.show()
