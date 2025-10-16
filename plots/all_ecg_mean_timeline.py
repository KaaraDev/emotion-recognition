# -*- coding: utf-8 -*-
"""
CASE Dataset — ECG: gemittelte Kurven pro Video (über alle Subjekte)
- Pro Video: Mittelwert ± Standardabweichung der Individualkurven
- Videos werden nacheinander auf der x-Achse gesetzt
Erwartete Spalten in physiological/interpolated:
daqtime, ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap, video
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
SUBJECTS = list(range(1, 30 + 1))  # 1..30
N_POINTS = 600  # normierte Segmentlänge
GAP_POINTS = 20  # Lücke zwischen Segmentblöcken
Z_SCORE_PER_SEGMENT = False  # optional: z-Standardisierung pro Segment
SAVE_FIG = True
FIG_OUT = Path("ecg_mean_serial_by_video.png")
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
PLOT_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]

COLOR_BY_CATEGORY = {
    "Amusement": "#1f77b4",
    "Boredom": "#ff7f0e",
    "Relaxation": "#2ca02c",
    "Scary": "#d62728",
}

TITLE = "ECG — Mean ± SD; videos in series (0–1 per segment)"
XLABEL = "Normalized time (segment blocks concatenated)"
YLABEL = "ECG (a.u.)"

ALPHA_FILL = 0.25  # Transparenz der SD-Fläche
LW_MEAN = 2.0  # Liniendicke der Mittelwertkurve


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


def maybe_zscore(y: np.ndarray) -> np.ndarray:
    if not Z_SCORE_PER_SEGMENT:
        return y
    m = np.nanmean(y)
    s = np.nanstd(y)
    return (y - m) / s if s > 0 and not np.isnan(s) else y - m


# ------------------------------------------------------------
# Daten sammeln: pro Video -> Liste normierter ECG-Segmente
# ------------------------------------------------------------
data_by_video = {vid_id: [] for vid_id in VIDEO_MAP}

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

    # Spaltencheck
    if not {"daqtime", "ecg", "video"}.issubset(df.columns):
        print(f"[WARN] sub_{sid}.csv: fehlende Spalten – übersprungen.")
        continue

    n_files += 1
    df = df[df["video"].isin(VIDEO_MAP.keys())]

    for vid_id in VIDEO_MAP:
        seg = df[df["video"] == vid_id]["ecg"].to_numpy()
        if seg.size == 0:
            continue
        seg_n = normalize_segment(seg, N_POINTS)
        seg_n = maybe_zscore(seg_n)
        data_by_video[vid_id].append(seg_n)

print(f"[INFO] Geladene Subjektdateien: {n_files}")
for vid_id in PLOT_ORDER:
    print(f"[INFO] {VIDEO_MAP[vid_id]}: N={len(data_by_video[vid_id])}")

# ------------------------------------------------------------
# Mittelwert & SD berechnen
# ------------------------------------------------------------
means = {}
sds = {}
for vid_id in VIDEO_MAP:
    curves = np.array(data_by_video[vid_id])
    if curves.size == 0:
        continue
    means[vid_id] = np.nanmean(curves, axis=0)
    sds[vid_id] = np.nanstd(curves, axis=0)

# ------------------------------------------------------------
# Plotten
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

x = np.arange(total_len)
fig, ax = plt.subplots(figsize=(13, 5))

for vid_id in PLOT_ORDER:
    if vid_id not in means:
        continue
    mean = means[vid_id]
    sd = sds[vid_id]
    x0 = starts[vid_id]
    xs = x[x0: x0 + block_len]

    vid_name = VIDEO_MAP[vid_id]
    cat = category_of(vid_name)
    color = COLOR_BY_CATEGORY.get(cat, "#333333")

    ax.plot(xs, mean, lw=LW_MEAN, color=color)
    ax.fill_between(xs, mean - sd, mean + sd, color=color, alpha=ALPHA_FILL)

# Trenner & Labels
yt = ax.get_ylim()
for i, vid_id in enumerate(PLOT_ORDER):
    x0 = starts[vid_id]
    if i > 0:
        ax.axvline(x0 - gap / 2, color="#888888", lw=0.8, alpha=0.6)
    center = x0 + block_len / 2
    ax.text(center, yt[0] + 0.02 * (yt[1] - yt[0]), VIDEO_MAP[vid_id],
            ha="center", va="bottom", fontsize=9)

ax.set_title("\n".join(wrap(TITLE + (" (z-standardisiert)" if Z_SCORE_PER_SEGMENT else ""), 100)), pad=10)
ax.set_xlabel(XLABEL)
ax.set_ylabel(YLABEL)
ax.grid(True, axis="y", alpha=0.25)
ax.set_xlim(0, total_len - 1)
ax.margins(x=0)

# Legende
handles = [mpatches.Patch(color=c, label=cat) for cat, c in COLOR_BY_CATEGORY.items()]
ax.legend(handles=handles, title="Kategorie", loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 0.86, 1])

if SAVE_FIG:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=DPI)
    print(f"[INFO] Abbildung gespeichert unter: {FIG_OUT.resolve()}")

plt.show()
