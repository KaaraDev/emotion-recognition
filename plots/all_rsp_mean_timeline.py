# -*- coding: utf-8 -*-
"""
CASE Dataset — RSP: Mean pro Video (optional: Individualkurven), Videos seriell
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
N_POINTS = 600
GAP_POINTS = 20
MAX_CURVES_PER_VIDEO = None  # z.B. 80; None = alle
Z_SCORE_PER_SEGMENT = False  # pro Segment z-standardisieren
DRAW_INDIVIDUALS = False  # <- auf True setzen, wenn du zusätzlich alle Kurven willst
DRAW_MEAN = True  # Mean pro Video zeichnen
FILL_CI = True  # 95%-Konfidenzintervall um den Mean
SAVE_FIG = True
FIG_OUT = Path("rsp_mean_serial_by_video.png")
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

TITLE = "RSP/Respiration — mean per video (0–1 per segment)"
XLABEL = "Normalized time (segment blocks concatenated)"
YLABEL = "Respiration (a.u.)"
if Z_SCORE_PER_SEGMENT:
    TITLE += " (z-standardisiert pro Segment)"

ALPHA_INDIV = 0.18
LW_INDIV = 0.8
LW_MEAN = 2.2


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
    if s == 0 or np.isnan(s):
        return y - m
    return (y - m) / s


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
    except Exception as e:
        print(f"[WARN] Konnte sub_{sid}.csv nicht laden: {e}")
        continue

    missing = [c for c in ["daqtime", "rsp", "video"] if c not in df.columns]
    if missing:
        print(f"[WARN] sub_{sid}.csv: fehlende Spalten {missing} – übersprungen.")
        continue

    n_files += 1
    df = df[df["video"].isin(VIDEO_MAP.keys())]

    for vid_id in VIDEO_MAP:
        seg = df[df["video"] == vid_id]["rsp"].to_numpy()
        if seg.size == 0:
            continue
        seg_n = normalize_segment(seg, N_POINTS)
        if seg_n.size == 0:
            continue
        seg_n = maybe_zscore(seg_n)
        data_by_video[vid_id].append(seg_n)

print(f"[INFO] Geladene Subjektdateien: {n_files}")
for vid_id in PLOT_ORDER:
    print(f"[INFO] {VIDEO_MAP[vid_id]}: N={len(data_by_video[vid_id])}")

# ------------------------------------------------------------
# X-Layout vorbereiten (Videos seriell)
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
fig, ax = plt.subplots(1, 1, figsize=(13, 5))

# ------------------------------------------------------------
# Plotten
# ------------------------------------------------------------
mean_handles = []
cats_seen = set()

for vid_id in PLOT_ORDER:
    curves = data_by_video[vid_id]
    if len(curves) == 0:
        continue

    vid_name = VIDEO_MAP[vid_id]
    cat = category_of(vid_name)
    color = COLOR_BY_CATEGORY.get(cat, "#333333")

    # ggf. Anzahl Kurven deckeln
    curves_to_plot = curves
    if isinstance(MAX_CURVES_PER_VIDEO, int) and len(curves) > MAX_CURVES_PER_VIDEO:
        rng = np.random.default_rng(123)
        sel = np.sort(rng.choice(np.arange(len(curves)), size=MAX_CURVES_PER_VIDEO, replace=False))
        curves_to_plot = [curves[i] for i in sel]

    # X-Position für diesen Block
    x0 = starts[vid_id]
    xs = x[x0: x0 + block_len]

    # Optional: Individualkurven
    if DRAW_INDIVIDUALS:
        for y in curves_to_plot:
            ax.plot(xs, y, lw=LW_INDIV, alpha=ALPHA_INDIV, color=color)

    # Mean + 95%-CI
    if DRAW_MEAN:
        M = np.vstack(curves_to_plot)  # [n_curves, N_POINTS]
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        n = M.shape[0]
        sem = std / np.sqrt(max(n, 1))
        ci95 = 1.96 * sem

        ax.plot(xs, mean, lw=LW_MEAN, alpha=1.0, color=color)

        if FILL_CI and n > 1:
            ax.fill_between(xs, mean - ci95, mean + ci95, alpha=0.18, color=color, linewidth=0)

        # Für Legende: pro Kategorie genau einen Eintrag
        if cat not in cats_seen:
            cats_seen.add(cat)
            mean_handles.append(mpatches.Patch(color=color, label=f"{cat}"))

# Vertikale Trenner & Labels
yt = ax.get_ylim()
for i, vid_id in enumerate(PLOT_ORDER):
    x0 = starts[vid_id]
    if i > 0:
        ax.axvline(x0 - gap / 2, color="#888888", lw=0.8, alpha=0.6)
    center = x0 + block_len / 2
    ax.text(center, yt[0] + 0.02 * (yt[1] - yt[0]), VIDEO_MAP[vid_id],
            ha="center", va="bottom", fontsize=9)

# Optik
ax.set_title("\n".join(wrap(TITLE, 100)), pad=10)
ax.set_xlabel(XLABEL)
ax.set_ylabel(YLABEL)
ax.grid(True, axis="y", alpha=0.25)
ax.set_xlim(0, total_len - 1)
ax.margins(x=0)

# Legende
if DRAW_MEAN:
    leg_title = "Kategorie (Mean" + (", 95% CI" if FILL_CI else "") + ")"
    ax.legend(handles=mean_handles, title=leg_title, loc="upper left",
              bbox_to_anchor=(1.01, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 0.86, 1])

if SAVE_FIG:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=DPI)
    print(f"[INFO] Abbildung gespeichert unter: {FIG_OUT.resolve()}")

plt.show()
