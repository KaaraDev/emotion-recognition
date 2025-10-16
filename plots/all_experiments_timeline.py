# -*- coding: utf-8 -*-
"""
CASE Dataset — Valence & Arousal: nur Individualkurven, Videos hintereinander (seriell)
Quelle: interpolated/annotations
Erwartete Spalten: jstime, valence, arousal, video
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
PATH_ANN = Path("../case_dataset-master/data/interpolated/annotations")
SUBJECTS = list(range(1, 30 + 1))  # 1..30
N_POINTS = 600  # gleichlange Segmente (normierte Zeit 0..1)
GAP_POINTS = 20  # Lücke zwischen Videos
MAX_CURVES_PER_VIDEO = None  # z.B. 120, um die Menge zu begrenzen; None = alle
Z_SCORE_PER_SEGMENT = False  # True: pro Segment z-standardisieren (separat für V und A)
CLIP_TO_UNIT = False  # True: Werte auf [-1, 1] clippen (typisch für V/A-Skala)

SAVE_FIG = True
FIG_OUT = Path("va_individual_serial_by_video.png")
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
PLOT_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]  # feste Reihenfolge

COLOR_BY_CATEGORY = {
    "Amusement": "#1f77b4",
    "Boredom": "#ff7f0e",
    "Relaxation": "#2ca02c",
    "Scary": "#d62728",
}

TITLE = "Valence & Arousal — individual curves only; videos in series (0–1 per segment)"
XLABEL = "Normalized time (segment blocks concatenated)"
YLABEL_V = "Valence (0.5…9.5)"
YLABEL_A = "Arousal (0.5…9.5)"

ALPHA = 0.18  # Transparenz der Individualkurven
LW = 0.8  # Liniendicke


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


def maybe_clip_unit(y: np.ndarray) -> np.ndarray:
    if not CLIP_TO_UNIT:
        return y
    return np.clip(y, -1.0, 1.0)


# ------------------------------------------------------------
# Daten sammeln: pro Video → Listen aus V- und A-Kurven (normierte Länge N_POINTS)
# ------------------------------------------------------------
dataV_by_video = {vid_id: [] for vid_id in VIDEO_MAP}
dataA_by_video = {vid_id: [] for vid_id in VIDEO_MAP}

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

    # Pflichtspalten
    missing = [c for c in ["jstime", "valence", "arousal", "video"] if c not in df.columns]
    if missing:
        print(f"[WARN] sub_{sid}.csv: fehlende Spalten {missing} – übersprungen.")
        continue

    n_files += 1
    df = df[df["video"].isin(VIDEO_MAP.keys())]

    for vid_id in VIDEO_MAP:
        seg_v = df[df["video"] == vid_id]["valence"].to_numpy()
        seg_a = df[df["video"] == vid_id]["arousal"].to_numpy()
        if seg_v.size == 0 or seg_a.size == 0:
            continue

        v_n = normalize_segment(seg_v, N_POINTS)
        a_n = normalize_segment(seg_a, N_POINTS)
        if v_n.size == 0 or a_n.size == 0:
            continue

        # optional z-Score je Segment separat für V/A
        v_n = maybe_zscore(v_n)
        a_n = maybe_zscore(a_n)

        # optional auf [-1,1] clippen (nur sinnvoll, wenn nicht z-gescored)
        if not Z_SCORE_PER_SEGMENT:
            v_n = maybe_clip_unit(v_n)
            a_n = maybe_clip_unit(a_n)

        dataV_by_video[vid_id].append(v_n)
        dataA_by_video[vid_id].append(a_n)

print(f"[INFO] Geladene Subjektdateien: {n_files}")
for vid_id in PLOT_ORDER:
    print(f"[INFO] {VIDEO_MAP[vid_id]}: N(V)={len(dataV_by_video[vid_id])}, N(A)={len(dataA_by_video[vid_id])}")

# ------------------------------------------------------------
# Plotten: Videos seriell auf der x-Achse anordnen
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

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axV, axA = axes

# -------- Valence --------
for vid_id in PLOT_ORDER:
    curves = dataV_by_video[vid_id]
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

    x0 = starts[vid_id]
    xs = x[x0: x0 + block_len]
    for y in curves_to_plot:
        axV.plot(xs, y, lw=LW, alpha=ALPHA, color=color)

# -------- Arousal --------
for vid_id in PLOT_ORDER:
    curves = dataA_by_video[vid_id]
    if len(curves) == 0:
        continue
    vid_name = VIDEO_MAP[vid_id]
    cat = category_of(vid_name)
    color = COLOR_BY_CATEGORY.get(cat, "#333333")

    curves_to_plot = curves
    if isinstance(MAX_CURVES_PER_VIDEO, int) and len(curves) > MAX_CURVES_PER_VIDEO:
        rng = np.random.default_rng(123)
        sel = np.sort(rng.choice(np.arange(len(curves)), size=MAX_CURVES_PER_VIDEO, replace=False))
        curves_to_plot = [curves[i] for i in sel]

    x0 = starts[vid_id]
    xs = x[x0: x0 + block_len]
    for y in curves_to_plot:
        axA.plot(xs, y, lw=LW, alpha=ALPHA, color=color)

# Vertikale Trenner & Labels pro Achse
for ax in (axV, axA):
    yt = ax.get_ylim()
    for i, vid_id in enumerate(PLOT_ORDER):
        x0 = starts[vid_id]
        if i > 0:
            ax.axvline(x0 - gap / 2, color="#888888", lw=0.8, alpha=0.6)
        center = x0 + block_len / 2
        ax.text(center, yt[0] + 0.02 * (yt[1] - yt[0]), VIDEO_MAP[vid_id],
                ha="center", va="bottom", fontsize=9, rotation=0)

# Optik / Labels
axV.set_title("\n".join(wrap(TITLE + (" (z-standardisiert)" if Z_SCORE_PER_SEGMENT else ""), 100)), pad=10)
axV.set_ylabel(YLABEL_V)
axA.set_ylabel(YLABEL_A)
axA.set_xlabel(XLABEL)

for ax in (axV, axA):
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xlim(0, total_len - 1)
    ax.margins(x=0)
    if not Z_SCORE_PER_SEGMENT and CLIP_TO_UNIT:
        ax.set_ylim(0.5, 10)

# Kategoriefarben-Legende (einmal rechts außen)
handles = []
cats_seen = set()
for vid_id in PLOT_ORDER:
    name = VIDEO_MAP[vid_id]
    cat = category_of(name)
    if cat in cats_seen:
        continue
    cats_seen.add(cat)
    handles.append(mpatches.Patch(color=COLOR_BY_CATEGORY[cat], label=cat))
axes[0].legend(handles=handles, title="Kategorie", loc="upper left",
               bbox_to_anchor=(1.01, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 0.86, 1])

if SAVE_FIG:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=DPI)
    print(f"[INFO] Abbildung gespeichert unter: {FIG_OUT.resolve()}")

plt.show()
