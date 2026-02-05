# -*- coding: utf-8 -*-
"""
CASE Dataset — EMG (zygo/coru/trap): Mean pro Video, Videos seriell
- pro Signal: Mean + Band (±SD oder 95% CI) über Subjekte
- Videos hintereinander auf der x-Achse (seriell)
- optional: Individualkurven
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

SIGNALS = ["emg_zygo", "emg_coru", "emg_trap"]  # <- diese drei

N_POINTS = 600
GAP_POINTS = 20
MAX_CURVES_PER_VIDEO = None  # z.B. 80; None = alle

Z_SCORE_PER_SEGMENT = False      # meist OFF lassen
DRAW_INDIVIDUALS = False         # True = zusätzlich Individualkurven
DRAW_MEAN = True

# Band um den Mean:
#   "sd"   = ±1 SD (Streuung zwischen Personen)
#   "ci95" = 95%-Konfidenzintervall (Unsicherheit des Mean)
BAND_MODE = "sd"   # "sd" oder "ci95"
FILL_BAND = True

SAVE_FIG = True
OUT_DIR = Path(".")
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

# Labels je Signal (optional schöner)
YLABEL_BY_SIGNAL = {
    "emg_zygo": "EMG Zygomaticus (a.u.)",
    "emg_coru": "EMG Corrugator (a.u.)",
    "emg_trap": "EMG Trapezius (a.u.)",
}

TITLE_BY_SIGNAL = {
    "emg_zygo": "EMG (Zygomaticus) — mean per video; serial layout",
    "emg_coru": "EMG (Corrugator) — mean per video; serial layout",
    "emg_trap": "EMG (Trapezius) — mean per video; serial layout",
}

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


def compute_band(std: np.ndarray, n: int, mode: str) -> np.ndarray:
    if mode == "ci95":
        sem = std / np.sqrt(max(n, 1))
        return 1.96 * sem
    return std


# ------------------------------------------------------------
# X-Layout (Videos seriell) vorbereiten
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
# Daten einmal laden (CSV nur einmal pro Subject) & pro Signal sammeln
# ------------------------------------------------------------
data_by_signal = {sig: {vid_id: [] for vid_id in VIDEO_MAP} for sig in SIGNALS}

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

    # Video filtern
    if "video" not in df.columns:
        continue
    df = df[df["video"].isin(VIDEO_MAP.keys())]

    # nur Subjekte zählen, die überhaupt eine der Signalspalten haben
    has_any = any(sig in df.columns for sig in SIGNALS)
    if not has_any:
        continue
    n_files += 1

    for sig in SIGNALS:
        if sig not in df.columns:
            continue

        for vid_id in VIDEO_MAP:
            seg = df[df["video"] == vid_id][sig].to_numpy()
            if seg.size == 0:
                continue
            seg_n = normalize_segment(seg, N_POINTS)
            if seg_n.size == 0:
                continue
            seg_n = maybe_zscore(seg_n)
            data_by_signal[sig][vid_id].append(seg_n)

print(f"[INFO] Geladene Subjektdateien (mind. 1 EMG-Signal): {n_files}")
for sig in SIGNALS:
    for vid_id in PLOT_ORDER:
        print(f"[INFO] {sig} | {VIDEO_MAP[vid_id]}: N={len(data_by_signal[sig][vid_id])}")

# ------------------------------------------------------------
# Pro Signal: Plot erstellen
# ------------------------------------------------------------
for sig in SIGNALS:
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))

    legend_handles = []
    cats_seen = set()

    for vid_id in PLOT_ORDER:
        curves = data_by_signal[sig][vid_id]
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
        xs = x_full[x0: x0 + block_len]

        # Optional: Individualkurven
        if DRAW_INDIVIDUALS:
            for y in curves_to_plot:
                ax.plot(xs, y, lw=0.8, alpha=0.18, color=color)

        # Mean + Band
        if DRAW_MEAN:
            M = np.vstack(curves_to_plot)
            mean = np.nanmean(M, axis=0)
            std = np.nanstd(M, axis=0)
            n = M.shape[0]

            ax.plot(xs, mean, lw=2.2, color=color)

            if FILL_BAND and n > 1:
                band = compute_band(std, n, BAND_MODE)
                ax.fill_between(xs, mean - band, mean + band, alpha=0.18, color=color, linewidth=0)

            # Legende: pro Kategorie einmal
            if cat not in cats_seen:
                cats_seen.add(cat)
                legend_handles.append(mpatches.Patch(color=color, label=cat))

    # Trenner
    for i, vid_id in enumerate(PLOT_ORDER):
        if i == 0:
            continue
        x0 = starts[vid_id]
        ax.axvline(x0 - gap / 2, color="#888888", lw=0.8, alpha=0.6)

    # Optik
    title = TITLE_BY_SIGNAL.get(sig, f"{sig} — mean per video; serial layout")
    if Z_SCORE_PER_SEGMENT:
        title += " (z-standardisiert pro Segment)"

    ax.set_title("\n".join(wrap(title, 100)), pad=10)
    ax.set_xlabel("Normalized time (segment blocks concatenated)")
    ax.set_ylabel(YLABEL_BY_SIGNAL.get(sig, f"{sig} (a.u.)"))
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xlim(0, total_len - 1)
    ax.margins(x=0)

    # Labels nach dem Plot (ylim korrekt)
    yt = ax.get_ylim()
    for vid_id in PLOT_ORDER:
        center = starts[vid_id] + block_len / 2
        ax.text(center, yt[0] + 0.02 * (yt[1] - yt[0]), VIDEO_MAP[vid_id],
                ha="center", va="bottom", fontsize=9)

    # Legende
    if legend_handles:
        band_txt = "±1 SD" if BAND_MODE == "sd" else "95% CI"
        ax.legend(handles=legend_handles, title=f"Kategorie (Mean {band_txt})",
                  loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.86, 1])

    if SAVE_FIG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{sig}_mean_serial_by_video.png"
        plt.savefig(out, dpi=DPI)
        print(f"[INFO] Gespeichert: {out.resolve()}")

    plt.show()
