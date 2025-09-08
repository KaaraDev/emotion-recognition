# Simple BVP plot with video shading — Subject 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

# --------------------------
# CONFIG
# --------------------------
PATH_PHYSIO = "../case_dataset-master/data/interpolated/physiological/sub_22.csv"  # change if needed

video_map = {
    1: "Amusement 1",
    2: "Amusement 2",
    3: "Boredom 1",
    4: "Boredom 2",
    5: "Relaxation 1",
    6: "Relaxation 2",
    7: "Scary 1",
    8: "Scary 2",
    10: "Start",
    11: "Pause",
    12: "End",
}
UTILITY_LABELS = {"Start", "Pause", "End"}


# --------------------------
# Helpers
# --------------------------
def pick(col_names, *candidates):
    lc = {c.lower(): c for c in col_names}
    for cand in candidates:
        if cand in lc:
            return lc[cand]
    return None


def ms_or_s_to_minutes(arr):
    arr = np.asarray(arr, dtype=float)
    if np.nanmax(arr) > 1e5:  # ms
        return arr / 1000.0 / 60.0
    return arr / 60.0  # s


def category_from_label(label: str) -> str:
    base = label.split()[0]
    if base in {"Amusement", "Boredom", "Relaxation", "Scary"}:
        return base
    return label  # Start/Pause/End unchanged


# --------------------------
# Load data
# --------------------------
if not Path(PATH_PHYSIO).exists():
    raise FileNotFoundError(f"File not found: {PATH_PHYSIO}")

df = pd.read_csv(PATH_PHYSIO)

time_col = pick(df.columns, "daqtime", "time_ms", "timestamp", "time", "jstime")
# common BVP/PPG variants
bvp_col = pick(df.columns, "bvp", "ppg", "bloodvolumepulse", "blood_volume_pulse",
               "photoplethysmography", "pulse")
video_col = pick(df.columns, "video", "videoid", "stimulus")

if not (time_col and bvp_col and video_col):
    raise ValueError(f"Missing required column(s). Found: {list(df.columns)}")

t_min = ms_or_s_to_minutes(df[time_col].values)
bvp = pd.to_numeric(df[bvp_col], errors="coerce").values
vids = pd.to_numeric(df[video_col], errors="coerce").values

# --------------------------
# Build segments
# --------------------------
segments = []
if len(vids):
    start = 0
    for i in range(1, len(vids)):
        if not np.isfinite(vids[i]) or not np.isfinite(vids[i - 1]) or vids[i] != vids[i - 1]:
            if np.isfinite(vids[i - 1]):
                lab = video_map.get(int(vids[i - 1]), str(int(vids[i - 1])))
                segments.append((t_min[start], t_min[i - 1], lab))
            start = i
    if np.isfinite(vids[-1]):
        lab = video_map.get(int(vids[-1]), str(int(vids[-1])))
        segments.append((t_min[start], t_min[-1], lab))

# --------------------------
# Colors (category-based so 1/2 share a color)
# --------------------------
color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
cats = []
for _, _, lab in segments:
    cat = category_from_label(lab)
    if cat not in cats:
        cats.append(cat)
cat_color = {cat: color_cycle[i % len(color_cycle)] for i, cat in enumerate(cats)}

# --------------------------
# Plot BVP with shaded videos
# --------------------------
plt.figure(figsize=(14, 6))
plt.plot(t_min, bvp, color="black", linewidth=0.8, label="BVP")
plt.xlabel("Time [minutes]")
plt.ylabel("BVP (raw units)")
plt.title("Subject 2 — BVP over time with video segments")

for st, en, lab in segments:
    if en > st:
        plt.axvspan(st, en, alpha=0.15, facecolor=cat_color[category_from_label(lab)])

# Legend: one entry per label in order; utility last
seen = set()
ordered_labels = []
for _, _, lab in segments:
    if lab not in seen:
        seen.add(lab)
        ordered_labels.append(lab)
ordered_labels = [l for l in ordered_labels if l not in UTILITY_LABELS] + \
                 [l for l in ordered_labels if l in UTILITY_LABELS]

handles = [mpatches.Patch(alpha=0.3, facecolor=cat_color[category_from_label(lab)], label=lab)
           for lab in ordered_labels]
plt.legend(handles=handles, title="Videos", loc="upper right", framealpha=0.95)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
