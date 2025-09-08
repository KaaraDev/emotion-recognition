# GSR (EDA) over time with video shading — Subject 1
# Legend: chronological order by segment; Baseline moved to the end.
# Colors: Emotion 1 & Emotion 2 share the same color.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
from pathlib import Path
import matplotlib.patches as mpatches

# --------------------------
# CONFIG
# --------------------------
PATH_PHYSIO = "../case_dataset-master/data/interpolated/physiological/sub_1.csv"  # <-- change if needed

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
    if np.nanmax(arr) > 1e5:
        return arr / 1000.0 / 60.0  # ms → min
    return arr / 60.0  # sec → min


def category_from_label(label: str) -> str:
    """Collapse 'Amusement 1/2' → 'Amusement', etc.; keep Baseline/Start/End."""
    base = label.split()[0]  # first word (Amusement/Boredom/Relaxation/Scary/Baseline/Start/End)
    if base in {"Amusement", "Boredom", "Relaxation", "Scary"}:
        return base
    return label  # Baseline / Start / End remain as-is


# --------------------------
# Load physiological CSV
# --------------------------
if not Path(PATH_PHYSIO).exists():
    raise FileNotFoundError(f"File not found: {PATH_PHYSIO}")

df = pd.read_csv(PATH_PHYSIO)
cols = df.columns

time_col = pick(cols, "daqtime", "time_ms", "timestamp", "time", "jstime")
gsr_col = pick(cols, "gsr", "eda", "skinconductance", "electrodermalactivity")
video_col = pick(cols, "video", "videoid", "stimulus")

if not (time_col and gsr_col and video_col):
    raise ValueError(f"Missing required column(s). Found: {list(cols)}")

t_min = ms_or_s_to_minutes(df[time_col].values)
gsr = df[gsr_col].astype(float).values
vids = df[video_col].astype(int).values

# --------------------------
# Build video segments (start_min, end_min, video_id)
# --------------------------
segments = []
if len(vids):
    start_idx = 0
    for i in range(1, len(vids)):
        if vids[i] != vids[i - 1]:
            segments.append((t_min[start_idx], t_min[i - 1], vids[i - 1]))
            start_idx = i
    segments.append((t_min[start_idx], t_min[-1], vids[-1]))

# Map IDs → labels and categories (for same-color 1/2)
labeled_segments = []
for st, en, vid in segments:
    label = video_map.get(int(vid), str(vid))
    cat = category_from_label(label)
    labeled_segments.append((st, en, label, cat))

# --------------------------
# Color mapping by CATEGORY
# --------------------------
# All "Amusement 1/2" share a color, etc.
color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
if not color_cycle:
    color_cycle = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
                   "#bab0ab"]

categories_in_order = []
for _, _, _, cat in labeled_segments:
    if cat not in categories_in_order:
        categories_in_order.append(cat)

cat_color = {cat: color_cycle[i % len(color_cycle)] for i, cat in enumerate(categories_in_order)}

# --------------------------
# Plot
# --------------------------
plt.figure(figsize=(14, 6))
plt.plot(t_min, gsr, label="GSR (EDA)", color="black", alpha=0.85)
plt.xlabel("Time [minutes]")
plt.ylabel("GSR (µS)")
plt.title("Subject 1 — GSR (EDA) over time with video segments")

# Shade segments
for st, en, label, cat in labeled_segments:
    if en > st:
        plt.axvspan(st, en, alpha=0.15, facecolor=cat_color[cat])

# --------------------------
# Legend: chronological order; Baseline last
# --------------------------
handles = []
baseline_handles = []
for st, en, label, cat in labeled_segments:
    patch = mpatches.Patch(alpha=0.3, facecolor=cat_color[cat], label="\n".join(wrap(label, width=42)))
    if label.startswith("Baseline"):
        baseline_handles.append(patch)
    else:
        handles.append(patch)

# Append Baseline entries at the end (if any)
handles.extend(baseline_handles)

plt.legend(handles=handles, title="Played video", loc="upper right", framealpha=0.95)

# Optional: vertical dashed lines at boundaries
for i in range(1, len(labeled_segments)):
    plt.axvline(labeled_segments[i][0], linestyle="--", linewidth=0.8, alpha=0.5)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
