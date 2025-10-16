# GSR (EDA) over time with video shading — Subject 1
# Colors match emotion timeline: same hues per category.
# Emotion 1 & 2 share same color family.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
from pathlib import Path
import matplotlib.patches as mpatches
import colorsys  # for lightness adjustments

# --------------------------
# CONFIG
# --------------------------
PATH_PHYSIO = "../case_dataset-master/data/interpolated/physiological/sub_17.csv"  # change if needed

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

# ---------------------------------------
# Shared color scheme (same as emotion timeline)
# ---------------------------------------
base_colors = {
    "Amusement": "orange",
    "Boredom": "violet",
    "Relaxation": "green",
    "Scary": "red",
    "Start": "tab:grey",
    "Pause": "tab:blue",
    "End": "tab:grey",
}

variant_lightness = {
    1: 0.8,  # darker
    2: 1.35,  # lighter
}


def adjust_lightness(color, factor=1.0):
    """Adjust color brightness: factor <1 = darker, >1 = lighter"""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    return colorsys.hls_to_rgb(h, l, s)


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
    base = label.split()[0]
    if base in {"Amusement", "Boredom", "Relaxation", "Scary"}:
        return base
    return label  # Start / Pause / End


def get_variant_index(label: str) -> int:
    import re
    m = re.search(r"\b(\d+)\b", label)
    return int(m.group(1)) if m else 1


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

# Map IDs → labels and categories
labeled_segments = []
for st, en, vid in segments:
    label = video_map.get(int(vid), str(vid))
    cat = category_from_label(label)
    labeled_segments.append((st, en, label, cat))

# --------------------------
# Plot
# --------------------------
plt.figure(figsize=(14, 6))
plt.plot(t_min, gsr, label="GSR (EDA)", color="black", alpha=0.85)
plt.xlabel("Time [minutes]")
plt.ylabel("GSR (µS)")
plt.title("Subject 17 — GSR (EDA) over time with video segments")

# Shade segments (same color logic)
for st, en, label, cat in labeled_segments:
    if en > st:
        var_idx = get_variant_index(label)
        base = base_colors.get(cat, "tab:purple")
        shade = adjust_lightness(base, variant_lightness.get(var_idx, 1.0))
        plt.axvspan(st, en, alpha=0.15, facecolor=shade)

# --------------------------
# Legend: chronological order
# --------------------------
handles = []
for st, en, label, cat in labeled_segments:
    var_idx = get_variant_index(label)
    base = base_colors.get(cat, "tab:purple")
    shade = adjust_lightness(base, variant_lightness.get(var_idx, 1.0))
    patch = mpatches.Patch(alpha=0.3, facecolor=shade, label="\n".join(wrap(label, width=42)))
    handles.append(patch)

plt.legend(handles=handles, title="Played video", loc="upper right", framealpha=0.95)

# Optional: vertical dashed lines at boundaries
for i in range(1, len(labeled_segments)):
    plt.axvline(labeled_segments[i][0], linestyle="--", linewidth=0.8, alpha=0.5)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
