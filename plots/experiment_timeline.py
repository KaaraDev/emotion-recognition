import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

# --------------------------
# Load one subject's annotation file
# --------------------------
df = pd.read_csv("./case_dataset-master/data/interpolated/annotations/sub_1.csv")

# Convert time to minutes for readability
df["time_min"] = df["jstime"] / 1000 / 60

# --------------------------
# Video ID → Video Name
# Video ID → Category
# --------------------------
video_map = {
    1: "Amusement 1",
    2: "Amusement 2",
    3: "Boredom 1",
    4: "Boredom 2",
    5: "Relaxation 1",
    6: "Relaxation 2",
    7: "Scary 1",
    8: "Scary 2",
    10: "Baseline",
    11: "Start",
    12: "End",
}

# --------------------------
# Create the plot
# --------------------------
fig, ax = plt.subplots(figsize=(14, 6))

# Plot valence and arousal
line_valence, = ax.plot(df["time_min"], df["valence"], label="Valence", alpha=0.8)
line_arousal, = ax.plot(df["time_min"], df["arousal"], label="Arousal", alpha=0.8)

# --------------------------
# Colored backgrounds per video block
# --------------------------
df["block"] = (df["video"] != df["video"].shift()).cumsum()

# Assign one color per category
unique_categories = sorted(set(video_map[v] for v in df["video"].unique()))
category_colors = {cat: plt.cm.tab10(i % 10) for i, cat in enumerate(unique_categories)}

for _, block in df.groupby("block"):
    vid = int(block["video"].iloc[0])
    category = video_map.get(vid, f"Video {vid}")
    start = block["time_min"].iloc[0]
    end = block["time_min"].iloc[-1]
    ax.axvspan(start, end, color=category_colors[category], alpha=0.15)

# --------------------------
# Axis labels & title
# --------------------------
ax.set_title("Valence & Arousal Timeline (Subject 1)")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Rating [0.5 – 9.5]")

# --------------------------
# Legends
# --------------------------
# (1) Line legend (Valence/Arousal)
line_handles, line_labels = ax.get_legend_handles_labels()
hl_ordered = OrderedDict(zip(line_labels, line_handles))  # remove duplicates
line_handles = list(hl_ordered.values())
line_labels = list(hl_ordered.keys())

# (2) Video legend (with shared colors per category)
video_patches = []
for vid in sorted(df["video"].unique()):
    label = video_map.get(vid, f"Video {vid}")
    category = video_map.get(vid, f"Video {vid}")
    patch = mpatches.Patch(color=category_colors[category], alpha=0.3, label=label)
    video_patches.append(patch)

# Combine legends (videos first, then lines)
handles = video_patches + line_handles
labels = [p.get_label() for p in video_patches] + line_labels
ax.legend(handles=handles, labels=labels, loc="upper right")

plt.tight_layout()
plt.show()
