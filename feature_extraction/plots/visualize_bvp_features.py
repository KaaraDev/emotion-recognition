import os
from math import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
CSV_PATH = "../case_features_bvp.csv"
df = pd.read_csv(CSV_PATH)

# Where figures are saved
FIG_DIR = "figures_bvp"
os.makedirs(FIG_DIR, exist_ok=True)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# Feature set for category plots; radar uses a subset
FEATURES = ["BVP_HR_Mean", "BVP_HR_SD", "BVP_HR_Min", "BVP_HR_Max", "BVP_Beat_Count", "BVP_IBIsec_Mean", "BVP_IBIsec_SD", "BVP_PRV_RMSSD", "BVP_PRV_pNN50"]

# Map numeric video IDs to readable categories
video_category_map = {
    1: "Amusing 1", 2: "Amusing 2",
    3: "Boring 1", 4: "Boring 2",
    5: "Relaxed 1", 6: "Relaxed 2",
    7: "Scary 1", 8: "Scary 2",
    10: "startVid", 11: "bluVid", 12: "endVid"
}
df["category"] = df["video"].map(video_category_map)
if df["category"].isna().any():
    missing_videos = sorted(df.loc[df["category"].isna(), "video"].unique())
    print("Warning: missing category for videos:", missing_videos)
df_cat = df.dropna(subset=["category"]).copy()


def save_current_fig(filename, tight=True, dpi=150):
    """Save current matplotlib figure to FIG_DIR."""
    if tight:
        plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, dpi=dpi)
    print(f"Saved: {out_path}")
    plt.close()


# --------- Per-video radar plots (normalized across videos) ---------
# Use ALL FEATURES on the radar
radar_features = FEATURES[:]  # copy to be explicit

# Compute per-video means for all radar features
video_means = df.groupby("video")[radar_features].mean()

# Min–max normalization per feature to make shapes comparable across videos
denom = (video_means.max() - video_means.min()).replace(0, np.nan)
norm_means = (video_means - video_means.min()) / (denom + 1e-12)

# Angles for each axis + closing angle to complete the polygon
angles = [n / float(len(radar_features)) * 2 * pi for n in range(len(radar_features))]
angles += angles[:1]  # close polygon

# Generate a radar for every video that exists in the normalized table
for v in sorted(norm_means.index):
    values = norm_means.loc[v, :].fillna(0.0).tolist()  # guard against all-constant columns
    values += values[:1]

    plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # Use the feature names as ticks; tilt them slightly for readability
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_features)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("left")

    # Optional: show radial grid from 0 to 1 (normalized scale)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)

    plt.title(f"Radar (normalized) — Video {v}")
    save_current_fig(f"radar_normalized_video_{v}.png")

# --------- Category-level boxplots (wide figure for long labels) ---------
for feat in FEATURES:
    plt.figure(figsize=(20, 6))
    df_cat.boxplot(column=feat, by="category", grid=False, rot=0)
    plt.title(f"{feat} by Category")
    plt.suptitle("")
    plt.xlabel("Category")
    plt.ylabel(feat)
    ax = plt.gca()
    ax.tick_params(axis="x", labelrotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    plt.subplots_adjust(bottom=0.20)
    save_current_fig(f"boxplot_{feat}_by_category.png")

# --------- Category-level bar charts (mean ± std) ---------
grouped_cat = df_cat.groupby("category")[FEATURES].agg(["mean", "std"]).sort_index()

for feat in FEATURES:
    means = grouped_cat[(feat, "mean")]
    stds = grouped_cat[(feat, "std")]
    plt.figure(figsize=(16, 6))
    plt.bar(means.index.astype(str), means.values, yerr=stds.values, capsize=4)
    plt.title(f"Average {feat} by Category (mean ± std)")
    plt.xlabel("Category")
    plt.ylabel(feat)
    ax = plt.gca()
    ax.tick_params(axis="x", labelrotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    plt.subplots_adjust(bottom=0.18)
    save_current_fig(f"bar_{feat}_by_category_mean_std.png")

print(f"\nSaved: per-video RADAR plots + category-level boxplots and bar charts in '{FIG_DIR}'.")
