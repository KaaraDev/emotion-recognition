import os
from math import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
CSV_PATH = "../case_features_gsr.csv"
df = pd.read_csv(CSV_PATH)

# Where figures are saved
FIG_DIR = "figures_gsr"
os.makedirs(FIG_DIR, exist_ok=True)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# Detect numeric GSR features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
FEATURES = [c for c in df.columns if c.startswith("EDA_") and c in num_cols]
if not FEATURES:
    raise ValueError("No numeric columns starting with 'GSR_' were found.")

# Map numeric video IDs to readable categories
video_category_map = {
    1: "Amusing 1", 2: "Amusing 2",
    3: "Boring 1", 4: "Boring 2",
    5: "Relaxed 1", 6: "Relaxed 2",
    7: "Scary 1", 8: "Scary 2",
    10: "startVid", 11: "bluVid", 12: "endVid"
}
if "video" not in df.columns:
    raise KeyError("Expected a 'video' column in the CSV.")
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
# Pick up to 3 most variable GSR features across videos for clear radar shapes
video_means_all = df.groupby("video")[FEATURES].mean()
variances = video_means_all.var(axis=0).sort_values(ascending=False)
radar_features = variances.index.tolist()[:3] if len(variances) >= 3 else variances.index.tolist()

# Min–max normalization per feature to make shapes comparable
video_means = video_means_all[FEATURES]
norm_means = (video_means - video_means.min()) / (video_means.max() - video_means.min() + 1e-12)

angles = [n / float(len(FEATURES)) * 2 * pi for n in range(len(FEATURES))]
angles += angles[:1]

for v in sorted(df["video"].unique()):
    # Skip videos that had NaNs for all radar features (if any)
    if v not in norm_means.index:
        continue
    values = norm_means.loc[v, :].fillna(0).tolist()
    values += values[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    plt.xticks(angles[:-1], FEATURES)
    plt.title(f"GSR Radar (normalized) — Video {v}")
    save_current_fig(f"gsr_radar_normalized_video_{v}.png")

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

print("\nSaved GSR per-video RADAR plots + category-level boxplots and bar charts in 'figures_gsr'.")
