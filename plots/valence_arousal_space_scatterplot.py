import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# 1. Load annotation files
# --------------------------
files = [f"../case_dataset-master/data/interpolated/annotations/sub_{i}.csv" for i in range(1, 31)]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["subject"] = f.split("_")[-1].replace(".csv", "")  # Subject ID
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# --------------------------
# 2. Compute mean valence/arousal per subject per video
# --------------------------
means = data.groupby(["subject", "video"])[["valence", "arousal"]].mean().reset_index()

# --------------------------
# 3. Compute grand mean across subjects per video
# --------------------------
video_means = means.groupby("video")[["valence", "arousal"]].mean().reset_index()

# --------------------------
# 4. Plot Valence–Arousal space
# --------------------------
fig, ax = plt.subplots(figsize=(7, 7))

# Color subject means by video ID
scatter = ax.scatter(means["valence"], means["arousal"],
                     c=means["video"], cmap="tab10", alpha=0.6, label="Subject means")

# Add colorbar to show which color = which video
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Video ID")

# Grand means highlighted
ax.scatter(video_means["valence"], video_means["arousal"],
           color="black", s=120, marker="X", label="Video mean")

# Label each grand mean with its video ID
for _, row in video_means.iterrows():
    ax.text(row["valence"]+0.1, row["arousal"]+0.1, str(int(row["video"])), fontsize=9)

# Axis formatting
ax.set_xlim(0.5, 9.5)
ax.set_ylim(0.5, 9.5)
ax.set_xlabel("Valence (0.5 = negative, 9.5 = positive)")
ax.set_ylabel("Arousal (0.5 = calm, 9.5 = excited)")
ax.set_title("Valence–Arousal Space (CASE Dataset)")
ax.grid(True)
ax.legend()

plt.show()
