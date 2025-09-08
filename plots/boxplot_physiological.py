import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# --------------------------
# 1. Load physiological data
# --------------------------
files = glob.glob("../case_dataset-master/data/interpolated/physiological/sub_*.csv")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["subject"] = f.split("_")[-1].replace(".csv","")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# --------------------------
# 2. Define video categories
# (example mapping, adjust if needed based on metadata)
# --------------------------
video_map = {
    1: "Amusement",
    2: "Amusement",
    3: "Boredom",
    4: "Boredom",
    5: "Relaxation",
    6: "Relaxation",
    7: "Scary",
    8: "Scary",
    # Baselines (blue screens) might have their own IDs — skip them here
}

# Keep only emotional videos
data = data[data["video"].isin(video_map.keys())]
data["category"] = data["video"].map(video_map)

# --------------------------
# 3. Compute per-subject mean GSR per category
# --------------------------
means = data.groupby(["subject", "category"])["emg_trap"].mean().reset_index()

# --------------------------
# 4. Plot boxplots
# --------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x="category", y="emg_trap", data=means, palette="Set2")
sns.stripplot(x="category", y="emg_trap", data=means, color="black", alpha=0.5, jitter=True)

plt.title("Surface Electromyography (sEMG) back muscles per Video Category")
plt.xlabel("Video Category")
plt.ylabel("Microvolts (uV)")
plt.tight_layout()
plt.show()
