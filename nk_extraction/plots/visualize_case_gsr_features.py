# =========================
# Visualize CASE GSR Features from case_features_gsr.csv
# =========================
# Requirements:
#   pip install pandas numpy matplotlib
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- USER SETTINGS ----------
CSV_PATH = "../case_features_gsr.csv"  # change if saved elsewhere
SAVE_FIGS = True  # also save PNGs next to the CSV


# -----------------------------------

def pick_feature_columns(df: pd.DataFrame):
    """Find reasonable columns to visualize."""
    # priority order for a 'reactivity' metric (rate/peaks), then amplitude, then cleaned mean
    candidates_in_order = [
        # common NK names
        "EDA_SCRs_per_min", "EDA_SCR_Per_Min", "EDA_SCR_Rate",
        "EDA_SCR_Count", "EDA_SCR_Peaks_N", "EDA_SCR_Peaks_Count",
        "EDA_Peaks_Amplitude_Mean", "EDA_SCR_Peaks_Amplitude_Mean",
        # fallback names from our robust script
        "EDA_Clean_Mean", "EDA_Clean_SD", "EDA_Clean_Slope",
    ]
    available = [c for c in candidates_in_order if c in df.columns]
    # also collect all numeric EDA columns for correlation/overview
    num_cols = [c for c in df.columns if c.startswith("EDA_") and pd.api.types.is_numeric_dtype(df[c])]
    return available, num_cols


def safe_bar_per_video(df: pd.DataFrame, col: str, title_suffix: str = ""):
    means = df.groupby("video")[col].mean().sort_index()
    plt.figure(figsize=(11, 5))
    means.plot(kind="bar")
    plt.title(f"{col} per Video{title_suffix}")
    plt.xlabel("Video ID")
    plt.ylabel(col)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"{col}_per_video.png", dpi=200)
    plt.show()


def box_per_video(df: pd.DataFrame, col: str):
    order = sorted(df["video"].unique())
    data = [df.loc[df["video"] == v, col].dropna().values for v in order]
    plt.figure(figsize=(11, 5))
    plt.boxplot(data, labels=order, showfliers=False)
    plt.title(f"Distribution of {col} per Video")
    plt.xlabel("Video ID")
    plt.ylabel(col)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"{col}_box_per_video.png", dpi=200)
    plt.show()


def heatmap_video_subject(df: pd.DataFrame, col: str):
    # pivot: rows=video, cols=subject
    pvt = df.pivot_table(index="video", columns="subject", values=col, aggfunc="mean")
    plt.figure(figsize=(12, 7))
    # simple imshow heatmap
    im = plt.imshow(pvt.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"{col} — Video × Subject")
    plt.yticks(ticks=np.arange(len(pvt.index)), labels=pvt.index)
    plt.xticks(ticks=np.arange(len(pvt.columns)), labels=pvt.columns, rotation=90)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"{col}_video_subject_heatmap.png", dpi=200)
    plt.show()


def correlation_heatmap(df: pd.DataFrame, numeric_cols):
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Correlation between EDA Features")
    plt.xticks(ticks=np.arange(len(numeric_cols)), labels=numeric_cols, rotation=90)
    plt.yticks(ticks=np.arange(len(numeric_cols)), labels=numeric_cols)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("eda_features_correlation.png", dpi=200)
    plt.show()


def scatter_matrix(df: pd.DataFrame, numeric_cols):
    if len(numeric_cols) < 2:
        return
    # use pandas' scatter_matrix (matplotlib backend)
    from pandas.plotting import scatter_matrix
    plt.figure(figsize=(8, 8))
    scatter_matrix(df[numeric_cols].dropna(), figsize=(10, 10), diagonal='hist')
    plt.suptitle("Scatter Matrix of EDA Features", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("eda_features_scatter_matrix.png", dpi=200)
    plt.show()


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required = {"subject", "video"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV missing required columns {required}. Got: {df.columns.tolist()}")

    # choose columns to plot
    priority_cols, numeric_cols = pick_feature_columns(df)

    if not priority_cols and not numeric_cols:
        raise SystemExit("No EDA feature columns found (columns beginning with 'EDA_').")

    # 1) Primary bar plot per video (first good column)
    if priority_cols:
        col = priority_cols[0]
        safe_bar_per_video(df, col)
        box_per_video(df, col)
        heatmap_video_subject(df, col)
    else:
        print("No priority EDA metric found; skipping per-video bar/box/heatmap.")

    # 2) Correlation heatmap among all EDA numeric features
    if numeric_cols:
        correlation_heatmap(df, numeric_cols)
        # 3) Scatter matrix to inspect relationships
        # (can be heavy if many columns; keep it but you can comment out)
        scatter_matrix(df, numeric_cols)
    else:
        print("No multiple EDA numeric features found for correlation/scatter matrix.")

    # 4) quick table: per-video summary of all numeric EDA features
    summary = df.groupby("video")[numeric_cols].mean().sort_index() if numeric_cols else None
    if summary is not None:
        print("\nPer-video mean of EDA features:")
        print(summary.round(3).head(10))  # print first 10 videos for brevity
        if SAVE_FIGS:
            summary.to_csv("per_video_eda_summary.csv")
            print("Saved per-video summary to per_video_eda_summary.csv")


if __name__ == "__main__":
    main()
