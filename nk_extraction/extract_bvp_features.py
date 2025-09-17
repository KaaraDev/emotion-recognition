# =========================
# CASE Dataset – BVP Feature Extraction (NeuroKit2 only)
# =========================
# Requirements:
#   pip install neurokit2 pandas numpy
# =========================

import os
import glob
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

import neurokit2 as nk  # Hard requirement: NeuroKit2 must be installed

# ---------- USER SETTINGS ----------
BASE_DIR = r"../case_dataset-master/data/interpolated/physiological"
OUT_CSV = "case_features_bvp.csv"


# ---------- Helpers ----------

def infer_sampling_rate_ms(daqtime_series: pd.Series) -> int:
    """
    Estimate the sampling rate (Hz, integer) from a 'daqtime' column
    given in milliseconds.
    """
    vals = daqtime_series.values.astype(float)
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("Could not infer sampling rate from 'daqtime'.")
    median_ms = float(np.median(diffs))
    sr = 1000.0 / median_ms if median_ms > 0 else 1000.0
    return max(1, int(round(sr)))


def detect_signal_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find the first matching signal column in a dataframe (case-insensitive).
    Useful for flexible column naming across datasets.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _safe_from_rate(sig_df: pd.DataFrame, rate_col: str) -> Dict[str, float]:
    """
    Extract HR mean and SD from a *_Rate column (if available).
    Returns NaNs if the column is missing.
    """
    out: Dict[str, float] = {}
    if rate_col in sig_df.columns:
        out["BVP_HR_Mean"] = float(np.nanmean(sig_df[rate_col]))
        out["BVP_HR_SD"] = float(np.nanstd(sig_df[rate_col]))
    else:
        out["BVP_HR_Mean"] = np.nan
        out["BVP_HR_SD"] = np.nan
    return out


# ---------- Feature Extraction (NeuroKit2) ----------

def extract_bvp_features(signal: pd.Series, sr: int) -> Dict[str, float]:
    """
    Extract robust BVP features using NeuroKit2:
      - BVP_HR_Mean, BVP_HR_SD (from PPG_Rate)
      - BVP_Beat_Count (number of detected peaks)
      - BVP_Signal_Mean / SD / Max (from PPG_Clean)
    """
    y = np.asarray(signal, dtype=float)

    # Run NeuroKit2 preprocessing and peak detection
    sig, info = nk.ppg_process(y, sampling_rate=sr)

    # HR statistics from PPG_Rate
    out = _safe_from_rate(sig, "PPG_Rate")

    # Beat count: prefer info dict, else fallback to binary peak vector
    if isinstance(info, dict) and "PPG_Peaks" in info and info["PPG_Peaks"] is not None:
        peaks_idx = np.asarray(info["PPG_Peaks"], dtype=int)
    else:
        peaks_idx = np.where(np.asarray(sig.get("PPG_Peaks", np.zeros(len(sig)))) == 1)[0]
    peaks_idx = np.unique(peaks_idx)
    peaks_idx.sort()
    out["BVP_Beat_Count"] = int(peaks_idx.size)

    # Clean signal statistics (mean, SD, max)
    if "PPG_Clean" in sig.columns:
        clean = sig["PPG_Clean"].to_numpy(dtype=float)
        out["BVP_Signal_Mean"] = float(np.nanmean(clean))
        out["BVP_Signal_SD"] = float(np.nanstd(clean))
        out["BVP_Signal_Max"] = float(np.nanmax(clean))
    else:
        out["BVP_Signal_Mean"] = np.nan
        out["BVP_Signal_SD"] = np.nan
        out["BVP_Signal_Max"] = np.nan

    return out


# ---------- Per-file processing ----------

def process_subject_file(filepath: str) -> pd.DataFrame:
    """
    Process a single subject file:
      - Estimate sampling rate
      - Detect BVP column
      - Extract features per video segment
    """
    df = pd.read_csv(filepath)

    # Check required columns
    for col in ("daqtime", "video"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {os.path.basename(filepath)}")

    sr = infer_sampling_rate_ms(df["daqtime"])
    subj = os.path.splitext(os.path.basename(filepath))[0]

    # Detect BVP column
    bvp_col = detect_signal_column(df, ["bvp", "ppg", "BVP", "PPG", "blood_volume_pulse"])
    if bvp_col is None:
        return pd.DataFrame(columns=[
            "subject", "video", "sampling_rate_hz",
            "BVP_HR_Mean", "BVP_HR_SD", "BVP_Beat_Count",
            "BVP_Signal_Mean", "BVP_Signal_SD", "BVP_Signal_Max"
        ])

    rows: List[Dict[str, float]] = []
    for vid, g in df.groupby("video"):
        g = g.reset_index(drop=True)
        base: Dict[str, float] = {
            "subject": subj,
            "video": vid,
            "sampling_rate_hz": int(sr),
        }
        base.update(extract_bvp_features(g[bvp_col], sr))
        rows.append(base)

    return pd.DataFrame(rows)


# ---------- Main ----------

def main():
    pattern = os.path.join(BASE_DIR, "sub_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files found in {BASE_DIR} (pattern: sub_*.csv).")

    all_features: List[pd.DataFrame] = []
    for fp in files:
        print(f"Processing {os.path.basename(fp)} ...")
        try:
            feats = process_subject_file(fp)
            if not feats.empty:
                all_features.append(feats)
            else:
                print(f"[Skip] {os.path.basename(fp)}: no BVP column found.")
        except Exception as e:
            print(f"[Skip] {os.path.basename(fp)} due to error: {e}")

    if not all_features:
        raise SystemExit("No BVP features extracted.")

    features_df = pd.concat(all_features, ignore_index=True)
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]
    features_df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved BVP features to: {OUT_CSV}")
    print(f"Shape: {features_df.shape}")


if __name__ == "__main__":
    main()
