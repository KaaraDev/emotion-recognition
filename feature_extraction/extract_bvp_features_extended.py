# =========================
# CASE Dataset – BVP Feature Extraction (NeuroKit2, extended)
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
    """Estimate the sampling rate (Hz, integer) from a 'daqtime' column (milliseconds)."""
    vals = daqtime_series.values.astype(float)
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("Could not infer sampling rate from 'daqtime'.")
    median_ms = float(np.median(diffs))
    sr = 1000.0 / median_ms if median_ms > 0 else 1000.0
    return max(1, int(round(sr)))


def detect_signal_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching signal column in a dataframe (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# ---- math/stat helpers ----
def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis_excess(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s == 0:
        return -3.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _auc(y: np.ndarray, dt: float) -> float:
    """Area under curve via trapezoid rule (NumPy >= 2.0)."""
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return float(np.trapezoid(y, dx=dt))


def _basic_stats(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    """Min/Max/Mean/SD/Median/Range/Skew/KurtosisExcess."""
    out: Dict[str, float] = {}
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        keys = ["Min", "Max", "Mean", "SD", "Median", "Range", "Skew", "KurtosisExcess"]
        out.update({f"{prefix}_{k}": np.nan for k in keys})
        return out
    out[f"{prefix}_Min"] = float(np.nanmin(x))
    out[f"{prefix}_Max"] = float(np.nanmax(x))
    out[f"{prefix}_Mean"] = float(np.nanmean(x))
    out[f"{prefix}_SD"] = float(np.nanstd(x))
    out[f"{prefix}_Median"] = float(np.nanmedian(x))
    out[f"{prefix}_Range"] = float(np.nanmax(x) - np.nanmin(x))
    out[f"{prefix}_Skew"] = _skew(x)
    out[f"{prefix}_KurtosisExcess"] = _kurtosis_excess(x)
    return out


# ---------- Feature Extraction (NeuroKit2) ----------

def _rate_features(sig_df: pd.DataFrame) -> Dict[str, float]:
    """Rich stats from PPG_Rate (Heart Rate in bpm)."""
    out: Dict[str, float] = {}
    if "PPG_Rate" in sig_df.columns:
        hr = sig_df["PPG_Rate"].to_numpy(dtype=float)
        out.update(_basic_stats("BVP_HR", hr))  # Mean/SD/Min/Max/Range + shape
    else:
        out.update({k: np.nan for k in
                    ["BVP_HR_Min", "BVP_HR_Max", "BVP_HR_Mean", "BVP_HR_SD",
                     "BVP_HR_Median", "BVP_HR_Range", "BVP_HR_Skew", "BVP_HR_KurtosisExcess"]})
    return out


def _signal_features(sig_df: pd.DataFrame, sr: int) -> Dict[str, float]:
    """Stats + AUC + slope on the cleaned PPG signal."""
    out: Dict[str, float] = {}
    if "PPG_Clean" in sig_df.columns:
        x = sig_df["PPG_Clean"].to_numpy(dtype=float)
        dt = 1.0 / float(sr) if sr > 0 else np.nan
        out.update(_basic_stats("BVP_Signal", x))
        out["BVP_Signal_AUC"] = _auc(x, dt)
        # linear trend/slope over Index
        if np.isfinite(x).sum() > 1:
            idx = np.arange(len(x), dtype=float)
            coeffs = np.polyfit(idx, x, 1)
            out["BVP_Signal_Slope"] = float(coeffs[0])
        else:
            out["BVP_Signal_Slope"] = np.nan
    else:
        out.update({k: np.nan for k in
                    ["BVP_Signal_Min", "BVP_Signal_Max", "BVP_Signal_Mean", "BVP_Signal_SD",
                     "BVP_Signal_Median", "BVP_Signal_Range", "BVP_Signal_Skew",
                     "BVP_Signal_KurtosisExcess", "BVP_Signal_AUC", "BVP_Signal_Slope"]})
    return out


def _prv_features_from_peaks(peaks_idx: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Pulse Rate Variability (PRV) metrics from PPG peaks:
      - IBI stats (seconds): mean, SD (SDNN), median, range, min, max
      - RMSSD (seconds)
      - pNN50 analog (percentage of successive IBI diffs > 50 ms)
      - Beat count
    """
    out: Dict[str, float] = {}
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    peaks_idx = peaks_idx[np.isfinite(peaks_idx)]
    peaks_idx = np.unique(peaks_idx)
    peaks_idx.sort()

    out["BVP_Beat_Count"] = int(peaks_idx.size)

    if peaks_idx.size >= 2 and sr > 0:
        ibi_sec = np.diff(peaks_idx) / float(sr)  # seconds
        out.update(_basic_stats("BVP_IBIsec", ibi_sec))
        # Time-domain PRV
        diff_ibi = np.diff(ibi_sec)
        rmssd = np.sqrt(np.nanmean(diff_ibi ** 2)) if diff_ibi.size else np.nan
        out["BVP_PRV_RMSSD"] = float(rmssd) if np.isfinite(rmssd) else np.nan
        # pNN50 analog with 50 ms threshold
        if diff_ibi.size:
            pnn50 = np.nanmean((np.abs(diff_ibi) > 0.050).astype(float)) * 100.0
        else:
            pnn50 = np.nan
        out["BVP_PRV_pNN50"] = float(pnn50) if np.isfinite(pnn50) else np.nan
    else:
        # no intervals possible
        keys = ["BVP_IBIsec_Min", "BVP_IBIsec_Max", "BVP_IBIsec_Mean", "BVP_IBIsec_SD",
                "BVP_IBIsec_Median", "BVP_IBIsec_Range", "BVP_IBIsec_Skew", "BVP_IBIsec_KurtosisExcess",
                "BVP_PRV_RMSSD", "BVP_PRV_pNN50"]
        out.update({k: np.nan for k in keys})

    return out


def extract_bvp_features(signal: pd.Series, sr: int) -> Dict[str, float]:
    """
    Extract robust BVP features using NeuroKit2 + custom aggregates:
      - Rich HR stats from PPG_Rate
      - Beat count + PRV (IBI stats, RMSSD, pNN50)
      - Clean signal stats, AUC, slope
    """
    y = np.asarray(signal, dtype=float)

    # NeuroKit2 preprocessing and peak detection
    sig, info = nk.ppg_process(y, sampling_rate=sr)

    out: Dict[str, float] = {}

    # 1) HR statistics from PPG_Rate (Mean/SD/Median/Min/Max/Range, shape)
    out.update(_rate_features(sig))

    # 2) Beat count & PRV from peaks
    if isinstance(info, dict) and "PPG_Peaks" in info and info["PPG_Peaks"] is not None:
        peaks_idx = np.asarray(info["PPG_Peaks"], dtype=int)
    else:
        # fallback to binary peak vector in the processed signal
        peaks_idx = np.where(np.asarray(sig.get("PPG_Peaks", np.zeros(len(sig)))) == 1)[0]
    out.update(_prv_features_from_peaks(peaks_idx, sr))

    # 3) Cleaned signal statistics + AUC + slope
    out.update(_signal_features(sig, sr))

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
        # Return empty frame with expected columns
        cols = ["subject", "video", "sampling_rate_hz",
                "BVP_HR_Min", "BVP_HR_Max", "BVP_HR_Mean", "BVP_HR_SD", "BVP_HR_Median", "BVP_HR_Range",
                "BVP_HR_Skew", "BVP_HR_KurtosisExcess",
                "BVP_Beat_Count",
                "BVP_IBIsec_Min", "BVP_IBIsec_Max", "BVP_IBIsec_Mean", "BVP_IBIsec_SD",
                "BVP_IBIsec_Median", "BVP_IBIsec_Range", "BVP_IBIsec_Skew", "BVP_IBIsec_KurtosisExcess",
                "BVP_PRV_RMSSD", "BVP_PRV_pNN50",
                "BVP_Signal_Min", "BVP_Signal_Max", "BVP_Signal_Mean", "BVP_Signal_SD",
                "BVP_Signal_Median", "BVP_Signal_Range", "BVP_Signal_Skew", "BVP_Signal_KurtosisExcess",
                "BVP_Signal_AUC", "BVP_Signal_Slope"]
        return pd.DataFrame(columns=cols)

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
