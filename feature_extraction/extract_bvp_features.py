# =========================
# CASE Dataset – BVP Feature Extraction (NeuroKit2, minimal)
# =========================
# Requirements:
#   pip install neurokit2 pandas numpy
# =========================

import os
import glob
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import neurokit2 as nk

# ---------- USER SETTINGS ----------
BASE_DIR = r"../case_dataset-master/data/interpolated/physiological"
OUT_CSV = "case_features_bvp.csv"


# ---------- Small helpers ----------

def infer_sampling_rate_ms(daqtime_series: pd.Series) -> int:
    """Estimate sampling rate (Hz) from 'daqtime' column in milliseconds."""
    vals = daqtime_series.values.astype(float)
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("Could not infer sampling rate from 'daqtime'.")
    median_ms = float(np.median(diffs))
    sr = 1000.0 / median_ms if median_ms > 0 else 1000.0
    return max(1, int(round(sr)))


def detect_signal_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching signal column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# ---------- Core feature extraction ----------

def extract_bvp_features(signal: pd.Series, sr: int) -> Dict[str, float]:
    """
    Minimal, robust BVP features using NeuroKit2:
      - HR (PPG_Rate): mean, sd, min, max
      - Beat count
      - IBI (sec): mean, sd   (SDNN analog)
      - RMSSD (sec)
      - pNN50 (%), threshold 50 ms
    """
    y = np.asarray(signal, dtype=float)

    # NeuroKit2 processing
    sig, info = nk.ppg_process(y, sampling_rate=sr)

    out: Dict[str, float] = {
        "BVP_HR_Mean": np.nan,
        "BVP_HR_SD": np.nan,
        "BVP_HR_Min": np.nan,
        "BVP_HR_Max": np.nan,
        "BVP_Beat_Count": 0,
        "BVP_IBIsec_Mean": np.nan,
        "BVP_IBIsec_SD": np.nan,
        "BVP_PRV_RMSSD": np.nan,
        "BVP_PRV_pNN50": np.nan,
    }

    # HR features
    if "PPG_Rate" in sig.columns:
        hr = sig["PPG_Rate"].to_numpy(dtype=float)
        hr = hr[np.isfinite(hr)]
        if hr.size:
            out["BVP_HR_Mean"] = float(np.nanmean(hr))
            out["BVP_HR_SD"] = float(np.nanstd(hr))
            out["BVP_HR_Min"] = float(np.nanmin(hr))
            out["BVP_HR_Max"] = float(np.nanmax(hr))

    # Peaks -> PRV
    if isinstance(info, dict) and "PPG_Peaks" in info and info["PPG_Peaks"] is not None:
        peaks_idx = np.asarray(info["PPG_Peaks"], dtype=int)
    else:
        peaks_idx = np.where(np.asarray(sig.get("PPG_Peaks", np.zeros(len(sig)))) == 1)[0]

    peaks_idx = np.unique(peaks_idx[np.isfinite(peaks_idx)]).astype(int)
    peaks_idx.sort()
    out["BVP_Beat_Count"] = int(peaks_idx.size)

    if sr > 0 and peaks_idx.size >= 2:
        ibi_sec = np.diff(peaks_idx) / float(sr)
        ibi_sec = ibi_sec[np.isfinite(ibi_sec)]
        if ibi_sec.size:
            out["BVP_IBIsec_Mean"] = float(np.nanmean(ibi_sec))
            out["BVP_IBIsec_SD"] = float(np.nanstd(ibi_sec))

            diff_ibi = np.diff(ibi_sec)
            if diff_ibi.size:
                rmssd = np.sqrt(np.nanmean(diff_ibi ** 2))
                out["BVP_PRV_RMSSD"] = float(rmssd)

                pnn50 = np.nanmean((np.abs(diff_ibi) > 0.050).astype(float)) * 100.0
                out["BVP_PRV_pNN50"] = float(pnn50)

    return out


# ---------- Per-file processing ----------

def process_subject_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Required columns
    for col in ("daqtime", "video"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {os.path.basename(filepath)}")

    sr = infer_sampling_rate_ms(df["daqtime"])
    subj = os.path.splitext(os.path.basename(filepath))[0]

    # BVP column
    bvp_col = detect_signal_column(df, ["bvp", "ppg", "BVP", "PPG", "blood_volume_pulse"])
    if bvp_col is None:
        cols = ["subject", "video", "sampling_rate_hz",
                "BVP_HR_Mean", "BVP_HR_SD", "BVP_HR_Min", "BVP_HR_Max",
                "BVP_Beat_Count", "BVP_IBIsec_Mean", "BVP_IBIsec_SD",
                "BVP_PRV_RMSSD", "BVP_PRV_pNN50"]
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, float]] = []
    for vid, g in df.groupby("video"):
        g = g.reset_index(drop=True)
        feats = extract_bvp_features(g[bvp_col], sr)
        rows.append({
            "subject": subj,
            "video": vid,
            "sampling_rate_hz": int(sr),
            **feats
        })

    return pd.DataFrame(rows)


# ---------- Main ----------

def main():
    pattern = os.path.join(BASE_DIR, "sub_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files found in {BASE_DIR} (pattern: sub_*.csv).")

    out_frames: List[pd.DataFrame] = []
    for fp in files:
        print(f"Processing {os.path.basename(fp)} ...")
        try:
            feats = process_subject_file(fp)
            if not feats.empty:
                out_frames.append(feats)
            else:
                print(f"[Skip] {os.path.basename(fp)}: no BVP column found.")
        except Exception as e:
            print(f"[Skip] {os.path.basename(fp)} due to error: {e}")

    if not out_frames:
        raise SystemExit("No BVP features extracted.")

    features_df = pd.concat(out_frames, ignore_index=True)
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]
    features_df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved BVP features to: {OUT_CSV}")
    print(f"Shape: {features_df.shape}")


if __name__ == "__main__":
    main()
