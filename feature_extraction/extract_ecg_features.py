# =========================
# CASE Dataset ECG Feature Extraction with NeuroKit2 (extended & robust, CORE only)
# =========================
# Requirements:
#   pip install neurokit2 pandas numpy matplotlib
# =========================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import neurokit2 as nk

# ---------- USER SETTINGS ----------
BASE_DIR = r"../case_dataset-master/data/interpolated/physiological"


# -----------------------------------

# ---------- small math helpers ----------
def _rolling_mean(x, w):
    if w <= 1:
        return x.copy()
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad_left = np.full(w // 2, out[0])
    pad_right = np.full(len(x) - len(out) - len(pad_left), out[-1])
    return np.concatenate([pad_left, out, pad_right])


def _zscore(x):
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / s if s > 0 else np.zeros_like(x)


def _skew(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s == 0:
        return 0.0
    return np.mean(((x - m) / s) ** 3)


def _kurtosis_excess(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s == 0:
        return -3.0
    return np.mean(((x - m) / s) ** 4) - 3.0


# ---------- I/O helpers ----------
def infer_sampling_rate_ms(daqtime_series: pd.Series) -> int:
    """Infer sampling rate in Hz (returned as integer) from daqtime in milliseconds."""
    diffs = np.diff(daqtime_series.values.astype(float))
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        warnings.warn("Could not infer sampling rate; defaulting to 1000 Hz.")
        return 1000
    median_ms = float(np.median(diffs))
    sr = 1000.0 / median_ms if median_ms > 0 else 1000.0
    return max(1, int(round(sr)))


# ---------- fallback ECG (NumPy-only) ----------
def _fallback_detect_rpeaks(ecg: np.ndarray, sr: int) -> np.ndarray:
    """
    Very simple R-peak detection:
      1) z-score
      2) positive threshold crossing + local maxima
      3) refractory period ~ 0.3 s
    This is intentionally crude, used only if NeuroKit2 fails.
    """
    z = _zscore(ecg.astype(float))
    thr = np.nanmean(z) + 1.0 * np.nanstd(z)  # conservative threshold
    refr = max(int(0.3 * sr), 1)
    peaks = []
    i = 1
    while i < len(z) - 1:
        if z[i] > thr and z[i] > z[i - 1] and z[i] >= z[i + 1]:
            peaks.append(i)
            i += refr  # skip refractory samples
        else:
            i += 1
    return np.asarray(peaks, dtype=int)


def _rr_from_peaks(peaks: np.ndarray, sr: int) -> np.ndarray:
    """Compute RR intervals (in seconds) from integer peak indices."""
    if peaks is None or len(peaks) < 2 or sr <= 0:
        return np.array([], dtype=float)
    return np.diff(peaks) / float(sr)


def _hr_from_rr(rr_s: np.ndarray) -> np.ndarray:
    """Convert RR intervals (s) to instantaneous heart rate (bpm)."""
    if rr_s is None or rr_s.size == 0:
        return np.array([], dtype=float)
    return 60.0 / rr_s


def _aggregate_array(prefix: str, arr: np.ndarray) -> dict:
    """Basic aggregate stats for an array."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_Mean": np.nan,
            f"{prefix}_SD": np.nan,
            f"{prefix}_Median": np.nan,
            f"{prefix}_Max": np.nan,
            f"{prefix}_Min": np.nan,
            f"{prefix}_Range": np.nan,
            f"{prefix}_Skew": np.nan,
            f"{prefix}_KurtosisExcess": np.nan,
        }
    return {
        f"{prefix}_Mean": float(np.nanmean(arr)),
        f"{prefix}_SD": float(np.nanstd(arr)),
        f"{prefix}_Median": float(np.nanmedian(arr)),
        f"{prefix}_Max": float(np.nanmax(arr)),
        f"{prefix}_Min": float(np.nanmin(arr)),
        f"{prefix}_Range": float(np.nanmax(arr) - np.nanmin(arr)),
        f"{prefix}_Skew": float(_skew(arr)),
        f"{prefix}_KurtosisExcess": float(_kurtosis_excess(arr)),
    }


def ecg_fallback_features(signal: pd.Series, sr: int) -> dict:
    """
    Version-agnostic fallback using basic preprocessing and peak detection.
    Produces core-like HR/HRV features from RR intervals.
    """
    y = np.asarray(signal, dtype=float)

    # Fill NaNs by linear interpolation
    if np.isnan(y).any():
        nans = np.isnan(y)
        y[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), y[~nans])

    # Light smoothing to suppress high-frequency noise
    y_smooth = _rolling_mean(y, max(int(0.05 * sr), 3))  # ~50 ms window

    # Detect peaks
    rpeaks = _fallback_detect_rpeaks(y_smooth, sr)
    rr_s = _rr_from_peaks(rpeaks, sr)
    hr_bpm = _hr_from_rr(rr_s)

    out = {}
    # Heart-rate aggregates
    out.update(_aggregate_array("ECG_HR", hr_bpm))

    # Time-domain HRV features
    if rr_s.size:
        rr_ms = rr_s * 1000.0
        diff_ms = np.diff(rr_ms)
        nn = rr_ms[np.isfinite(rr_ms)]
        out["ECG_HRV_SDNN"] = float(np.nanstd(nn)) if nn.size else np.nan
        out["ECG_HRV_RMSSD"] = float(np.sqrt(np.nanmean(diff_ms ** 2))) if diff_ms.size else np.nan
        out["ECG_HRV_pNN50"] = float(np.nanmean(np.abs(diff_ms) > 50.0) * 100.0) if diff_ms.size else np.nan
    else:
        out["ECG_HRV_SDNN"] = np.nan
        out["ECG_HRV_RMSSD"] = np.nan
        out["ECG_HRV_pNN50"] = np.nan

    # Beat count and beats-per-minute rate over the whole segment
    dur_sec = len(y) / float(sr) if sr > 0 else np.nan
    out["ECG_R_Peaks_N"] = int(rpeaks.size)
    out["ECG_Beats_per_min"] = (rpeaks.size / dur_sec) * 60.0 if (dur_sec and dur_sec > 0) else np.nan

    return out


# ---------- primary ECG extraction with NeuroKit2 + manual extras ----------
def extract_ecg_features(signal: pd.Series, sr: int) -> dict:
    """
    Extended feature extraction:
      1) Try NeuroKit2 to clean ECG, detect R-peaks, and compute HR/HRV (time & frequency).
      2) Add manual aggregates (HR summary, peak count, segment-level BPM).
      3) On failure -> robust NumPy fallback.
    """
    y = np.asarray(signal, dtype=float)

    if nk is not None:
        try:
            # Full pipeline: clean + peaks + rate series
            # ecg_process returns cleaned signal and peaks as a DataFrame plus an info dict
            sig, info = nk.ecg_process(y, sampling_rate=sr)

            # Get R-peaks indices from 'ECG_R_Peaks' column if present, else from info
            if "ECG_R_Peaks" in sig.columns:
                rpeaks_bool = np.asarray(sig["ECG_R_Peaks"], dtype=float)
                rpeaks_idx = np.flatnonzero(np.nan_to_num(rpeaks_bool) > 0.5)
            else:
                # Fallback to dedicated peak detection on cleaned signal
                cleaned = sig["ECG_Clean"] if "ECG_Clean" in sig.columns else y
                _, rpeaks = nk.ecg_peaks(np.asarray(cleaned, dtype=float), sampling_rate=sr)
                rpeaks_idx = np.asarray(rpeaks.get("ECG_R_Peaks", []), dtype=int)

            # Instantaneous rate in bpm (if provided by NK)
            if "ECG_Rate" in sig.columns:
                hr_bpm = np.asarray(sig["ECG_Rate"], dtype=float)
                hr_bpm = hr_bpm[np.isfinite(hr_bpm)]
                hr_bpm = hr_bpm[hr_bpm > 0]
            else:
                rr_s = _rr_from_peaks(rpeaks_idx, sr)
                hr_bpm = _hr_from_rr(rr_s)

            # Time-domain HRV from R-peaks
            # nk.hrv_time expects a dict with "ECG_R_Peaks"
            hrv_time = nk.hrv_time({"ECG_R_Peaks": rpeaks_idx}, sampling_rate=sr, show=False)
            # Frequency-domain HRV (Welch by default)
            hrv_freq = nk.hrv_frequency({"ECG_R_Peaks": rpeaks_idx}, sampling_rate=sr, psd_method="welch", show=False)

            # Non-linear (request only stable features to avoid DFA_alpha2 warning)
            try:
                hrv_nl = nk.hrv_nonlinear({"ECG_R_Peaks": rpeaks_idx},
                                          sampling_rate=sr,
                                          show=False,
                                          features=["SD1", "SD2", "SD1SD2", "SampEn"])
            except Exception:
                hrv_nl = pd.DataFrame(index=[0])

            # Consolidate NK features into flat dict with 'ECG_' prefix
            base = {}
            for df, pref in [(hrv_time, "ECG_HRV"), (hrv_freq, "ECG_HRV"), (hrv_nl, "ECG_HRV")]:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    for k, v in df.iloc[0].to_dict().items():
                        base[f"{pref}_{k}"] = v

            # Manual aggregates on HR series
            base.update(_aggregate_array("ECG_HR", hr_bpm))

            # Counts and rates
            dur_sec = len(y) / float(sr) if sr > 0 else np.nan
            base["ECG_R_Peaks_N"] = int(len(rpeaks_idx))
            base["ECG_Beats_per_min"] = (len(rpeaks_idx) / dur_sec) * 60.0 if (dur_sec and dur_sec > 0) else np.nan

            # Add optional cleaned/raw summary if available (kept minimal)
            if "ECG_Clean" in sig.columns:
                base.update(_aggregate_array("ECG_Clean", sig["ECG_Clean"].values))

            return base

        except Exception as e:
            warnings.warn(f"NeuroKit2 ECG pipeline failed; using NumPy fallback. Reason: {e}")

    # ---- Fallback path ----
    return ecg_fallback_features(signal, sr)


# ========== CORE: keep only canonical ECG features ==========
# We define canonical core features + aliases (to handle NK version/label differences)
# Choose a concise, modeling-friendly subset.
CORE_FEATURES_MAP = {
    # Heart rate summary (instantaneous HR aggregates)
    "ECG_HR_Mean": ["ECG_HR_Mean"],
    "ECG_HR_SD": ["ECG_HR_SD"],
    "ECG_HR_Median": ["ECG_HR_Median"],
    "ECG_HR_Max": ["ECG_HR_Max"],
    "ECG_HR_Min": ["ECG_HR_Min"],

    # Beat counts / rate over the segment
    "ECG_R_Peaks_N": ["ECG_R_Peaks_N"],
    "ECG_Beats_per_min": ["ECG_Beats_per_min"],

    # Time-domain HRV (standard names used by NeuroKit2 hrv_time)
    # SDNN: Standard deviation of NN intervals (ms)
    "ECG_HRV_SDNN": ["ECG_HRV_SDNN", "ECG_HRV_HRV_SDNN", "ECG_HRV_Time_SDNN"],
    # RMSSD: Root mean square of successive differences (ms)
    "ECG_HRV_RMSSD": ["ECG_HRV_RMSSD", "ECG_HRV_HRV_RMSSD", "ECG_HRV_Time_RMSSD"],
    # pNN50: Percentage of successive intervals > 50 ms
    "ECG_HRV_pNN50": ["ECG_HRV_pNN50", "ECG_HRV_HRV_pNN50", "ECG_HRV_Time_pNN50"],

    # Frequency-domain HRV (Welch)
    # LF power (ms^2), HF power (ms^2), LF/HF ratio
    "ECG_HRV_LF": ["ECG_HRV_LF", "ECG_HRV_Frequency_LF"],
    "ECG_HRV_HF": ["ECG_HRV_HF", "ECG_HRV_Frequency_HF"],
    "ECG_HRV_LFHF": ["ECG_HRV_LFHF", "ECG_HRV_Frequency_LFHF"],

    # Optional non-linear (if computed)
    "ECG_HRV_SD1": ["ECG_HRV_SD1", "ECG_HRV_Nonlinear_SD1"],
    "ECG_HRV_SD2": ["ECG_HRV_SD2", "ECG_HRV_Nonlinear_SD2"],
}


def _select_core_features(feature_dict: dict) -> dict:
    """Pick the first available alias per canonical key; if missing, use NaN."""
    selected = {}
    for canon, aliases in CORE_FEATURES_MAP.items():
        val = np.nan
        for a in aliases:
            if a in feature_dict and np.any(pd.notna(feature_dict[a])):
                val = feature_dict[a]
                break
        selected[canon] = val
    return selected


# ---------- per-file processing ----------
def process_subject_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    for col in ("daqtime", "video", "ecg"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {os.path.basename(filepath)}")

    sr = infer_sampling_rate_ms(df["daqtime"])  # integer Hz
    subj = os.path.splitext(os.path.basename(filepath))[0]

    rows = []
    for vid, g in df.groupby("video"):
        g = g.reset_index(drop=True)
        full_feats = extract_ecg_features(g["ecg"], sr)
        core_feats = _select_core_features(full_feats)  # reduce to core
        features = {"subject": subj, "video": vid, "sampling_rate_hz": int(sr)}
        features.update(core_feats)
        rows.append(features)

    return pd.DataFrame(rows)


# ---------- main ----------
def main():
    pattern = os.path.join(BASE_DIR, "sub_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files found in {BASE_DIR}")

    all_features = []
    for fp in files:
        print(f"Processing {os.path.basename(fp)} ...")
        try:
            feats = process_subject_file(fp)
            all_features.append(feats)
        except Exception as e:
            warnings.warn(f"Failed on {os.path.basename(fp)}: {e}")

    features_df = pd.concat(all_features, ignore_index=True).sort_values(["subject", "video"], ignore_index=True)
    out_csv = "case_features_ecg.csv"
    features_df.to_csv(out_csv, index=False)
    print(f"\nSaved ECG features to: {out_csv}")
    print(f"Shape: {features_df.shape}")

    # Quick visualization: prefer mean HR; else SDNN
    plot_cols = [c for c in features_df.columns if c.endswith("ECG_HR_Mean")]
    if not plot_cols:
        plot_cols = [c for c in features_df.columns if "ECG_HRV_SDNN" in c]
    if plot_cols:
        col = plot_cols[0]
        plt.figure(figsize=(10, 5))
        features_df.groupby("video")[col].mean().sort_index().plot(kind="bar")
        plt.title(f"Mean {col} per Video")
        plt.xlabel("Video ID")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()
    else:
        print("No suitable ECG columns found for plotting.")


if __name__ == "__main__":
    main()
