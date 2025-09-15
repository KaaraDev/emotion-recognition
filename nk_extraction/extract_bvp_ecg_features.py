# =========================
# CASE Dataset BVP & ECG Feature Extraction with NeuroKit2 (robust)
# =========================
# Requirements:
#   pip install neurokit2 pandas numpy matplotlib
# (Optional) For faster peak detection / extra metrics you may also: pip install scipy
# =========================

import os
import glob
import warnings
import numpy as np
import pandas as pd

try:
    import neurokit2 as nk
except Exception:
    nk = None
    warnings.warn("NeuroKit2 not available; will use NumPy-only fallbacks.")

# ---------- USER SETTINGS ----------
BASE_DIR = r"../case_dataset-master/data/interpolated/physiological"
OUT_CSV = "case_features_cardio.csv"  # merged BVP+ECG features per subject/video


# -----------------------------------

# ---------- helpers ----------

def infer_sampling_rate_ms(daqtime_series: pd.Series) -> int:
    """Infer sampling rate in Hz (returned as integer) from 'daqtime' in milliseconds."""
    diffs = np.diff(daqtime_series.values.astype(float))
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        warnings.warn("Could not infer sampling rate; defaulting to 1000 Hz.")
        return 1000
    median_ms = float(np.median(diffs))
    sr = 1000.0 / median_ms if median_ms > 0 else 1000.0
    return max(1, int(round(sr)))


def detect_signal_column(df: pd.DataFrame, candidates) -> str | None:
    """Return the first matching column among candidate names (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


# ---------- Fallbacks (NumPy) ----------

def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad_left = np.full(w // 2, out[0])
    pad_right = np.full(len(x) - len(out) - len(pad_left), out[-1])
    return np.concatenate([pad_left, out, pad_right])


def _naninterp(y: np.ndarray) -> np.ndarray:
    if np.isnan(y).any():
        nans = np.isnan(y)
        y[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), y[~nans])
    return y


def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / s if s > 0 else np.zeros_like(x)


def _simple_peak_indices(x: np.ndarray, sr: int, zthr: float = 1.0, min_dist_s: float = 0.35) -> np.ndarray:
    """Very crude peak finder on z-scored signal with min distance (seconds)."""
    z = _zscore(x)
    thr = np.nanmean(z) + zthr * np.nanstd(z)
    md = max(int(sr * min_dist_s), 1)
    peaks = []
    for i in range(1, len(z) - 1):
        if z[i] > thr and z[i] > z[i - 1] and z[i] > z[i + 1]:
            if not peaks or (i - peaks[-1]) >= md:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def cardio_fallback_features(signal: pd.Series, sr: int, label_prefix: str) -> dict:
    """Compute minimal HR/HRV-like descriptors without NeuroKit2.
    Works for both BVP and ECG; uses crude peaks to estimate beats.
    """
    y = _naninterp(np.asarray(signal, dtype=float))
    # light smoothing for BVP/ECG-like signals
    win = max(int(0.15 * sr), 3)  # 150 ms
    sm = _rolling_mean(y, win)

    peaks = _simple_peak_indices(sm, sr, zthr=0.8, min_dist_s=0.35)
    out = {
        f"{label_prefix}_Beat_Count": int(peaks.size),
        f"{label_prefix}_Signal_Mean": float(np.nanmean(sm)),
        f"{label_prefix}_Signal_SD": float(np.nanstd(sm)),
        f"{label_prefix}_Signal_Max": float(np.nanmax(sm)),
    }
    if peaks.size >= 2:
        ibi_samples = np.diff(peaks).astype(float)
        ibi_ms = (1000.0 * ibi_samples) / float(sr)
        hr = 60000.0 / ibi_ms  # bpm
        out.update({
            f"{label_prefix}_HR_Mean": float(np.nanmean(hr)),
            f"{label_prefix}_HR_SD": float(np.nanstd(hr)),
            f"{label_prefix}_IBI_Mean_ms": float(np.nanmean(ibi_ms)),
            f"{label_prefix}_RMSSD_ms": float(
                np.sqrt(np.nanmean(np.square(np.diff(ibi_ms))))) if ibi_ms.size >= 2 else np.nan,
        })
    else:
        out.update({
            f"{label_prefix}_HR_Mean": np.nan,
            f"{label_prefix}_HR_SD": np.nan,
            f"{label_prefix}_IBI_Mean_ms": np.nan,
            f"{label_prefix}_RMSSD_ms": np.nan,
        })
    return out


# ---------- NeuroKit2-based extractors ----------

def extract_bvp_features(signal: pd.Series, sr: int) -> dict:
    label = "BVP"
    if nk is not None:
        try:
            sig, info = nk.ppg_process(np.asarray(signal, dtype=float), sampling_rate=sr)
            feats = nk.ppg_analyze(sig, sampling_rate=sr)
            out = {f"{label}_{k}": v for k, v in feats.iloc[0].to_dict().items()}
            # Best-effort HRV from PPG peaks (not as reliable as ECG but useful)
            if "PPG_Peaks" in info:
                peaks = info["PPG_Peaks"]
            else:
                # if not provided, try from the processed signals
                peaks = np.where(sig.get("PPG_Peaks", np.zeros(len(signal))) == 1)[0]
            if peaks is not None and len(peaks) > 3:
                ibi = np.diff(peaks) / sr
                hr = 60.0 / ibi
                out.update({
                    f"{label}_HR_Mean": float(np.nanmean(hr)),
                    f"{label}_HR_SD": float(np.nanstd(hr)),
                })
            return out
        except Exception as e:
            warnings.warn(f"NeuroKit2 PPG pipeline failed; using fallback. Reason: {e}")
    return cardio_fallback_features(signal, sr, label_prefix=label)


def extract_ecg_features(signal: pd.Series, sr: int) -> dict:
    label = "ECG"
    if nk is not None:
        try:
            sig, info = nk.ecg_process(np.asarray(signal, dtype=float), sampling_rate=sr)
            feats = nk.ecg_analyze(sig, sampling_rate=sr)
            out = {f"{label}_{k}": v for k, v in feats.iloc[0].to_dict().items()}

            # HRV features from R-peaks
            try:
                # nk.hrv expects either the 'peaks' dict or the signal dataframe
                hrv = nk.hrv(info, sampling_rate=sr, show=False)
            except Exception:
                hrv = nk.hrv(nk.ecg_peaks(np.asarray(signal, dtype=float), sampling_rate=sr)[1], sampling_rate=sr,
                             show=False)
            for k, v in hrv.iloc[0].to_dict().items():
                out[f"{label}_HRV_{k}"] = v
            return out
        except Exception as e:
            warnings.warn(f"NeuroKit2 ECG pipeline failed; using fallback. Reason: {e}")
    return cardio_fallback_features(signal, sr, label_prefix=label)


# ---------- per-file processing ----------

def process_subject_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    for col in ("daqtime", "video"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {os.path.basename(filepath)}")

    sr = infer_sampling_rate_ms(df["daqtime"])  # integer Hz
    subj = os.path.splitext(os.path.basename(filepath))[0]

    # Try common column names for BVP/ECG
    bvp_col = detect_signal_column(df, ["bvp", "ppg", "BVP", "PPG", "blood_volume_pulse"])
    ecg_col = detect_signal_column(df, ["ecg", "ECG", "electrocardiogram"])

    rows = []
    for vid, g in df.groupby("video"):
        g = g.reset_index(drop=True)
        base = {"subject": subj, "video": vid, "sampling_rate_hz": int(sr)}
        if bvp_col is not None:
            base.update(extract_bvp_features(g[bvp_col], sr))
        if ecg_col is not None:
            base.update(extract_ecg_features(g[ecg_col], sr))
        if bvp_col is None and ecg_col is None:
            warnings.warn(f"No BVP/ECG columns found in {os.path.basename(filepath)}. Skipping.")
            continue
        rows.append(base)

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
            if not feats.empty:
                all_features.append(feats)
        except Exception as e:
            warnings.warn(f"Failed on {os.path.basename(fp)}: {e}")

    if not all_features:
        raise SystemExit("No features extracted.")

    features_df = pd.concat(all_features, ignore_index=True)
    features_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved cardio features to: {OUT_CSV}")
    print(f"Shape: {features_df.shape}")


if __name__ == "__main__":
    main()
