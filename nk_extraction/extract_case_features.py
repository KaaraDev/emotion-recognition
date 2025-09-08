# =========================
# CASE Dataset GSR Feature Extraction with NeuroKit2 (robust)
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

try:
    import neurokit2 as nk
except Exception:
    nk = None
    warnings.warn("NeuroKit2 not available; will try NumPy fallback only.")

# ---------- USER SETTINGS ----------
BASE_DIR = r"../case_dataset-master/data/interpolated/physiological"


# -----------------------------------

def infer_sampling_rate_ms(daqtime_series: pd.Series) -> int:
    """Infer sampling rate in Hz (return as *integer*) from daqtime in milliseconds."""
    diffs = np.diff(daqtime_series.values.astype(float))
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        warnings.warn("Could not infer sampling rate; defaulting to 1000 Hz.")
        return 1000
    median_ms = float(np.median(diffs))
    sr = 1000.0 / median_ms if median_ms > 0 else 1000.0
    return max(1, int(round(sr)))  # <-- make sure it's an integer


def _rolling_mean(x, w):
    if w <= 1:
        return x.copy()
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out = (cumsum[w:] - cumsum[:-w]) / float(w)
    # pad to original length
    pad_left = np.full(w // 2, out[0])
    pad_right = np.full(len(x) - len(out) - len(pad_left), out[-1])
    return np.concatenate([pad_left, out, pad_right])


def _zscore(x):
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / s if s > 0 else np.zeros_like(x)


def eda_fallback_features(signal: pd.Series, sr: int) -> dict:
    """Version-agnostic fallback: simple clean + SCR-ish features with NumPy only."""
    y = np.asarray(signal, dtype=float)

    # fill NaNs by linear interpolation
    if np.isnan(y).any():
        nans = np.isnan(y)
        y[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), y[~nans])

    # very simple baseline removal (10s moving average), then smoothing (0.5s)
    baseline = _rolling_mean(y, max(sr * 10, 5))
    detr = y - baseline
    cleaned = _rolling_mean(detr, max(int(sr * 0.5), 3))

    # stats
    out = {
        "EDA_Clean_Mean": float(np.nanmean(cleaned)),
        "EDA_Clean_SD": float(np.nanstd(cleaned)),
        "EDA_Clean_Max": float(np.nanmax(cleaned)),
    }
    # slope
    if np.isfinite(cleaned).sum() > 1:
        xidx = np.arange(len(cleaned), dtype=float)
        coeffs = np.polyfit(xidx, cleaned, 1)
        out["EDA_Clean_Slope"] = float(coeffs[0])
    else:
        out["EDA_Clean_Slope"] = np.nan

    # crude peak count on z-scored cleaned signal
    z = _zscore(cleaned)
    thr = np.nanmean(z) + 0.5 * np.nanstd(z)
    min_dist = max(sr, 1)  # at least 1s apart
    peaks = []
    for i in range(1, len(z) - 1):
        if z[i] > thr and z[i] > z[i - 1] and z[i] > z[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
    peaks = np.array(peaks, dtype=int)

    dur_sec = len(cleaned) / float(sr) if sr > 0 else np.nan
    out["EDA_SCR_Count"] = int(peaks.size)
    out["EDA_SCRs_per_min"] = (peaks.size / dur_sec) * 60.0 if dur_sec and dur_sec > 0 else np.nan
    out["EDA_SCR_Amp_Mean_z"] = float(np.nanmean(z[peaks])) if peaks.size else np.nan
    out["EDA_SCR_Amp_Max_z"] = float(np.nanmax(z[peaks])) if peaks.size else np.nan
    return out


def extract_eda_features(signal: pd.Series, sr: int) -> dict:
    """Try NeuroKit2 pipeline (with integer sr). If it fails, use NumPy fallback."""
    if nk is not None:
        try:
            sig, info = nk.eda_process(np.asarray(signal, dtype=float), sampling_rate=sr)
            feats = nk.eda_analyze(sig, sampling_rate=sr)
            return {f"EDA_{k}": v for k, v in feats.iloc[0].to_dict().items()}
        except Exception as e:
            warnings.warn(f"NeuroKit2 EDA pipeline failed; using NumPy fallback. Reason: {e}")

    # fallback
    return eda_fallback_features(signal, sr)


def process_subject_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    for col in ("daqtime", "video", "gsr"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {os.path.basename(filepath)}")

    sr = infer_sampling_rate_ms(df["daqtime"])  # integer Hz
    subj = os.path.splitext(os.path.basename(filepath))[0]

    rows = []
    for vid, g in df.groupby("video"):
        g = g.reset_index(drop=True)
        features = {"subject": subj, "video": vid, "sampling_rate_hz": int(sr)}
        features.update(extract_eda_features(g["gsr"], sr))
        rows.append(features)

    return pd.DataFrame(rows)


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

    features_df = pd.concat(all_features, ignore_index=True)
    out_csv = "case_features_gsr.csv"
    features_df.to_csv(out_csv, index=False)
    print(f"\nSaved GSR features to: {out_csv}")
    print(f"Shape: {features_df.shape}")

    # Quick visualization: prefer SCR rate; else mean of cleaned EDA
    plot_cols = [c for c in features_df.columns if "SCRs_per_min" in c]
    if not plot_cols:
        plot_cols = [c for c in features_df.columns if "EDA_Clean_Mean" in c]
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
        print("No suitable EDA columns found for plotting.")


if __name__ == "__main__":
    main()
