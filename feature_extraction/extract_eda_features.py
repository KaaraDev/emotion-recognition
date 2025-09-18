# =========================
# CASE Dataset GSR Feature Extraction with NeuroKit2 (extended & robust, CORE only)
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


def _auc(y, dt):
    # Trapezoidal integral; returns "area under the curve"
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return float(np.trapezoid(y, dx=dt))


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


# ---------- feature builders ----------
BASIC_STAT_KEYS = [
    ("EDA_Min", np.nanmin),
    ("EDA_Max", np.nanmax),
    ("EDA_Mean", np.nanmean),
    ("EDA_SD", np.nanstd),
    ("EDA_Median", np.nanmedian),
]


def _basic_stats(prefix, arr):
    out = {}
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        # produce keys with NaN
        for k, _ in BASIC_STAT_KEYS:
            out[f"{prefix}_{k.split('_', 1)[1]}"] = np.nan
        out[f"{prefix}_Range"] = np.nan
        out[f"{prefix}_Skew"] = np.nan
        out[f"{prefix}_KurtosisExcess"] = np.nan
        return out
    for k, fn in BASIC_STAT_KEYS:
        out[f"{prefix}_{k.split('_', 1)[1]}"] = float(fn(x))
    out[f"{prefix}_Range"] = float(np.nanmax(x) - np.nanmin(x))
    out[f"{prefix}_Skew"] = float(_skew(x))
    out[f"{prefix}_KurtosisExcess"] = float(_kurtosis_excess(x))
    return out


def _scr_morphology_from_sig(sig):
    """Collect SCR morphology arrays if present in NK signal dataframe."""
    out = {}
    for name in ["SCR_Amplitude", "SCR_RiseTime", "SCR_RecoveryTime", "SCR_Width", "SCR_Latency"]:
        if name in sig.columns:
            vals = np.asarray(sig[name], dtype=float)
            vals = vals[np.isfinite(vals)]
            # keep positive-only morphology
            vals = vals[vals > 0] if vals.size else vals
            if vals.size:
                out[name] = vals
    return out


def _aggregate(name, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{name}_Mean": np.nan,
            f"{name}_SD": np.nan,
            f"{name}_Median": np.nan,
            f"{name}_Max": np.nan,
            f"{name}_Min": np.nan,
        }
    return {
        f"{name}_Mean": float(np.nanmean(arr)),
        f"{name}_SD": float(np.nanstd(arr)),
        f"{name}_Median": float(np.nanmedian(arr)),
        f"{name}_Max": float(np.nanmax(arr)),
        f"{name}_Min": float(np.nanmin(arr)),
    }


def eda_fallback_features(signal: pd.Series, sr: int) -> dict:
    """Version-agnostic fallback: simple clean + SCR-like features with NumPy only."""
    y = np.asarray(signal, dtype=float)

    # fill NaNs by linear interpolation
    if np.isnan(y).any():
        nans = np.isnan(y)
        y[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), y[~nans])

    # very simple baseline removal (10s moving average), then smoothing (0.5s)
    baseline = _rolling_mean(y, max(sr * 10, 5))
    detr = y - baseline
    cleaned = _rolling_mean(detr, max(int(sr * 0.5), 3))

    dt = 1.0 / float(sr) if sr > 0 else np.nan

    # Global stats & trends
    out = {}
    out.update(_basic_stats("EDA_Clean", cleaned))
    out["EDA_Clean_AUC"] = _auc(cleaned, dt)

    # Slope (global trend)
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

    out["EDA_SCR_Peaks_N"] = int(peaks.size)
    out["EDA_SCRs_per_min"] = (peaks.size / dur_sec) * 60.0 if dur_sec and dur_sec > 0 else np.nan
    out["EDA_SCR_Peaks_Amplitude_Mean_z"] = float(np.nanmean(z[peaks])) if peaks.size else np.nan
    out["EDA_SCR_Peaks_Amplitude_Max_z"] = float(np.nanmax(z[peaks])) if peaks.size else np.nan

    # Tonic (approximated baseline) stats
    out.update(_basic_stats("EDA_TonicApprox", baseline))

    return out


def extract_eda_features(signal: pd.Series, sr: int) -> dict:
    """
    Extended feature extraction:
      1) Try NeuroKit2 (eda_process + eda_analyze + eda_intervalrelated)
      2) Add manual extras (stats, AUC, slopes, morphology aggregates, SCR rate)
      3) On failure -> robust NumPy fallback
    """
    y = np.asarray(signal, dtype=float)
    dt = 1.0 / float(sr) if sr > 0 else np.nan

    # ---- Primary path: NeuroKit2 ----
    if nk is not None:
        try:
            sig, info = nk.eda_process(y, sampling_rate=sr)  # sig is a DataFrame
            feats = nk.eda_analyze(sig, sampling_rate=sr)  # one-row DataFrame

            # Optional: richer set for fixed windows
            try:
                inter = nk.eda_intervalrelated(sig, sampling_rate=sr)
                feats = pd.concat([feats, inter], axis=1)
            except Exception:
                pass

            # Convert to dict and prefix 'EDA_'
            base = {f"EDA_{k}": v for k, v in feats.iloc[0].to_dict().items()}

            # ---- Manual extras on raw EDA & NK components ----
            base.update(_basic_stats("EDA", y))
            base["EDA_AUC"] = _auc(y, dt)

            if np.isfinite(y).sum() > 1:
                xidx = np.arange(len(y), dtype=float)
                coeffs = np.polyfit(xidx, y, 1)
                base["EDA_Slope"] = float(coeffs[0])
            else:
                base["EDA_Slope"] = np.nan

            if "EDA_Tonic" in sig.columns:
                base.update(_basic_stats("EDA_Tonic", sig["EDA_Tonic"].values))

            if "EDA_Phasic" in sig.columns:
                base.update(_basic_stats("EDA_Phasic", sig["EDA_Phasic"].values))

            # SCR rate per minute
            n_peaks = None
            for cand in ["SCR_Peaks_N", "EDA_SCR_Peaks_N", "EDA_Peaks_N"]:
                if f"EDA_{cand}" in base:
                    try:
                        n_peaks = int(base[f"EDA_{cand}"])
                        break
                    except Exception:
                        pass
            if n_peaks is None and "SCR_Peaks" in sig.columns:
                val = np.nansum(np.asarray(sig["SCR_Peaks"], dtype=float))
                n_peaks = int(val) if np.isfinite(val) else None

            dur_sec = len(y) / float(sr) if sr > 0 else np.nan
            base["EDA_SCRs_per_min"] = (n_peaks / dur_sec) * 60.0 if (
                    n_peaks is not None and dur_sec and dur_sec > 0) else np.nan

            # Morphology aggregates (if available)
            morph = _scr_morphology_from_sig(sig)
            for key, arr in morph.items():
                base.update(_aggregate(f"EDA_{key}", arr))

            return base

        except Exception as e:
            warnings.warn(f"NeuroKit2 EDA pipeline failed; using NumPy fallback. Reason: {e}")

    # ---- Fallback path ----
    return eda_fallback_features(signal, sr)


# We define canonical core features + aliases (to handle NK2 version differences)
CORE_FEATURES_MAP = {
    # global EDA
    "EDA_Mean": ["EDA_Mean"],
    "EDA_SD": ["EDA_SD"],
    "EDA_AUC": ["EDA_AUC", "EDA_Clean_AUC"],  # fallback name
    "EDA_Slope": ["EDA_Slope", "EDA_Clean_Slope"],

    # tonic/phasic (if available)
    "EDA_Tonic_Mean": ["EDA_Tonic_Mean"],
    "EDA_Tonic_SD": ["EDA_Tonic_SD"],
    "EDA_Phasic_Mean": ["EDA_Phasic_Mean"],
    "EDA_Phasic_SD": ["EDA_Phasic_SD"],

    # SCR summary
    "EDA_SCR_Peaks_N": ["EDA_SCR_Peaks_N", "EDA_Peaks_N"],  # depending on NK version
    "EDA_SCRs_per_min": ["EDA_SCRs_per_min"],

    # SCR amplitudes (from intervalrelated or morphology)
    "EDA_SCR_Amplitude_Mean": [
        "EDA_SCR_Amplitude_Mean",  # from morphology aggregate (our aggregate)
        "EDA_SCR_Peaks_Amplitude_Mean"  # from eda_intervalrelated
    ],
    "EDA_SCR_Amplitude_Max": [
        "EDA_SCR_Amplitude_Max",  # from morphology aggregate
        "EDA_SCR_Peaks_Amplitude_Max"  # if available
    ],

    # useful model-based NK2 outputs (if available)
    "EDA_Sympathetic": ["EDA_EDA_Sympathetic"],
    "EDA_SympatheticN": ["EDA_EDA_SympatheticN"],
    "EDA_Autocorrelation": ["EDA_EDA_Autocorrelation"],
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
    for col in ("daqtime", "video", "gsr"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {os.path.basename(filepath)}")

    sr = infer_sampling_rate_ms(df["daqtime"])  # integer Hz
    subj = os.path.splitext(os.path.basename(filepath))[0]

    rows = []
    for vid, g in df.groupby("video"):
        g = g.reset_index(drop=True)
        full_feats = extract_eda_features(g["gsr"], sr)
        core_feats = _select_core_features(full_feats)  # >>> NEW: reduce to core
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
    out_csv = "case_features_gsr.csv"
    features_df.to_csv(out_csv, index=False)
    print(f"\nSaved GSR features to: {out_csv}")
    print(f"Shape: {features_df.shape}")

    # Quick visualization: prefer SCR rate; else mean EDA
    plot_cols = [c for c in features_df.columns if "SCRs_per_min" in c]
    if not plot_cols:
        plot_cols = [c for c in features_df.columns if "EDA_Mean" in c]
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
