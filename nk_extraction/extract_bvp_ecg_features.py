# =========================
# CASE Dataset BVP & ECG Feature Extraction (warning-free, robust)
# =========================
# Requirements:
#   pip install neurokit2 pandas numpy
# Notes:
# - Keine Warnings: DFA wird NICHT berechnet, Frequenz-HRV hat Welch→Lomb Fallback,
#   alle Exceptions werden intern abgefangen (keine warnings.warn-Aufrufe).
# - Output: case_features_cardio.csv (eine Zeile pro subject × video)
# =========================

import os
import glob
import numpy as np
import pandas as pd
import warnings
from typing import Optional

# Optional: alle NeuroKit2-Warnings unterdrücken, falls intern doch welche entstehen
warnings.filterwarnings("ignore", module="neurokit2")

try:
    import neurokit2 as nk
except Exception:
    nk = None  # Wir fallen dann auf NumPy-Feature-Set zurück

# ---------- USER SETTINGS ----------
BASE_DIR = r"../case_dataset-master/data/interpolated/physiological"
OUT_CSV = "case_features_cardio.csv"
# -----------------------------------

# ---------- helpers ----------

def infer_sampling_rate_ms(daqtime_series: pd.Series) -> int:
    """Samplingrate (Hz, int) aus 'daqtime' in Millisekunden schätzen."""
    vals = daqtime_series.values.astype(float)
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 1000
    median_ms = float(np.median(diffs))
    sr = 1000.0 / median_ms if median_ms > 0 else 1000.0
    return max(1, int(round(sr)))


def detect_signal_column(df: pd.DataFrame, candidates) -> Optional[str]:
    """Erste passende Spalte (case-insensitive) zurückgeben."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
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
    """Einfacher Peak-Finder im z-transformierten Signal mit Mindestabstand."""
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
    """Basale HR/HRV-Deskriptoren ohne NeuroKit2 (für BVP & ECG)."""
    y = _naninterp(np.asarray(signal, dtype=float))
    sm = _rolling_mean(y, max(int(0.15 * sr), 3))  # ~150 ms Glättung

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
            f"{label_prefix}_RMSSD_ms": float(np.sqrt(np.nanmean(np.square(np.diff(ibi_ms))))) if ibi_ms.size >= 2 else np.nan,
        })
    else:
        out.update({
            f"{label_prefix}_HR_Mean": np.nan,
            f"{label_prefix}_HR_SD": np.nan,
            f"{label_prefix}_IBI_Mean_ms": np.nan,
            f"{label_prefix}_RMSSD_ms": np.nan,
        })
    return out


# ---------- NeuroKit2-based extractors (ohne DFA, warning-frei) ----------

def _safe_hr_from_rate(sig_df: pd.DataFrame, rate_prefix: str) -> dict:
    """Extrahiere HR-Mittel/SD aus der *_Rate-Spalte, ohne Exceptions zu werfen."""
    out = {}
    rate_col = f"{rate_prefix}_Rate"
    if rate_col in sig_df.columns:
        out[f"{rate_prefix}_HR_Mean"] = float(np.nanmean(sig_df[rate_col]))
        out[f"{rate_prefix}_HR_SD"] = float(np.nanstd(sig_df[rate_col]))
    return out


def extract_bvp_features(signal: pd.Series, sr: int) -> dict:
    label = "BVP"
    if nk is not None:
        try:
            sig, info = nk.ppg_process(np.asarray(signal, dtype=float), sampling_rate=sr)

            # HR aus PPG_Rate
            out = _safe_hr_from_rate(sig, "PPG")

            # Peaks bestimmen (für Beat_Count)
            if isinstance(info, dict) and "PPG_Peaks" in info and info["PPG_Peaks"] is not None:
                peaks_idx = np.asarray(info["PPG_Peaks"], dtype=int)
            else:
                peaks_idx = np.where(np.asarray(sig.get("PPG_Peaks", np.zeros(len(sig)))) == 1)[0]

            peaks_idx = np.unique(peaks_idx)
            peaks_idx.sort()

            out.update({
                f"{label}_Beat_Count": int(peaks_idx.size)
            })

            # Ein paar robuste Signalstats aus dem verarbeiteten PPG-Signal (optional)
            if "PPG_Clean" in sig.columns:
                out[f"{label}_Signal_Mean"] = float(np.nanmean(sig["PPG_Clean"]))
                out[f"{label}_Signal_SD"] = float(np.nanstd(sig["PPG_Clean"]))
                out[f"{label}_Signal_Max"] = float(np.nanmax(sig["PPG_Clean"]))

            return out
        except Exception:
            pass  # ohne Warning – wir fallen zurück
    return cardio_fallback_features(signal, sr, label_prefix=label)


def extract_ecg_features(signal: pd.Series, sr: int) -> dict:
    """ECG-Features ohne DFA und ohne Warnings.
    - Zeitbereich: SDNN, RMSSD, MeanNN, pNN50
    - Frequenzbereich: LF, HF, LF/HF (Welch→Lomb Fallback)
    - HR aus Rate
    """
    label = "ECG"
    if nk is not None:
        try:
            sig, info = nk.ecg_process(np.asarray(signal, dtype=float), sampling_rate=sr)

            # HR aus ECG_Rate
            out = _safe_hr_from_rate(sig, "ECG")

            # R-Peaks holen
            if isinstance(info, dict) and "ECG_R_Peaks" in info and info["ECG_R_Peaks"] is not None:
                rpeaks_idx = np.asarray(info["ECG_R_Peaks"], dtype=int)
            else:
                rpeaks_idx = np.where(np.asarray(sig.get("ECG_R_Peaks", np.zeros(len(sig)))) == 1)[0]

            # sortieren & deduplizieren
            rpeaks_idx = np.unique(rpeaks_idx)
            rpeaks_idx.sort()
            peaks = {"ECG_R_Peaks": rpeaks_idx}

            # Mindestanzahl prüfen
            if rpeaks_idx.size >= 3:
                # Zeitbereich (robust)
                try:
                    hrv_time = nk.hrv_time(peaks, sampling_rate=sr, show=False)
                except Exception:
                    hrv_time = None

                # Frequenzbereich: Welch → Lomb Fallback
                hrv_freq = None
                if rpeaks_idx.size >= 5:
                    try:
                        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sr, show=False, method="welch")
                    except Exception:
                        try:
                            hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sr, show=False, method="lomb")
                        except Exception:
                            hrv_freq = None

                # Nur stabile Keys übernehmen
                keep_keys = {
                    "HRV_SDNN", "HRV_RMSSD", "HRV_MeanNN", "HRV_pNN50",
                    "HRV_LF", "HRV_HF", "HRV_LFHF"
                }

                def _pick(df: Optional[pd.DataFrame], prefix: str):
                    if df is None or df.empty:
                        return {}
                    d = {}
                    row = df.iloc[0].to_dict()
                    for k, v in row.items():
                        if k in keep_keys:
                            d[f"{label}_{k}"] = float(v) if v is not None else np.nan
                    return d

                out.update(_pick(hrv_time, "time"))
                out.update(_pick(hrv_freq, "freq"))

            # Ein paar robuste Signalstats aus dem verarbeiteten ECG-Signal (optional)
            if "ECG_Clean" in sig.columns:
                out[f"{label}_Signal_Mean"] = float(np.nanmean(sig["ECG_Clean"]))
                out[f"{label}_Signal_SD"] = float(np.nanstd(sig["ECG_Clean"]))
                out[f"{label}_Signal_Max"] = float(np.nanmax(sig["ECG_Clean"]))

            return out
        except Exception:
            pass  # ohne Warning – wir fallen zurück
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
            # Keine Warnung ausgeben – einfach überspringen
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
            # Keine Warnings – aber wir zeigen die Datei an, die Probleme machte
            print(f"[Skip] {os.path.basename(fp)} due to error: {e}")

    if not all_features:
        raise SystemExit("No features extracted.")

    features_df = pd.concat(all_features, ignore_index=True)

    # Sicherstellen: doppelte Spalten vermeiden (falls Keys kollidieren)
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]

    features_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved cardio features to: {OUT_CSV}")
    print(f"Shape: {features_df.shape}")


if __name__ == "__main__":
    main()
