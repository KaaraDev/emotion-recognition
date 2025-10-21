# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import neurokit2 as nk
import numpy as np
import pandas as pd

try:
    import heartpy as hp
except Exception as e:
    raise ImportError("Bitte heartpy installieren: pip install heartpy") from e


@dataclass
class ExtractConfig:
    base_path: Path
    subjects: List[int]
    window_size_s: float = 10.0
    step_size_s: float = 1.0
    fs_phys: float = 20.0  # Fallback, echte fs wird geschätzt
    out_file: Path = Path("outputs/features_case_bvp_gsr_skt.csv")


# ------------------------- Utilities (robust) -------------------------

def _find_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in df.columns}

    def pick(cands: List[str]) -> Optional[str]:
        for k in cands:
            if k in lower:
                return lower[k]
        return None

    # heuristisches Matching
    time_col = pick(["time_ms", "time (ms)", "timestamp", "time", "t", "jstime", "daqtime"])
    video_col = pick(["video_id", "videoid", "video"])

    # fuzzy: suche Substrings
    def fuzzy(keys: List[str]) -> Optional[str]:
        for lc, orig in lower.items():
            if any(k in lc for k in keys):
                return orig
        return None

    bvp_col = fuzzy(["bvp", "ppg", "photopleth", "blood volume"])
    gsr_col = fuzzy(["gsr", "eda", "skin conduc", "electroderm"])
    skt_col = fuzzy(["skt", "skin temp", "temperature"])

    return {"time": time_col, "video": video_col, "bvp": bvp_col, "gsr": gsr_col, "skt": skt_col}


def _estimate_fs(t_s: np.ndarray) -> float:
    dt = np.diff(t_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    return 1.0 / float(np.median(dt))


def _sliding_windows(n: int, win: int, step: int) -> List[Tuple[int, int]]:
    out, i = [], 0
    while i + win <= n:
        out.append((i, i + win))
        i += step
    return out


def _safe_mean(x: np.ndarray) -> float:
    v = x[np.isfinite(x)]
    return float(v.mean()) if v.size else np.nan


def _safe_slope(y: np.ndarray, fs: float) -> float:
    v = y[np.isfinite(y)]
    if v.size < 2 or not np.isfinite(fs) or fs <= 0:
        return np.nan
    x = np.arange(v.size) / fs
    xm, ym = x.mean(), v.mean()
    num = np.sum((x - xm) * (v - ym))
    den = np.sum((x - xm) ** 2)
    return float(num / den) if den != 0 else 0.0


def _valid_segment(x: np.ndarray,
                   fs: float,
                   min_seconds: float = 6.0,
                   max_nan_frac: float = 0.3,
                   min_std: float = 1e-6) -> bool:
    if x is None or x.size == 0 or not np.isfinite(fs) or fs <= 0:
        return False
    n = x.size
    finite = np.isfinite(x)
    if not finite.any():
        return False
    frac_nan = 1.0 - (finite.sum() / n)
    if frac_nan > max_nan_frac:
        return False
    if n < int(min_seconds * fs):
        return False
    if np.nanstd(x) < min_std:
        return False
    return True


def _bvp_features_ppg_nk(bvp_win: np.ndarray, fs: float,
                         min_peaks: int = 3) -> Dict[str, float]:
    """
    Stabil: Peaks via NeuroKit2, Zeitdomänen-HRV. Keine Freq-HRV -> keine Spline-/Short-signal-Warnungen.
    """
    out = {
        "bvp_bpm": np.nan, "bvp_ibi": np.nan, "bvp_sdnn": np.nan, "bvp_sdsd": np.nan,
        "bvp_rmssd": np.nan, "bvp_pnn20": np.nan, "bvp_pnn50": np.nan, "bvp_mad": np.nan,
        "bvp_sd1": np.nan, "bvp_sd2": np.nan, "bvp_s": np.nan, "bvp_sd1sd2_ratio": np.nan,
        "bvp_breathingrate": np.nan,  # bleibt im Schema, aber nicht berechnet
    }
    try:
        sig = nk.ppg_clean(bvp_win, sampling_rate=fs)  # robustes Clean
        peaks = nk.ppg_findpeaks(sig, sampling_rate=fs)["PPG_Peaks"]
        if np.sum(peaks) < min_peaks:
            return out

        # Herzrate (BPM) & IBI
        rate = nk.signal_rate(peaks, sampling_rate=fs, desired_length=len(sig))
        out["bvp_bpm"] = float(np.nanmean(rate))

        # RR / IBIs in ms
        peaks_out = nk.ppg_findpeaks(sig, sampling_rate=fs)["PPG_Peaks"]

        # peaks_out kann ein Index-Array sein (int) ODER ein 0/1-Vektor (bool/int)
        peaks_arr = np.asarray(peaks_out)

        # Anzahl Peaks robust bestimmen
        n_peaks = (np.sum(peaks_arr) if set(np.unique(peaks_arr)).issubset({0, 1})
                   else peaks_arr.size)
        if n_peaks < min_peaks:
            return out

        # Für alle weiteren Berechnungen IMMER mit Indizes arbeiten:
        if set(np.unique(peaks_arr)).issubset({0, 1}):
            peak_idx = np.where(peaks_arr == 1)[0]
        else:
            peak_idx = peaks_arr.astype(int)

        # BPM über Indizes (funktioniert mit signal_rate):
        rate = nk.signal_rate(peak_idx, sampling_rate=fs, desired_length=len(sig))
        out["bvp_bpm"] = float(np.nanmean(rate))

        # RR-Intervalle in ms
        rr_ms = np.diff(peak_idx) / fs * 1000.0
        if rr_ms.size < 2:
            return out
        out["bvp_ibi"] = float(np.nanmean(rr_ms))

        # Zeitdomäne
        # SDNN
        out["bvp_sdnn"] = float(np.nanstd(rr_ms, ddof=1)) if rr_ms.size > 1 else np.nan
        # RMSSD
        diff_rr = np.diff(rr_ms)
        out["bvp_rmssd"] = float(np.sqrt(np.nanmean(diff_rr ** 2))) if diff_rr.size else np.nan
        # SDSD = std der Differenzen
        out["bvp_sdsd"] = float(np.nanstd(diff_rr, ddof=1)) if diff_rr.size > 1 else np.nan
        # pNN20 / pNN50
        denom = rr_ms.size - 1 if rr_ms.size > 1 else np.nan
        if np.isfinite(denom) and denom > 0:
            out["bvp_pnn20"] = float(np.sum(np.abs(diff_rr) > 20.0) / denom * 100.0)
            out["bvp_pnn50"] = float(np.sum(np.abs(diff_rr) > 50.0) / denom * 100.0)
        # MAD der HR (als Ersatz für HeartPy hr_mad)
        out["bvp_mad"] = float(np.nanmedian(np.abs(rate - np.nanmedian(rate))))

        # Poincaré: SD1/SD2/S, SD1/SD2-Ratio (klassische Formeln)
        # SD1 = sqrt(0.5) * RMSSD
        if np.isfinite(out["bvp_rmssd"]):
            sd1 = (out["bvp_rmssd"] / np.sqrt(2.0))
        else:
            sd1 = np.nan
        # SD2 ≈ sqrt(2*SDNN^2 - 0.5*RMSSD^2)
        if np.isfinite(out["bvp_sdnn"]) and np.isfinite(out["bvp_rmssd"]):
            sd2_sq = 2.0 * (out["bvp_sdnn"] ** 2) - 0.5 * (out["bvp_rmssd"] ** 2)
            sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else np.nan
        else:
            sd2 = np.nan
        out["bvp_sd1"] = float(sd1) if np.isfinite(sd1) else np.nan
        out["bvp_sd2"] = float(sd2) if np.isfinite(sd2) else np.nan
        out["bvp_s"] = float(np.pi * sd1 * sd2) if np.isfinite(sd1) and np.isfinite(sd2) else np.nan
        out["bvp_sd1sd2_ratio"] = float(sd1 / sd2) if np.isfinite(sd1) and np.isfinite(sd2) and sd2 != 0 else np.nan

        return out
    except Exception:
        return out


# ------------------------- Hauptfunktion -------------------------

def extract_features_case(cfg: ExtractConfig) -> pd.DataFrame:
    phys_dir = cfg.base_path / "data" / "interpolated" / "physiological"
    rows: List[Dict] = []

    for subject in cfg.subjects:
        f = phys_dir / f"sub_{subject}.csv"
        if not f.exists():
            print(f"[WARN] Datei fehlt: {f}")
            continue

        df = pd.read_csv(f)
        cols = _find_columns(df)

        # --- Zeit in Sekunden + fs schätzen ---
        t = pd.to_numeric(df[cols["time"]], errors="coerce").to_numpy()
        t_s = t / 1000.0 if np.nanmax(t) > 1e5 else t.astype(float)
        fs_est = _estimate_fs(t_s)
        fs_eff = fs_est if np.isfinite(fs_est) and 5 <= fs_est <= 2000 else float(cfg.fs_phys)
        print(f"[INFO] sub_{subject}: fs≈{fs_est:.2f} Hz (using {fs_eff:.2f} Hz)")

        # Video-ID optional
        if cols["video"]:
            vraw = pd.to_numeric(df[cols["video"]], errors="coerce").to_numpy()
        else:
            vraw = np.full_like(t_s, np.nan, dtype=float)

        # Kanäle lesen & lokal interpolieren (vermeidet leere Slices)
        def as_array(name: Optional[str]) -> Optional[np.ndarray]:
            if name is None:
                return None
            x = pd.to_numeric(df[name], errors="coerce")
            x = x.interpolate(limit_direction="both")  # füllt lokale NaNs
            return x.to_numpy(dtype=float)

        bvp = as_array(cols["bvp"])
        gsr = as_array(cols["gsr"])
        skt = as_array(cols["skt"])

        # Fensterung
        win = int(round(cfg.window_size_s * fs_eff))
        step = int(round(cfg.step_size_s * fs_eff))
        windows = _sliding_windows(len(t_s), win, step)

        for i0, i1 in windows:
            t0, t1 = t_s[i0], t_s[i1 - 1]

            # Video-ID als Mehrheitswert im Fenster (falls vorhanden)
            vv = vraw[i0:i1]
            if np.isfinite(vv).any():
                vals, counts = np.unique(vv[np.isfinite(vv)], return_counts=True)
                video = float(vals[np.argmax(counts)]) if vals.size else np.nan
            else:
                video = np.nan

            row = {"subject": subject, "video": video, "t_start_s": float(t0), "t_end_s": float(t1)}

            # ----- BVP sicher auswerten -----
            if bvp is not None:
                seg = bvp[i0:i1]
                if _valid_segment(seg, fs_eff, min_seconds=max(6.0, 0.6 * cfg.window_size_s), max_nan_frac=0.3):
                    feats_bvp = _bvp_features_ppg_nk(seg, fs=fs_eff, min_peaks=3)
                    row.update(feats_bvp)
                else:
                    row.update({
                        "bvp_bpm": np.nan, "bvp_ibi": np.nan, "bvp_sdnn": np.nan, "bvp_sdsd": np.nan,
                        "bvp_rmssd": np.nan, "bvp_pnn20": np.nan, "bvp_pnn50": np.nan, "bvp_mad": np.nan,
                        "bvp_sd1": np.nan, "bvp_sd2": np.nan, "bvp_s": np.nan, "bvp_sd1sd2_ratio": np.nan,
                        "bvp_breathingrate": np.nan,
                    })
            else:
                row.update({
                    "bvp_bpm": np.nan, "bvp_ibi": np.nan, "bvp_sdnn": np.nan, "bvp_sdsd": np.nan,
                    "bvp_rmssd": np.nan, "bvp_pnn20": np.nan, "bvp_pnn50": np.nan, "bvp_mad": np.nan,
                    "bvp_sd1": np.nan, "bvp_sd2": np.nan, "bvp_s": np.nan, "bvp_sd1sd2_ratio": np.nan,
                    "bvp_breathingrate": np.nan,
                })

            # ----- GSR/SKT safe-Stats (ohne leere Slices) -----
            if gsr is not None:
                g = gsr[i0:i1]
                row["gsr_mean"] = _safe_mean(g)
                row["gsr_slope_per_s"] = _safe_slope(g, fs_eff)
            else:
                row["gsr_mean"] = np.nan
                row["gsr_slope_per_s"] = np.nan

            if skt is not None:
                s = skt[i0:i1]
                row["skt_mean"] = _safe_mean(s)
                row["skt_slope_per_s"] = _safe_slope(s, fs_eff)
            else:
                row["skt_mean"] = np.nan
                row["skt_slope_per_s"] = np.nan

            rows.append(row)

        print(f"[OK] sub_{subject}: {len(windows)} Fenster verarbeitet.")

    # DataFrame bauen, Spaltenordnungen stabilisieren
    df_feat = pd.DataFrame(rows)
    meta = ["subject", "video", "t_start_s", "t_end_s"]
    bvp_cols = [
        "bvp_bpm", "bvp_ibi", "bvp_sdnn", "bvp_sdsd", "bvp_rmssd", "bvp_pnn20", "bvp_pnn50",
        "bvp_mad", "bvp_sd1", "bvp_sd2", "bvp_s", "bvp_sd1sd2_ratio", "bvp_breathingrate"
    ]
    gsr_cols = ["gsr_mean", "gsr_slope_per_s"]
    skt_cols = ["skt_mean", "skt_slope_per_s"]

    for c in meta + bvp_cols + gsr_cols + skt_cols:
        if c not in df_feat.columns:
            df_feat[c] = np.nan

    df_feat = df_feat[meta + bvp_cols + gsr_cols + skt_cols]

    # Speichern
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(cfg.out_file, index=False)
    print(f"[DONE] Features gespeichert: {cfg.out_file.resolve()}")
    return df_feat


if __name__ == "__main__":
    cfg = ExtractConfig(
        base_path=Path("../case_dataset-master"),
        subjects=list(range(1, 29)),
        window_size_s=10.0,
        step_size_s=1.0,
        fs_phys=20.0,  # nur Fallback
        out_file=Path("outputs/features_case_bvp_gsr_skt.csv"),
    )
    extract_features_case(cfg)
