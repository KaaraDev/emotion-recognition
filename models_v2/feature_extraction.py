# -*- coding: utf-8 -*-
"""
Feature-Extraktion für das CASE-Dataset
======================================

Ziel: Exakt die in der Masterarbeit beschriebenen 17 physiologischen Features
(13 BVP, 2 GSR, 2 SKT) fensterbasiert berechnen und als CSV speichern.

• Fenster: 10 s, Schrittweite: 1 s
• Signale: BVP (PPG), GSR, SKT (Skin Temperature)
• Preprocessing:
    - BVP: Butterworth Bandpass 0.25–3.0 Hz, order=3 (zero-phase)
    - GSR: Butterworth Lowpass 1.5 Hz, order=3 (zero-phase)
    - SKT: Butterworth Lowpass 1.5 Hz, order=3 (zero-phase)
• BVP-Features (HeartPy + Fallbacks):
    bpm, ibi, sdnn, sdsd, rmssd, pnn20, pnn50, mad, sd1, sd2, s, sd1/sd2,
    breathingrate (aus BVP per Spektralschätzung, 0.1–0.6 Hz)
• GSR-Features: mean, slope (lineare Regression)
• SKT-Features: mean, slope (lineare Regression)

Robustheit:
- Automatische Spaltenerkennung (fuzzy)
- Samplingrate aus Zeitstempeln geschätzt (Fallback konfigurierbar)
- Mehrstufige Peak-Detektion (HeartPy → NeuroKit2 → FFT-BPM) gegen NaNs

Ausgabe:
- CSV mit Spalten: subject, window_start_ms, window_end_ms, video_id (falls vorhanden),
  sowie alle 17 Features.

Benötigte Pakete: numpy, pandas, scipy, heartpy, neurokit2

Nutzung (Beispiel):
-------------------
python case_feature_extraction.py \
  --base "../CASE_dataset" \
  --subjects 1 2 3 5 6 7 \
  --out "outputs/features_case_bvp_gsr_skt.csv" \
  --win 10 --step 1

Ordnerstruktur (erwartet):
CASE_dataset/
  data/interpolated/physiological/*.csv  (eine Datei je Subjekt)

Autor: Metin-Projekt (Bachelorarbeit) – ChatGPT Assist
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import re
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

# Optionale Bibliotheken: robust gegen Importfehler
try:
    import heartpy as hp
except Exception:
    hp = None

try:
    import neurokit2 as nk
except Exception:
    nk = None


# ------------------------- Config -------------------------
@dataclass
class ExtractConfig:
    base_path: Path
    subjects: List[int]
    window_size_s: float = 10.0
    step_size_s: float = 1.0
    fs_fallback: float = 1000.0
    out_file: Path = Path("outputs/features_case_bvp_gsr_skt.csv")


# ------------------------- Utilities -------------------------

def _glob_phys_files(base_path: Path) -> List[Path]:
    p = base_path / "data" / "interpolated" / "physiological"
    return sorted(p.rglob("*.csv"))


def _subject_from_name(p: Path) -> Optional[int]:
    # Versucht eine Subjektnummer aus dem Dateinamen zu ziehen (z.B. S01, subject_12, 23)
    m = re.search(r"(?:subject[_-]?)?(\d{1,2})", p.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # zweiter Versuch: Sxx
    m = re.search(r"s(\d{1,2})", p.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _find_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Fuzzy-Mapping der benötigten Spaltennamen.
    Erwartet Spalten für Zeit, BVP, GSR, SKT, optional Video-ID.
    """
    lower = {c.lower(): c for c in df.columns}

    def pick(cands: List[str]) -> Optional[str]:
        for k in cands:
            if k in lower:
                return lower[k]
        return None

    def fuzzy(keys: List[str]) -> Optional[str]:
        for lc, orig in lower.items():
            if any(k in lc for k in keys):
                return orig
        return None

    time_col = pick(["time", "time_ms"]) or fuzzy(["time", "ms"])  # ms bevorzugt
    bvp_col = pick(["bvp", "ppg"]) or fuzzy(["bvp", "ppg", "pulse"])
    gsr_col = pick(["gsr", "eda"]) or fuzzy(["gsr", "eda", "skin"])
    skt_col = pick(["skt"]) or fuzzy(["temp", "skt", "skin_temp"])
    vid_col = pick(["videoid", "video_id"]) or fuzzy(["video", "vid"])  # optional

    return {"time": time_col, "bvp": bvp_col, "gsr": gsr_col, "skt": skt_col, "vid": vid_col}


def _estimate_fs_ms(time_ms: np.ndarray, fs_fallback: float) -> float:
    # Erwartet Millisekunden – robust gegen unregelmäßige Abstände
    if time_ms is None or len(time_ms) < 3:
        return fs_fallback
    diffs = np.diff(time_ms.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return fs_fallback
    median_dt_ms = np.median(diffs)
    if not np.isfinite(median_dt_ms) or median_dt_ms <= 0:
        return fs_fallback
    return 1000.0 / median_dt_ms


def _butter_filter(sig: np.ndarray, fs: float, low: Optional[float] = None, high: Optional[float] = None,
                   order: int = 3):
    if sig is None:
        return None
    if low is not None and high is not None:
        b, a = butter(order, [low / (fs / 2.0), high / (fs / 2.0)], btype="band")
    elif high is not None:
        b, a = butter(order, high / (fs / 2.0), btype="low")
    elif low is not None:
        b, a = butter(order, low / (fs / 2.0), btype="high")
    else:
        return sig
    return filtfilt(b, a, sig)


def _safe_polyfit_slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    if len(y) < 3:
        return np.nan
    try:
        coeffs = np.polyfit(x, y.astype(float), 1)
        return float(coeffs[0])
    except Exception:
        return np.nan


# ------------------------- BVP: Peaks & Features -------------------------

def _bpm_fft(sig: np.ndarray, fs: float, fmin_hz: float = 0.7, fmax_hz: float = 3.5) -> float:
    """Fallback-BPM via Spektralpeak im physiologischen Bereich (~42–210 bpm)."""
    try:
        f, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 2048))
        mask = (f >= fmin_hz) & (f <= fmax_hz)
        if mask.sum() == 0:
            return np.nan
        f_sel = f[mask]
        P_sel = Pxx[mask]
        if len(P_sel) == 0:
            return np.nan
        f_peak = f_sel[np.argmax(P_sel)]
        return float(f_peak * 60.0)
    except Exception:
        return np.nan


def _estimate_breathing_from_bvp_fft(sig: np.ndarray, fs: float, fmin: float = 0.1, fmax: float = 0.6) -> float:
    """Atemfrequenz grob aus BVP-Amplitudenmodulation via PSD (0.1–0.6 Hz ≈ 6–36 bpm Resp)."""
    try:
        # Leicht bandbegrenzt, um Herzanteil zu dämpfen
        sig_d = sig - np.nanmean(sig)
        f, Pxx = welch(sig_d, fs=fs, nperseg=min(len(sig_d), 2048))
        mask = (f >= fmin) & (f <= fmax)
        if mask.sum() == 0:
            return np.nan
        f_sel = f[mask]
        P_sel = Pxx[mask]
        f_peak = f_sel[np.argmax(P_sel)]
        return float(f_peak * 60.0)  # in breaths per minute
    except Exception:
        return np.nan


def _mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _rr_from_peaks(sig: np.ndarray, fs: float):
    """Erzeugt RR-Intervalle (ms) über NK2-Peaks als Fallback."""
    if nk is None:
        return None
    try:
        # nk.ppg_process liefert ein (signals, info) Tuple; wir nutzen die Peak-Markierungen
        signals, info = nk.ppg_process(sig, sampling_rate=fs)
        peaks_bool = signals["PPG_Peaks"].to_numpy().astype(bool)
        peaks_idx = np.where(peaks_bool)[0]
        if peaks_idx.size < 2:
            return None
        rr = np.diff(peaks_idx) / float(fs) * 1000.0  # in Millisekunden
        rr = rr[np.isfinite(rr)]
        return rr if rr.size >= 2 else None
    except Exception:
        return None


def _features_bvp(seg: np.ndarray, fs: float) -> Dict[str, float]:
    feats = {k: np.nan for k in [
        'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
        'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate']}

    wd, m = None, None  # für evtl. RR-Liste von HeartPy

    # 1) HeartPy: bevorzugt
    if hp is not None:
        try:
            wd, m = hp.process(seg, sample_rate=fs)
            mapping = {
                'bvp_bpm': 'bpm', 'bvp_ibi': 'ibi', 'bvp_sdnn': 'sdnn', 'bvp_sdsd': 'sdsd', 'bvp_rmssd': 'rmssd',
                'bvp_pnn20': 'pnn20', 'bvp_pnn50': 'pnn50', 'bvp_mad': 'mad', 'bvp_sd1': 'sd1', 'bvp_sd2': 'sd2',
                'bvp_s': 's', 'bvp_sd1sd2': 'sd1/sd2'
            }
            for out_k, hp_k in mapping.items():
                val = m.get(hp_k, np.nan)
                feats[out_k] = float(val) if np.isfinite(val) else np.nan
        except Exception as e:
            # Kein hartes Abbrechen – wir fallen später auf NK2/FFT zurück
            print(f"[HeartPy] Fehler: {e}")

    # 2) NeuroKit2 (Fallback) für BPM, wenn HeartPy kein valides bpm geliefert hat
    if nk is not None and (not np.isfinite(feats['bvp_bpm']) or feats['bvp_bpm'] <= 0):
        try:
            processed = nk.ppg_process(seg, sampling_rate=fs)
            rate = np.asarray(processed[1]['PPG_Rate'])
            if rate.size > 0 and np.isfinite(rate).any():
                feats['bvp_bpm'] = float(np.nanmean(rate))
        except Exception as e:
            print(f"[NK2 bpm] Fehler: {e}")

    # 3) FFT-Fallback für BPM, falls weiterhin NaN/ungültig
    if not np.isfinite(feats['bvp_bpm']) or feats['bvp_bpm'] <= 0:
        feats['bvp_bpm'] = _bpm_fft(seg, fs)

    # --- NEU: RR-basiertes Fallback für IBI/MAD/SDNN/RMSSD/SDSD/pNN20/pNN50 ---
    need_rr_metrics = any(
        (not np.isfinite(feats[k])) for k in
        ['bvp_mad', 'bvp_sdnn', 'bvp_rmssd', 'bvp_sdsd', 'bvp_pnn20', 'bvp_pnn50', 'bvp_ibi']
    )

    if need_rr_metrics:
        rr = None
        # a) Wenn HeartPy lief, probiere dessen RR-Liste
        if isinstance(wd, dict) and ('RR_list' in wd) and (len(wd['RR_list']) >= 2):
            rr = np.asarray(wd['RR_list'], float)
            rr = rr[np.isfinite(rr)]
            if rr.size < 2:
                rr = None

        # b) Sonst NK2-Fallback
        if rr is None:
            rr = _rr_from_peaks(seg, fs)

        if rr is not None and rr.size >= 2:
            diffs = np.diff(rr)
            diffs = diffs[np.isfinite(diffs)]

            # IBI (ms)
            if not np.isfinite(feats['bvp_ibi']):
                feats['bvp_ibi'] = float(np.nanmean(rr)) if rr.size > 0 else np.nan

            # MAD der RR
            if not np.isfinite(feats['bvp_mad']) or feats['bvp_mad'] <= 0:
                feats['bvp_mad'] = _mad(rr)

            # SDNN (Std der RR)
            if not np.isfinite(feats['bvp_sdnn']):
                feats['bvp_sdnn'] = float(np.nanstd(rr, ddof=1)) if rr.size > 1 else np.nan

            # RMSSD (Wurzel des Mittelwerts der Quadrate der aufeinanderfolgenden RR-Differenzen)
            if not np.isfinite(feats['bvp_rmssd']):
                feats['bvp_rmssd'] = float(np.sqrt(np.nanmean(diffs ** 2))) if diffs.size > 0 else np.nan

            # SDSD (Std der RR-Differenzen)
            if not np.isfinite(feats['bvp_sdsd']):
                feats['bvp_sdsd'] = float(np.nanstd(diffs, ddof=1)) if diffs.size > 1 else np.nan

            # pNN20 / pNN50 (Anteil |ΔRR| > 20/50 ms)
            if not np.isfinite(feats['bvp_pnn20']):
                feats['bvp_pnn20'] = float(np.mean(np.abs(diffs) > 20.0)) if diffs.size > 0 else np.nan
            if not np.isfinite(feats['bvp_pnn50']):
                feats['bvp_pnn50'] = float(np.mean(np.abs(diffs) > 50.0)) if diffs.size > 0 else np.nan
        else:
            print("[RR-Fallback] Keine ausreichenden RR-Intervalle verfügbar.")

    # Atemfrequenz aus BVP per PSD-Schätzung (wie zuvor)
    feats['bvp_breathingrate'] = _estimate_breathing_from_bvp_fft(seg, fs)

    return feats


def _features_gsr(seg: np.ndarray) -> Dict[str, float]:
    return {
        'gsr_mean': float(np.nanmean(seg)) if seg.size else np.nan,
        'gsr_slope': _safe_polyfit_slope(seg)
    }


def _features_skt(seg: np.ndarray) -> Dict[str, float]:
    return {
        'skt_mean': float(np.nanmean(seg)) if seg.size else np.nan,
        'skt_slope': _safe_polyfit_slope(seg)
    }


# ------------------------- Sliding Windows -------------------------

def _iter_windows(n: int, fs: float, win_s: float, step_s: float) -> List[Tuple[int, int]]:
    win = int(round(win_s * fs))
    step = int(round(step_s * fs))
    if win <= 1 or step <= 0:
        return []
    out = []
    for start in range(0, max(0, n - win + 1), step):
        out.append((start, start + win))
    return out


# ------------------------- Hauptlogik -------------------------

def process_subject_csv(path: Path, cfg: ExtractConfig) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = _find_columns(df)
    missing = [k for k, v in cols.items() if k != 'vid' and v is None]
    if missing:
        warnings.warn(f"Fehlende Spalten in {path.name}: {missing}")
        return pd.DataFrame()

    t = df[cols['time']].to_numpy()
    # Falls Zeit in Sekunden vorliegt → in ms umrechnen (heuristik)
    if np.nanmax(t) < 1e5:  # < 100 s → wahrscheinlich Sekunden
        t_ms = t.astype(float) * 1000.0
    else:
        t_ms = t.astype(float)

    fs = _estimate_fs_ms(t_ms, cfg.fs_fallback)

    bvp = df[cols['bvp']].to_numpy(dtype=float)
    gsr = df[cols['gsr']].to_numpy(dtype=float)
    skt = df[cols['skt']].to_numpy(dtype=float)

    # Filter
    try:
        bvp_f = _butter_filter(bvp, fs, low=0.25, high=3.0, order=3)
    except Exception:
        bvp_f = bvp
    try:
        gsr_f = _butter_filter(gsr, fs, high=1.5, order=3)
    except Exception:
        gsr_f = gsr
    try:
        skt_f = _butter_filter(skt, fs, high=1.5, order=3)
    except Exception:
        skt_f = skt

    vid = df[cols['vid']].to_numpy() if cols['vid'] is not None else None

    rows = []
    for i0, i1 in _iter_windows(len(bvp_f), fs, cfg.window_size_s, cfg.step_size_s):
        seg_bvp = bvp_f[i0:i1]
        seg_gsr = gsr_f[i0:i1]
        seg_skt = skt_f[i0:i1]

        feats = {}
        feats.update(_features_bvp(seg_bvp, fs))
        feats.update(_features_gsr(seg_gsr))
        feats.update(_features_skt(seg_skt))

        row = {
            'subject': _subject_from_name(path) or np.nan,
            'window_start_ms': float(t_ms[i0]),
            'window_end_ms': float(t_ms[i1 - 1]),
        }
        if vid is not None:
            # Mehrheit der Video-IDs im Fenster (robust gg. Zwischenwerte)
            vid_win = pd.Series(vid[i0:i1]).mode()
            row['video_id'] = vid_win.iloc[0] if len(vid_win) else np.nan
        row.update(feats)
        rows.append(row)

    return pd.DataFrame(rows)


def extract_features_case(cfg: ExtractConfig) -> pd.DataFrame:
    files = _glob_phys_files(cfg.base_path)
    if not files:
        raise FileNotFoundError(f"Keine CSVs in {cfg.base_path}/data/interpolated/physiological gefunden.")

    # Nur gewünschte Subjekte
    targets = set(cfg.subjects) if cfg.subjects else None
    dfs: List[pd.DataFrame] = []
    for f in files:
        sid = _subject_from_name(f)
        if targets and sid not in targets:
            continue
        print(f"[INFO] Verarbeite {f.name} (subject={sid}) …")
        try:
            df_sub = process_subject_csv(f, cfg)
        except Exception as e:
            warnings.warn(f"Fehler bei {f.name}: {e}")
            df_sub = pd.DataFrame()
        if not df_sub.empty:
            dfs.append(df_sub)

    if not dfs:
        raise RuntimeError("Keine Feature-Daten erzeugt.")

    out = pd.concat(dfs, axis=0, ignore_index=True)

    # Ordnung der Spalten – 17 physiologische Features
    feat_cols = [
        'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
        'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate',
        'gsr_mean', 'gsr_slope', 'skt_mean', 'skt_slope'
    ]

    meta = ['subject', 'window_start_ms', 'window_end_ms'] + (['video_id'] if 'video_id' in out.columns else [])
    cols = meta + feat_cols
    # Fehlende Spalten ergänzen
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out[cols]
    out.to_csv(cfg.out_file, index=False)
    print(f"[OK] Gespeichert: {cfg.out_file}  (n={len(out)})")
    return out


# ------------------------- CLI -------------------------

def _parse_args() -> ExtractConfig:
    ap = argparse.ArgumentParser(description="CASE Feature Extraction (Masterarbeit-kompatibel)")
    ap.add_argument('--base', type=str, required=True, help='Pfad zum CASE-Dataset Root (enthält data/)')
    ap.add_argument('--subjects', type=int, nargs='*', default=[], help='Subjekt-IDs, z.B. 1 2 3 5 6 (leer = alle)')
    ap.add_argument('--win', type=float, default=10.0, help='Fenstergröße in Sekunden (Default 10)')
    ap.add_argument('--step', type=float, default=1.0, help='Schrittweite in Sekunden (Default 1)')
    ap.add_argument('--out', type=str, default='outputs/features_case_bvp_gsr_skt.csv', help='Ausgabe-CSV')
    ap.add_argument('--fs_fallback', type=float, default=1000.0, help='Fallback-Samplingrate (Hz)')
    args = ap.parse_args()

    return ExtractConfig(
        base_path=Path(args.base),
        subjects=list(args.subjects) if args.subjects else [],
        window_size_s=args.win,
        step_size_s=args.step,
        fs_fallback=args.fs_fallback,
        out_file=Path(args.out),
    )


if __name__ == "__main__":
    cfg = _parse_args()
    # Ausgabeordner anlegen
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    extract_features_case(cfg)
