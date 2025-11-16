# -*- coding: utf-8 -*-
"""
Feature-Extraktion für das CASE-Dataset (Masterarbeit-kompatibel)
=================================================================

Liefert:
- 17 physiologische Fenster-Features (BVP/GSR/SKT)
- optional: kontinuierliche Annotationen (arousal, valence) pro Fenster gemittelt
- Meta: subject, window_start_ms, window_end_ms, video_id

Fenster: 10 s, Schrittweite: 1 s (konfigurierbar)

Physio Preprocessing:
- BVP: Butterworth Bandpass 0.25–3.0 Hz, order=3 (zero-phase)
- GSR: Butterworth Lowpass 1.5 Hz, order=3 (zero-phase)
- SKT: Butterworth Lowpass 1.5 Hz, order=3 (zero-phase)

Robustheit:
- Fuzzy-Spaltenerkennung (physio & annotation)
- Samplingrate aus Zeitstempeln (ms) geschätzt (Fallback konfigurierbar)
- Mehrstufige Peak-Detektion (HeartPy → NeuroKit2 → FFT-BPM)
- Annotationen werden aus /data/interpolated/annotation/*.csv je Subject geladen
  und per Zeit (ms) fensterweise gemittelt.

Nutzung (Beispiel):
python case_feature_extraction.py \
  --base "../CASE_dataset" \
  --subjects 1 2 3 5 6 7 \
  --out "outputs_10w1s/features_case_bvp_gsr_skt.csv" \
  --win 10 --step 1
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

# tqdm (optional)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Optional: HeartPy & NeuroKit2
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
    out_file: Path = Path("../models_v3/outputs/features_case_bvp_gsr_skt.csv")
    progress: bool = True  # Fortschrittsbalken


# ------------------------- Utilities -------------------------

def _glob_phys_files(base_path: Path) -> List[Path]:
    return sorted((base_path / "data" / "interpolated" / "physiological").rglob("*.csv"))


def _glob_annot_files(base_path: Path) -> List[Path]:
    return sorted((base_path / "data" / "interpolated" / "annotation").rglob("*.csv"))


def _subject_from_name(p: Path) -> Optional[int]:
    m = re.search(r"(?:subject[_-]?|s)?(\d{1,2})", p.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _find_columns_phys(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in df.columns}

    def pick(cands):
        for k in cands:
            if k in lower: return lower[k]
        return None

    def fuzzy(keys):
        for lc, orig in lower.items():
            if any(k in lc for k in keys): return orig
        return None

    time_col = pick(["time_ms", "time"]) or fuzzy(["time", "ms"])  # ms bevorzugt
    bvp_col = pick(["bvp", "ppg"]) or fuzzy(["bvp", "ppg", "pulse"])
    gsr_col = pick(["gsr", "eda"]) or fuzzy(["gsr", "eda", "skin"])
    skt_col = pick(["skt"]) or fuzzy(["temp", "skt", "skin_temp"])
    vid_col = pick(["video_id", "videoid"]) or fuzzy(["video", "vid"])
    return {"time": time_col, "bvp": bvp_col, "gsr": gsr_col, "skt": skt_col, "vid": vid_col}


def _find_columns_annot(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in df.columns}

    def pick(cands):
        for k in cands:
            if k in lower: return lower[k]
        return None

    def fuzzy(keys):
        for lc, orig in lower.items():
            if any(k in lc for k in keys): return orig
        return None

    time_col = pick(["time_ms", "jstime"]) or fuzzy(["jstime", "ms"])
    # CASE liefert meist getrennte 'valence' und 'arousal'
    val_col = pick(["valence"]) or fuzzy(["val"])
    aro_col = pick(["arousal"]) or fuzzy(["aro"])
    vid_col = pick(["video_id", "videoid"]) or fuzzy(["video", "vid"])
    return {"time": time_col, "valence": val_col, "arousal": aro_col, "vid": vid_col}


def _estimate_fs_ms(time_ms: np.ndarray, fs_fallback: float) -> float:
    if time_ms is None or len(time_ms) < 3: return fs_fallback
    diffs = np.diff(time_ms.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0: return fs_fallback
    median_dt_ms = np.median(diffs)
    if not np.isfinite(median_dt_ms) or median_dt_ms <= 0: return fs_fallback
    return 1000.0 / median_dt_ms


def _butter_filter(sig: np.ndarray, fs: float, low: Optional[float] = None, high: Optional[float] = None,
                   order: int = 3):
    if sig is None: return None
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
    if len(y) < 3: return np.nan
    try:
        return float(np.polyfit(x, y.astype(float), 1)[0])
    except Exception:
        return np.nan


# ------------------------- BVP & Features -------------------------

def _bpm_fft(sig: np.ndarray, fs: float, fmin_hz: float = 0.7, fmax_hz: float = 3.5) -> float:
    try:
        f, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 2048))
        mask = (f >= fmin_hz) & (f <= fmax_hz)
        if mask.sum() == 0: return np.nan
        f_peak = f[mask][np.argmax(Pxx[mask])]
        return float(f_peak * 60.0)
    except Exception:
        return np.nan


def _estimate_breathing_from_bvp_fft(sig: np.ndarray, fs: float, fmin: float = 0.1, fmax: float = 0.6) -> float:
    try:
        sig_d = sig - np.nanmean(sig)
        f, Pxx = welch(sig_d, fs=fs, nperseg=min(len(sig_d), 2048))
        mask = (f >= fmin) & (f <= fmax)
        if mask.sum() == 0: return np.nan
        f_peak = f[mask][np.argmax(Pxx[mask])]
        return float(f_peak * 60.0)
    except Exception:
        return np.nan


def _mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _rr_from_peaks(sig: np.ndarray, fs: float):
    if nk is None: return None
    try:
        signals, info = nk.ppg_process(sig, sampling_rate=fs)
        peaks_idx = np.where(signals["PPG_Peaks"].to_numpy().astype(bool))[0]
        if peaks_idx.size < 2: return None
        rr = np.diff(peaks_idx) / float(fs) * 1000.0
        rr = rr[np.isfinite(rr)]
        return rr if rr.size >= 2 else None
    except Exception:
        return None


def _features_bvp(seg: np.ndarray, fs: float) -> Dict[str, float]:
    feats = {k: np.nan for k in [
        'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
        'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate'
    ]}
    wd, m = None, None

    # 1) HeartPy
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
            print(f"[HeartPy] Fehler: {e}")

    # 2) NK2 bpm
    if nk is not None and (not np.isfinite(feats['bvp_bpm']) or feats['bvp_bpm'] <= 0):
        try:
            signals, info = nk.ppg_process(seg, sampling_rate=fs)
            rate = np.asarray(signals.get('PPG_Rate', []))
            if rate.size > 0 and np.isfinite(rate).any():
                feats['bvp_bpm'] = float(np.nanmean(rate))
        except Exception as e:
            print(f"[NK2 bpm] Fehler: {e}")

    # 3) FFT Fallback
    if not np.isfinite(feats['bvp_bpm']) or feats['bvp_bpm'] <= 0:
        feats['bvp_bpm'] = _bpm_fft(seg, fs)

    # RR-basierte Metriken nachziehen wenn nötig
    need_rr = any(not np.isfinite(feats[k]) for k in
                  ['bvp_mad', 'bvp_sdnn', 'bvp_rmssd', 'bvp_sdsd', 'bvp_pnn20', 'bvp_pnn50', 'bvp_ibi'])
    if need_rr:
        rr = None
        if isinstance(wd, dict) and ('RR_list' in wd) and (len(wd['RR_list']) >= 2):
            rr = np.asarray(wd['RR_list'], float)
            rr = rr[np.isfinite(rr)]
            if rr.size < 2: rr = None
        if rr is None:
            rr = _rr_from_peaks(seg, fs)

        if rr is not None and rr.size >= 2:
            diffs = np.diff(rr)
            diffs = diffs[np.isfinite(diffs)]
            if not np.isfinite(feats['bvp_ibi']):
                feats['bvp_ibi'] = float(np.nanmean(rr)) if rr.size > 0 else np.nan
            if not np.isfinite(feats['bvp_mad']) or feats['bvp_mad'] <= 0:
                feats['bvp_mad'] = _mad(rr)
            if not np.isfinite(feats['bvp_sdnn']):
                feats['bvp_sdnn'] = float(np.nanstd(rr, ddof=1)) if rr.size > 1 else np.nan
            if not np.isfinite(feats['bvp_rmssd']):
                feats['bvp_rmssd'] = float(np.sqrt(np.nanmean(diffs ** 2))) if diffs.size > 0 else np.nan
            if not np.isfinite(feats['bvp_sdsd']):
                feats['bvp_sdsd'] = float(np.nanstd(diffs, ddof=1)) if diffs.size > 1 else np.nan
            if not np.isfinite(feats['bvp_pnn20']):
                feats['bvp_pnn20'] = float(np.mean(np.abs(diffs) > 20.0)) if diffs.size > 0 else np.nan
            if not np.isfinite(feats['bvp_pnn50']):
                feats['bvp_pnn50'] = float(np.mean(np.abs(diffs) > 50.0)) if diffs.size > 0 else np.nan
        else:
            print("[RR-Fallback] Keine ausreichenden RR-Intervalle verfügbar.")

    feats['bvp_breathingrate'] = _estimate_breathing_from_bvp_fft(seg, fs)
    return feats


def _features_gsr(seg: np.ndarray) -> Dict[str, float]:
    return {'gsr_mean': float(np.nanmean(seg)) if seg.size else np.nan,
            'gsr_slope': _safe_polyfit_slope(seg)}


def _features_skt(seg: np.ndarray) -> Dict[str, float]:
    return {'skt_mean': float(np.nanmean(seg)) if seg.size else np.nan,
            'skt_slope': _safe_polyfit_slope(seg)}


# ------------------------- Sliding Windows -------------------------

def _iter_windows(n: int, fs: float, win_s: float, step_s: float) -> List[Tuple[int, int]]:
    win = int(round(win_s * fs))
    step = int(round(step_s * fs))
    if win <= 1 or step <= 0: return []
    return [(start, start + win) for start in range(0, max(0, n - win + 1), step)]


# ------------------------- Laden Annotation je Subject -------------------------

def _load_annotation_for_subject(base_path: Path, subject_id: int) -> Optional[pd.DataFrame]:
    ann_dir = base_path / "data" / "interpolated" / "annotations"
    if not ann_dir.exists(): return None
    # Suche passende Datei(n) für Subject
    cands = [p for p in ann_dir.rglob("*.csv") if _subject_from_name(p) == subject_id]
    if not cands: return None
    # Nimm die erste passende (oder mergen, falls mehrere – hier reicht i. d. R. eine)
    try:
        df_a = pd.read_csv(cands[0])
        cols = _find_columns_annot(df_a)
        if cols['time'] is None or (cols['valence'] is None and cols['arousal'] is None):
            return None
        # Zeit in ms
        t = df_a[cols['time']].to_numpy()
        t_ms = t.astype(float) * 1000.0 if np.nanmax(t) < 1e5 else t.astype(float)
        out = pd.DataFrame({'time_ms': t_ms})
        if cols['valence'] is not None:
            out['valence'] = pd.to_numeric(df_a[cols['valence']], errors='coerce')
        if cols['arousal'] is not None:
            out['arousal'] = pd.to_numeric(df_a[cols['arousal']], errors='coerce')
        if cols['vid'] is not None:
            out['video_id'] = df_a[cols['vid']]
        return out
    except Exception:
        return None


# ------------------------- Hauptlogik -------------------------

def process_subject_csv(path: Path, cfg: ExtractConfig, df_ann: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = _find_columns_phys(df)
    missing = [k for k, v in cols.items() if k != 'vid' and v is None]
    if missing:
        warnings.warn(f"Fehlende Spalten in {path.name}: {missing}")
        return pd.DataFrame()

    # Zeit in ms
    t_raw = df[cols['time']].to_numpy()
    t_ms = t_raw.astype(float) * 1000.0 if np.nanmax(t_raw) < 1e5 else t_raw.astype(float)
    fs = _estimate_fs_ms(t_ms, cfg.fs_fallback)

    # Signale
    bvp = pd.to_numeric(df[cols['bvp']], errors='coerce').to_numpy(dtype=float)
    gsr = pd.to_numeric(df[cols['gsr']], errors='coerce').to_numpy(dtype=float)
    skt = pd.to_numeric(df[cols['skt']], errors='coerce').to_numpy(dtype=float)

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

    # Fenster
    windows = _iter_windows(len(bvp_f), fs, cfg.window_size_s, cfg.step_size_s)
    rows = []

    iterable = windows
    pbar = None
    if cfg.progress and tqdm is not None:
        subj_id = _subject_from_name(path) or -1
        pbar = tqdm(iterable, total=len(windows), desc=f"S{subj_id:02d} windows", leave=False, unit="win")
    elif cfg.progress and tqdm is None:
        print("[HINWEIS] tqdm nicht installiert -> kein Fortschrittsbalken (pip install tqdm)")

    for (i0, i1) in (pbar if pbar is not None else iterable):
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

        # video_id: (a) aus Physio-CSV, (b) sonst aus Annotations-Mehrheit im Zeitfenster
        if vid is not None:
            vid_win = pd.Series(vid[i0:i1]).mode()
            row['video_id'] = vid_win.iloc[0] if len(vid_win) else np.nan
        elif df_ann is not None and 'video_id' in df_ann.columns:
            t0, t1 = row['window_start_ms'], row['window_end_ms']
            mask_ann = (df_ann['time_ms'] >= t0) & (df_ann['time_ms'] <= t1)
            if mask_ann.any():
                row['video_id'] = pd.Series(df_ann.loc[mask_ann, 'video_id']).mode().iloc[0]
            else:
                row['video_id'] = np.nan

        # Annotationen (Arousal/Valence) per Zeitfenster mitteln
        if df_ann is not None:
            t0, t1 = row['window_start_ms'], row['window_end_ms']
            mask = (df_ann['time_ms'] >= t0) & (df_ann['time_ms'] <= t1)
            if mask.any():
                if 'valence' in df_ann.columns:
                    row['valence'] = float(np.nanmean(df_ann.loc[mask, 'valence'].to_numpy(dtype=float)))
                if 'arousal' in df_ann.columns:
                    row['arousal'] = float(np.nanmean(df_ann.loc[mask, 'arousal'].to_numpy(dtype=float)))
            else:
                if 'valence' in df_ann.columns: row['valence'] = np.nan
                if 'arousal' in df_ann.columns: row['arousal'] = np.nan

        row.update(feats)
        rows.append(row)

    if pbar is not None:
        pbar.close()

    return pd.DataFrame(rows)


def extract_features_case(cfg: ExtractConfig) -> pd.DataFrame:
    phys_files = _glob_phys_files(cfg.base_path)
    if not phys_files:
        raise FileNotFoundError(f"Keine Physio-CSVs in {cfg.base_path}/data/interpolated/physiological gefunden.")

    # Subjekt-Filter
    targets = set(cfg.subjects) if cfg.subjects else None
    selected = []
    for f in phys_files:
        sid = _subject_from_name(f)
        if (targets is None) or (sid in targets):
            selected.append(f)
    if not selected:
        raise RuntimeError("Keine passenden Dateien für die gewünschten Subjects gefunden.")

    # Äußerer Fortschritt
    it = selected
    outer = tqdm(it, desc="Subjects", unit="subj") if (cfg.progress and tqdm is not None) else None

    dfs: List[pd.DataFrame] = []
    for f in (outer if outer is not None else it):
        sid = _subject_from_name(f)
        if outer is None:
            print(f"[INFO] Verarbeite {f.name} (subject={sid}) …")

        # Annotation für dieses Subject laden (einmal pro Subject)
        df_ann = _load_annotation_for_subject(cfg.base_path, sid) if sid is not None else None

        try:
            df_sub = process_subject_csv(f, cfg, df_ann)
        except Exception as e:
            warnings.warn(f"Fehler bei {f.name}: {e}")
            df_sub = pd.DataFrame()

        if not df_sub.empty:
            dfs.append(df_sub)

    if outer is not None:
        outer.close()

    if not dfs:
        raise RuntimeError("Keine Feature-Daten erzeugt.")

    out = pd.concat(dfs, axis=0, ignore_index=True)

    # Zielspalten in gewünschter Reihenfolge
    feat_cols = [
        'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
        'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate',
        'gsr_mean', 'gsr_slope', 'skt_mean', 'skt_slope'
    ]
    meta = ['subject', 'window_start_ms', 'window_end_ms'] + (['video_id'] if 'video_id' in out.columns else [])

    # --- FIX: Valence & Arousal immer vorsehen (werden ggf. mit NaN gefüllt) ---
    anno_cols = ['arousal', 'valence']

    cols = meta + anno_cols + feat_cols

    # fehlende Spalten anlegen
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out[cols]
    out.to_csv(cfg.out_file, index=False)
    print(f"[OK] Gespeichert: {cfg.out_file}  (n={len(out)})")
    return out


# ------------------------- CLI -------------------------

def _parse_args() -> ExtractConfig:
    ap = argparse.ArgumentParser(description="CASE Feature Extraction (Masterarbeit-kompatibel, inkl. AV)")
    ap.add_argument('--base', type=str, required=True, help='Pfad zum CASE-Dataset Root (enthält data/)')
    ap.add_argument('--subjects', type=int, nargs='*', default=[], help='IDs, z.B. 1 2 3 5 6 (leer = alle)')
    ap.add_argument('--win', type=float, default=10.0, help='Fenstergröße in Sekunden (Default 10)')
    ap.add_argument('--step', type=float, default=1.0, help='Schrittweite in Sekunden (Default 1)')
    ap.add_argument('--out', type=str, default='outputs_10w1s/features_case_bvp_gsr_skt.csv', help='Ausgabe-CSV')
    ap.add_argument('--fs_fallback', type=float, default=1000.0, help='Fallback-Samplingrate (Hz)')
    ap.add_argument('--no-progress', action='store_true', help='Deaktiviert tqdm-Fortschrittsbalken')
    args = ap.parse_args()

    return ExtractConfig(
        base_path=Path(args.base),
        subjects=list(args.subjects) if args.subjects else [],
        window_size_s=args.win,
        step_size_s=args.step,
        fs_fallback=args.fs_fallback,
        out_file=Path(args.out),
        progress=(not args.no_progress),
    )


if __name__ == "__main__":
    cfg = _parse_args()
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    extract_features_case(cfg)
