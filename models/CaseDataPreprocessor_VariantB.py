from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import neurokit2 as nk


@dataclass
class WindowMeta:
    subject: int
    start_s: float
    end_s: float
    video: Optional[int]


class CaseDataPreprocessor:
    """
    Variante B (empfohlen bei 20/10s + 30 Subjects):
    - NeuroKit2 *_process() wird EINMAL pro zusammenhängendem Video-Segment ausgeführt
      (segmentweise, video-ID bleibt konstant).
    - Die daraus resultierenden "derived signals" (Rate/Tonic/Phasic/Amplitude etc.)
      werden sample-aligned als neue Spuren in df geschrieben.
    - Windowing bleibt wie gehabt; Features werden anschließend fensterweise aggregiert.
    """

    def __init__(
            self,
            base_path: str | Path,
            window_size: int = 20,
            step_size: int = 10,
            subjects: Optional[List[int]] = None,
            label_shift_s: float = 0.0,
            use_video_as_feature: bool = False,
            normalize_video_lengths: bool = False,
            target_video_len_s: Optional[float] = None,
            use_raw: bool = True,
            # --- Neu/Optionen ---
            add_neurokit_signals: bool = True,
            neurokit_min_seconds_ecg_ppg_eda_rsp: float = 5.0,
            neurokit_min_seconds_emg: float = 2.0,
            # Feature-Output NaNs später auffüllen? (wie in deinem __main__ Cleanup)
            fillna_value: float = 0.0,
    ):
        self.base_path = Path(base_path)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.subjects = subjects or list(range(1, 30))
        self.label_shift_s = float(label_shift_s)
        self.use_video_as_feature = use_video_as_feature
        self.normalize_video_lengths = bool(normalize_video_lengths)
        self.target_video_len_s = target_video_len_s
        self.use_raw = bool(use_raw)

        self.add_neurokit_signals = bool(add_neurokit_signals)
        self.neurokit_min_seconds_ecg_ppg_eda_rsp = float(neurokit_min_seconds_ecg_ppg_eda_rsp)
        self.neurokit_min_seconds_emg = float(neurokit_min_seconds_emg)
        self.fillna_value = float(fillna_value)

        # Roh-Physio-Kanäle
        self.phys_cols = [
            "ecg", "bvp", "gsr", "rsp", "skt",
            "emg_zygo", "emg_coru", "emg_trap"
        ]

        # NeuroKit2-derived Spuren, die wir segmentweise erzeugen und dann fensterweise aggregieren
        self.nk_signal_cols = [
            "ECG_Rate",
            "PPG_Rate",
            "EDA_Tonic", "EDA_Phasic", "SCR_Peaks", "SCR_Amplitude", "SCR_RiseTime", "SCR_RecoveryTime",
            "RSP_Rate", "RSP_Amplitude",
            "EMG_ZYGO_Amplitude", "EMG_CORU_Amplitude", "EMG_TRAP_Amplitude",
        ]

    # ---------- Helpers ----------

    @staticmethod
    def _slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 2 or np.allclose(np.nanstd(y), 0):
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = np.nanmean(y)
        num = np.nansum((x - x_mean) * (y - y_mean))
        den = np.nansum((x - x_mean) ** 2)
        if den == 0:
            return 0.0
        return float(num / den)

    def _slope_per_second(self, y: np.ndarray, fs_local: float) -> float:
        if fs_local <= 0:
            return 0.0
        return self._slope(y) * fs_local

    @staticmethod
    def _zero_cross_rate(y: np.ndarray, fs_local: float) -> float:
        if len(y) < 2:
            return 0.0
        s = np.signbit(y - np.nanmean(y))
        zc = np.count_nonzero(s[1:] != s[:-1])
        dur_s = max(len(y) / fs_local, 1e-6) if fs_local > 0 else 1.0
        return float(zc / dur_s)

    def _estimate_fs(self, window: pd.DataFrame) -> float:
        t = window["time_s"].to_numpy(dtype=float)
        if len(t) < 2:
            return 0.0
        diffs = np.diff(t)
        dt = np.median(diffs)
        if dt <= 0:
            return 0.0
        return float(1.0 / dt)

    @staticmethod
    def _safe_stats_arr(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    def _add_stats(self, feats: Dict[str, float], sig_name: str, y: np.ndarray, fs_local: float) -> None:
        yv = self._safe_stats_arr(y)
        if yv.size == 0:
            feats[f"{sig_name}_mean"] = 0.0
            feats[f"{sig_name}_std"] = 0.0
            feats[f"{sig_name}_min"] = 0.0
            feats[f"{sig_name}_max"] = 0.0
            feats[f"{sig_name}_median"] = 0.0
            feats[f"{sig_name}_iqr"] = 0.0
            feats[f"{sig_name}_range"] = 0.0
            feats[f"{sig_name}_slope_per_s"] = 0.0
            return

        feats[f"{sig_name}_mean"] = float(np.mean(yv))
        feats[f"{sig_name}_std"] = float(np.std(yv, ddof=1)) if yv.size >= 2 else 0.0
        feats[f"{sig_name}_min"] = float(np.min(yv))
        feats[f"{sig_name}_max"] = float(np.max(yv))
        feats[f"{sig_name}_median"] = float(np.median(yv))
        feats[f"{sig_name}_iqr"] = float(np.percentile(yv, 75) - np.percentile(yv, 25)) if yv.size else 0.0
        feats[f"{sig_name}_range"] = float(np.max(yv) - np.min(yv))
        feats[f"{sig_name}_slope_per_s"] = self._slope_per_second(yv, fs_local)

    # ---------- Loading ----------

    def load_subject(self, subject_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Lädt entweder:
        - non-interpolated physiological (mit 'daqtime' in ms) + interpolierte Annotationen
        - oder komplett interpolierte Daten (wenn use_raw=False)
        Ergebnis:
        - phys: hat 'time_s' + phys channels + 'video'
        - ann : hat 'time_s' + 'valence','arousal' (+ optional 'video')
        """
        if self.use_raw:
            p_phys = (
                    self.base_path
                    / "case_dataset-master"
                    / "data"
                    / "interpolated"
                    / "physiological"
                    / f"sub_{subject_id}.csv"
            )
            p_ann = (
                    self.base_path
                    / "case_dataset-master"
                    / "data"
                    / "interpolated"
                    / "annotations"
                    / f"sub_{subject_id}.csv"
            )

            phys = pd.read_csv(p_phys)

            required_phys_cols = [
                "daqtime", "ecg", "bvp", "gsr", "rsp",
                "skt", "emg_zygo", "emg_coru", "emg_trap", "video"
            ]
            missing = [c for c in required_phys_cols if c not in phys.columns]
            if missing:
                raise ValueError(f"In {p_phys} fehlen erwartete Spalten: {missing}")

            phys["time_s"] = phys["daqtime"].astype(float) / 1000.0

            ann = pd.read_csv(p_ann)
            if "jstime" not in ann.columns:
                raise ValueError(f"'jstime' fehlt in {p_ann} (Annotations-Datei).")
            ann["time_s"] = ann["jstime"].astype(float) / 1000.0
            if "valence" not in ann.columns or "arousal" not in ann.columns:
                raise ValueError(f"'valence' und/oder 'arousal' fehlen in {p_ann}.")
            return phys, ann

        # komplett interpolated
        p_phys = (
                self.base_path
                / "case_dataset-master"
                / "data"
                / "interpolated"
                / "physiological"
                / f"sub_{subject_id}.csv"
        )
        p_ann = (
                self.base_path
                / "case_dataset-master"
                / "data"
                / "interpolated"
                / "annotations"
                / f"sub_{subject_id}.csv"
        )

        phys = pd.read_csv(p_phys)
        ann = pd.read_csv(p_ann)

        if "daqtime" not in phys.columns or "jstime" not in ann.columns:
            raise ValueError("Erwarte Spalten 'daqtime' (phys) bzw. 'jstime' (ann) in Millisekunden.")

        phys["time_s"] = phys["daqtime"] / 1000.0
        ann["time_s"] = ann["jstime"] / 1000.0
        return phys, ann

    def interpolate_annotations(self, phys: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
        """
        Bleibt auf physiologischer Zeitachse (native Sensor-fs),
        interpoliert valence/arousal (und video, falls vorhanden) auf phys time_s.
        """
        out = phys.copy()

        for col in ("valence", "arousal"):
            if col not in ann.columns:
                raise ValueError(f"'{col}' fehlt in annotations CSV.")
            out[col] = np.interp(
                out["time_s"].to_numpy(),
                ann["time_s"].to_numpy(),
                ann[col].to_numpy()
            )

        # diskrete Video-ID via nearest (round)
        if "video" in ann.columns:
            out["video"] = np.interp(
                out["time_s"].to_numpy(),
                ann["time_s"].to_numpy(),
                ann["video"].to_numpy(),
            ).round().astype(int)
        elif "video" in phys.columns:
            out["video"] = phys["video"]
        else:
            out["video"] = -1

        return out

    def _zscore_per_subject(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        zdf = df.copy()
        for c in cols:
            if c not in zdf.columns:
                continue
            mu = np.nanmean(zdf[c].to_numpy(dtype=float))
            sd = np.nanstd(zdf[c].to_numpy(dtype=float))
            if not np.isfinite(sd) or sd == 0:
                zdf[c + "_z"] = 0.0
            else:
                zdf[c + "_z"] = (zdf[c] - mu) / sd
        return zdf

    def _apply_label_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zeitbasierter Shift der Labels (valence/arousal).
        """
        if abs(self.label_shift_s) < 1e-9:
            return df

        shift_s = self.label_shift_s
        out = df.copy()

        out["valence_shifted"] = np.nan
        out["arousal_shifted"] = np.nan

        t = out["time_s"].to_numpy(dtype=float)
        for i, ti in enumerate(t):
            tgt_t = ti + shift_s
            j = np.searchsorted(t, tgt_t)
            if j < len(t):
                out.at[i, "valence_shifted"] = out["valence"].iloc[j]
                out.at[i, "arousal_shifted"] = out["arousal"].iloc[j]

        out["valence"] = out["valence_shifted"]
        out["arousal"] = out["arousal_shifted"]
        out = out.drop(columns=["valence_shifted", "arousal_shifted"])

        valid = out["valence"].notna() & out["arousal"].notna()
        return out.loc[valid].reset_index(drop=True)

    # ---------- Windowing ----------

    def build_windows(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Fenster pro zusammenhängendem Video-Segment (keine Überschreitung von Video-Grenzen).
        """
        if "time_s" not in df.columns:
            raise ValueError("Spalte 'time_s' fehlt im DataFrame.")
        if "video" not in df.columns:
            raise ValueError("Spalte 'video' fehlt im DataFrame – nötig für videoweise Fenster.")

        times = df["time_s"].to_numpy(dtype=float)
        videos = df["video"].to_numpy()
        if len(times) == 0:
            return []

        idx_pairs: List[Tuple[int, int]] = []

        change = np.r_[True, videos[1:] != videos[:-1]]
        block_starts = np.flatnonzero(change)
        block_ends = np.r_[block_starts[1:], len(videos)]

        w = float(self.window_size)
        step = float(self.step_size)

        for b_start, b_end in zip(block_starts, block_ends):
            seg_times = times[b_start:b_end]
            if len(seg_times) < 2:
                continue

            t_start_seg = seg_times[0]
            t_end_seg = seg_times[-1]

            cur_start_t = t_start_seg
            while cur_start_t + w <= t_end_seg + 1e-9:
                t_lo = cur_start_t
                t_hi = cur_start_t + w

                in_window_local = np.where((seg_times >= t_lo) & (seg_times < t_hi))[0]
                if len(in_window_local) > 1:
                    s_idx = b_start + in_window_local[0]
                    e_idx = b_start + in_window_local[-1] + 1
                    idx_pairs.append((s_idx, e_idx))

                cur_start_t += step

        return idx_pairs

    # ---------- Variante B: NeuroKit2 segmentweise ----------

    def add_neurokit_signals_per_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Führt NeuroKit2 *_process() EINMAL pro zusammenhängendem Video-Segment aus.
        Schreibt sample-aligned derived signals in df-Spalten (self.nk_signal_cols).
        """
        out = df.copy()

        # init columns with NaN
        for c in self.nk_signal_cols:
            if c not in out.columns:
                out[c] = np.nan

        videos = out["video"].to_numpy()
        change = np.r_[True, videos[1:] != videos[:-1]]
        block_starts = np.flatnonzero(change)
        block_ends = np.r_[block_starts[1:], len(videos)]

        for s, e in zip(block_starts, block_ends):
            seg = out.iloc[s:e].copy()
            fs = self._estimate_fs(seg)
            if fs <= 0:
                continue

            min5s = int(round(fs * self.neurokit_min_seconds_ecg_ppg_eda_rsp))
            min2s = int(round(fs * self.neurokit_min_seconds_emg))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # ECG
                if "ecg" in seg.columns:
                    y = seg["ecg"].to_numpy(dtype=float)
                    if np.isfinite(y).sum() >= min5s:
                        try:
                            signals, info = nk.ecg_process(y, sampling_rate=fs)
                            if "ECG_Rate" in signals:
                                out.loc[out.index[s:e], "ECG_Rate"] = signals["ECG_Rate"].to_numpy()
                        except Exception:
                            pass

                # PPG/BVP
                if "bvp" in seg.columns:
                    y = seg["bvp"].to_numpy(dtype=float)
                    if np.isfinite(y).sum() >= min5s:
                        try:
                            signals, info = nk.ppg_process(y, sampling_rate=fs)
                            if "PPG_Rate" in signals:
                                out.loc[out.index[s:e], "PPG_Rate"] = signals["PPG_Rate"].to_numpy()
                        except Exception:
                            pass

                # EDA/GSR
                if "gsr" in seg.columns:
                    y = seg["gsr"].to_numpy(dtype=float)
                    if np.isfinite(y).sum() >= min5s:
                        try:
                            signals, info = nk.eda_process(y, sampling_rate=fs)
                            for c in ["EDA_Tonic", "EDA_Phasic", "SCR_Peaks", "SCR_Amplitude", "SCR_RiseTime",
                                      "SCR_RecoveryTime"]:
                                if c in signals:
                                    out.loc[out.index[s:e], c] = signals[c].to_numpy()
                        except Exception:
                            pass

                # RSP
                if "rsp" in seg.columns:
                    y = seg["rsp"].to_numpy(dtype=float)
                    if np.isfinite(y).sum() >= min5s:
                        try:
                            signals, info = nk.rsp_process(y, sampling_rate=fs)
                            for c in ["RSP_Rate", "RSP_Amplitude"]:
                                if c in signals:
                                    out.loc[out.index[s:e], c] = signals[c].to_numpy()
                        except Exception:
                            pass

                # EMG (Amplitude)
                for src_col, out_col in [
                    ("emg_zygo", "EMG_ZYGO_Amplitude"),
                    ("emg_coru", "EMG_CORU_Amplitude"),
                    ("emg_trap", "EMG_TRAP_Amplitude"),
                ]:
                    if src_col in seg.columns:
                        y = seg[src_col].to_numpy(dtype=float)
                        if np.isfinite(y).sum() >= min2s:
                            try:
                                signals, info = nk.emg_process(y, sampling_rate=fs)
                                if "EMG_Amplitude" in signals:
                                    out.loc[out.index[s:e], out_col] = signals["EMG_Amplitude"].to_numpy()
                            except Exception:
                                pass

        return out

    # ---------- Features (window-wise aggregation) ----------

    def extract_features(self, window: pd.DataFrame, fs_local: float) -> Dict[str, float]:
        """
        Fensterweise Feature-Extraktion:
        - Rohkanäle: stats + slope + diff-features
        - NeuroKit-derived Spuren: stats + slope (und SCR-peak-amps robust)
        """
        feats: Dict[str, float] = {}

        # 1) Rohkanäle: Basisstats + slope
        for sig in self.phys_cols:
            if sig in window.columns:
                y = window[sig].to_numpy(dtype=float)
                self._add_stats(feats, sig, y, fs_local)

        # 2) NeuroKit-derived signals: ebenfalls aggregieren
        for sig in self.nk_signal_cols:
            if sig in window.columns:
                y = window[sig].to_numpy(dtype=float)
                # Namen konsistent klein schreiben (optional)
                self._add_stats(feats, sig.lower(), y, fs_local)

        # 3) Robustere SCR-Amplituden nur an Peaks (wie extraction.py)
        #    (Falls Spuren existieren; ansonsten bleibt es 0.)
        if "SCR_Peaks" in window.columns and "SCR_Amplitude" in window.columns:
            peaks = window["SCR_Peaks"].to_numpy(dtype=float)
            amp = window["SCR_Amplitude"].to_numpy(dtype=float)
            mask = (peaks == 1) & np.isfinite(amp) & (amp > 0)
            scr_amp_peaks = amp[mask]
            feats["scr_count"] = float(np.nansum(peaks == 1)) if np.isfinite(peaks).sum() else 0.0
            if scr_amp_peaks.size:
                feats["scr_amp_mean"] = float(np.mean(scr_amp_peaks))
                feats["scr_amp_median"] = float(np.median(scr_amp_peaks))
                feats["scr_amp_std"] = float(np.std(scr_amp_peaks, ddof=1)) if scr_amp_peaks.size >= 2 else 0.0
                feats["scr_amp_max"] = float(np.max(scr_amp_peaks))
                feats["scr_amp_min"] = float(np.min(scr_amp_peaks))
                feats["scr_amp_iqr"] = float(np.percentile(scr_amp_peaks, 75) - np.percentile(scr_amp_peaks, 25))
            else:
                feats["scr_amp_mean"] = 0.0
                feats["scr_amp_median"] = 0.0
                feats["scr_amp_std"] = 0.0
                feats["scr_amp_max"] = 0.0
                feats["scr_amp_min"] = 0.0
                feats["scr_amp_iqr"] = 0.0

        # 4) Generische Dynamikfeatures auf Rohkanälen (wie vorher)
        for sig in self.phys_cols:
            if sig in window.columns:
                y = window[sig].to_numpy(dtype=float)
                if len(y) > 1 and fs_local > 0:
                    dy = np.diff(y) * fs_local
                    dy = dy[np.isfinite(dy)]
                    feats[f"{sig}_diff_mean"] = float(np.mean(dy)) if dy.size else 0.0
                    feats[f"{sig}_diff_std"] = float(np.std(dy, ddof=1)) if dy.size >= 2 else 0.0
                    feats[f"{sig}_pos_diff_ratio"] = float(np.mean(dy > 0)) if dy.size else 0.0
                else:
                    feats[f"{sig}_diff_mean"] = 0.0
                    feats[f"{sig}_diff_std"] = 0.0
                    feats[f"{sig}_pos_diff_ratio"] = 0.0

        # 5) Optional: Video-ID als Feature
        if self.use_video_as_feature and "video" in window.columns:
            vid = int(np.round(window["video"].mode(dropna=False).iloc[0]))
            feats["video_id"] = float(vid)

        return feats

    def _window_labels(self, window: pd.DataFrame) -> Tuple[float, float]:
        v = float(window["valence"].mean())
        a = float(window["arousal"].mean())
        return v, a

    # ---------- Main pipeline ----------

    def prepare_all(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        X_rows: List[Dict[str, float]] = []
        y_val_list: List[float] = []
        y_aro_list: List[float] = []
        metas: List[WindowMeta] = []

        for sid in self.subjects:
            try:
                phys, ann = self.load_subject(sid)
            except FileNotFoundError:
                print(f"[WARN] Dateien für Subject {sid} nicht gefunden – überspringe.")
                continue

            # 1) Annotationen auf native physiologische Zeitachse
            df = self.interpolate_annotations(phys, ann)

            # 2) (Optional) Zeit-Warping (normalerweise AUS!)
            if self.normalize_video_lengths:
                df = self._time_normalize_videos(df)  # falls du die Funktion noch drin hast

            # 3) pro-Subjekt z-Normalisierung (nur Rohkanäle)
            df = self._zscore_per_subject(df, self.phys_cols)

            # 4) Label shift
            df = self._apply_label_shift(df)

            # 5) NEU: NeuroKit2 process segmentweise (Variante B)
            if self.add_neurokit_signals:
                df = self.add_neurokit_signals_per_segment(df)

            # 6) Fenster
            idx_pairs = self.build_windows(df)
            n_windows = len(idx_pairs)
            if not n_windows:
                print(f"[WARN] Keine Fenster für Subject {sid} – überspringe.")
                continue

            print(f"[INFO] Subject {sid}: {n_windows} Fenster – Starte Verarbeitung...")

            for i, (s_idx, e_idx) in enumerate(idx_pairs, start=1):
                w = df.iloc[s_idx:e_idx]

                fs_local = self._estimate_fs(w)
                feats = self.extract_features(w, fs_local)
                X_rows.append(feats)

                v, a = self._window_labels(w)
                y_val_list.append(v)
                y_aro_list.append(a)

                metas.append(
                    WindowMeta(
                        subject=sid,
                        start_s=float(w["time_s"].iloc[0]),
                        end_s=float(w["time_s"].iloc[-1]),
                        video=int(np.round(w["video"].mode(dropna=False).iloc[0])) if "video" in w.columns else -1,
                    )
                )

                if i % max(1, n_windows // 10) == 0 or i == n_windows:
                    progress = 100 * i / n_windows
                    print(f"    Fortschritt Subject {sid}: {progress:5.1f}%")

            print(f"[DONE] Subject {sid} abgeschlossen.\n")

        if not X_rows:
            raise RuntimeError("Keine Fenster/Features erzeugt. Prüfe Pfade, Subjektliste und Parameter.")

        X_df = pd.DataFrame(X_rows).reset_index(drop=True)
        y_val = pd.Series(y_val_list, name="valence")
        y_aro = pd.Series(y_aro_list, name="arousal")
        meta_df = pd.DataFrame([m.__dict__ for m in metas])

        # Cleanup wie in deinem bisherigen Skript (optional hier direkt)
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(self.fillna_value)

        # Konstanten entfernen (optional, aber oft sinnvoll)
        nunique_per_col = X_df.nunique(dropna=False)
        constant_cols = nunique_per_col[nunique_per_col <= 1].index.tolist()
        if constant_cols:
            X_df = X_df.drop(columns=constant_cols)

        print("[ALL DONE] Verarbeitung aller Subjekte abgeschlossen.")
        return X_df, y_val, y_aro, meta_df

    # ---------- Optional: wenn du _time_normalize_videos noch nutzt ----------
    def _time_normalize_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        WARNUNG: Zeit-Warping zerstört Morphologie (ECG Peaks etc.).
        Standard: normalize_video_lengths=False.
        """
        v = df.get("video", None)
        if v is None:
            return df

        v = df["video"].to_numpy()
        t = df["time_s"].to_numpy(dtype=np.float64)

        change = np.r_[True, v[1:] != v[:-1]]
        block_starts = np.flatnonzero(change)
        block_ends = np.r_[block_starts[1:], len(v)]

        blocks = []
        durations = []
        for s, e in zip(block_starts, block_ends):
            vid = v[s]
            if vid == -1 or np.isnan(vid):
                continue
            blocks.append((s, e, int(vid)))
            durations.append(t[e - 1] - t[s])

        if not blocks:
            return df

        target_len_s = self.target_video_len_s
        if target_len_s is None:
            target_len_s = float(np.median(durations)) if len(durations) else durations[0]

        out_rows = []
        t_cursor = float(t[0])

        num_cols = [c for c in df.columns if c not in ("video",)]
        for s, e, vid in blocks:
            seg = df.iloc[s:e].copy()
            t0 = float(seg["time_s"].iloc[0])
            seg_t_rel = seg["time_s"].to_numpy(dtype=np.float64) - t0

            fs_block = self._estimate_fs(seg)
            if fs_block <= 0:
                fs_block = 20.0

            n_target = max(int(round(target_len_s * fs_block)), 2)
            new_t_rel = np.linspace(0.0, target_len_s, n_target, dtype=np.float64)
            new_t_abs = t_cursor + new_t_rel

            resampled = {
                "time_s": new_t_abs,
                "video": np.full(n_target, vid, dtype=np.int16)
            }
            for c in num_cols:
                y = seg[c].to_numpy(dtype=np.float64)
                if seg_t_rel[-1] <= 0:
                    resampled[c] = np.full(n_target, float(y[-1]) if len(y) else 0.0, dtype=np.float32)
                else:
                    resampled[c] = np.interp(new_t_rel, seg_t_rel, y).astype(np.float32)

            out_rows.append(pd.DataFrame(resampled))
            new_dt = np.median(np.diff(new_t_abs)) if len(new_t_abs) > 1 else 1.0 / fs_block
            t_cursor = new_t_abs[-1] + new_dt

        out = pd.concat(out_rows, axis=0, ignore_index=True)
        return out


# ----------------------------- Beispiel-Nutzung -----------------------------
if __name__ == "__main__":
    from pathlib import Path

    base_path = ".."  # anpassen

    prep = CaseDataPreprocessor(
        base_path=base_path,
        window_size=20,
        step_size=10,
        subjects=list(range(1, 31)),
        label_shift_s=0.0,
        use_video_as_feature=False,
        normalize_video_lengths=False,
        use_raw=True,
        add_neurokit_signals=True,
        fillna_value=0.0,
    )

    print("[INFO] Starte Vorbereitung...")
    X, yv, ya, meta = prep.prepare_all()

    # ---------- HIER SPEICHERN ----------
    out_dir = Path("features_case_vB_20w10s")
    out_dir.mkdir(parents=True, exist_ok=True)

    X.to_parquet(out_dir / "X.parquet", index=False)
    yv.rename("label_valence").to_csv(out_dir / "y_valence.csv", index=False)
    ya.rename("label_arousal").to_csv(out_dir / "y_arousal.csv", index=False)
    meta.to_csv(out_dir / "meta.csv", index=False)

    # optional: alles zusammen
    combined = pd.concat([meta, yv.rename("label_valence"), ya.rename("label_arousal"), X], axis=1)
    combined.to_parquet(out_dir / "combined.parquet", index=False)

    print(f"[GESPEICHERT] → {out_dir.resolve()}")

