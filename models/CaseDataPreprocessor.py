from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings
import neurokit2 as nk
from typing import Dict, List, Tuple, Optional


@dataclass
class WindowMeta:
    subject: int
    start_s: float
    end_s: float
    video: Optional[int]


class CaseDataPreprocessor:
    """
    Datenvorbereitung für das CASE-Dataset:
    - Laden & Synchronisieren (interpolated/physiological + interpolated/annotations)
    - Interpolation von Valence/Arousal auf die Physiologie-Zeitachse
    - Sliding Windows (default 5 s Länge, 1 s Schritt)
    - Basis-Features pro Kanal
    - Rückgabe: X, y_valence, y_arousal sowie Meta-Infos pro Fenster

    Ordnerstruktur:
      base_path/
        case_dataset-master/
          data/
            interpolated/
              physiological/sub_{ID}.csv
              annotations/sub_{ID}.csv
    """

    def __init__(
            self,
            base_path: str | Path,
            fs: int = 20,
            window_size: int = 5,
            step_size: int = 1,
            subjects: Optional[List[int]] = None,
            label_shift_s: float = 0.0,  # optionaler Zeitversatz (z.B. 3.0 s) um GSR-Latenz zu kompensieren
            use_video_as_feature: bool = False,
            normalize_video_lengths: bool = False,
            target_video_len_s: Optional[float] = None
    ):
        self.base_path = Path(base_path)
        self.fs = int(fs)
        self.window_size = int(window_size)  # Sekunden
        self.step_size = int(step_size)  # Sekunden
        self.subjects = subjects or list(range(1, 30 + 1))
        self.label_shift_s = float(label_shift_s)
        self.use_video_as_feature = use_video_as_feature
        self.normalize_video_lengths = bool(normalize_video_lengths)
        self.target_video_len_s = target_video_len_s

        # erwartete Kanäle in physiological CSVs (CASE interpolated)
        self.phys_cols = [
            "ecg", "bvp", "gsr", "rsp", "skt",
            "emg_zygo", "emg_coru", "emg_trap"
        ]

    # ---------- Hilfsfunktionen ----------

    @staticmethod
    def _slope(y: np.ndarray) -> float:
        """Lineare Steigung (polyfit 1. Ordnung) in 'pro Sample'."""
        n = len(y)
        if n < 2 or np.allclose(y.std(), 0):
            return 0.0
        x = np.arange(n, dtype=float)
        # geschlossene Form der linearen Regression für Effizienz
        x_mean = x.mean()
        y_mean = y.mean()
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if den == 0:
            return 0.0
        return float(num / den)

    def _slope_per_second(self, y: np.ndarray) -> float:
        """Steigung in 'pro Sekunde' statt 'pro Sample'."""
        return self._slope(y) * self.fs

    @staticmethod
    def _safe_std(y: np.ndarray) -> float:
        return float(np.std(y)) if len(y) else 0.0

    @staticmethod
    def _zero_cross_rate(y: np.ndarray) -> float:
        """Grobe Atem-/EMG-Aktivitäts-Proxy: Rate der Nulldurchgänge pro Sekunde."""
        if len(y) < 2:
            return 0.0
        s = np.signbit(y - np.mean(y))
        zc = np.count_nonzero(s[1:] != s[:-1])
        return float(zc)

    def _rsp_rate_per_min(self, y: np.ndarray) -> float:
        """
        Sehr einfache Schätzung der Atemfrequenz: Zählung lokaler Maxima über gleitende Ableitung.
        Robust genug als Feature, aber nicht als medizinische Messung gedacht.
        """
        if len(y) < 3:
            return 0.0
        dy = np.diff(y)
        # Peak als Vorzeichenwechsel von + nach -
        peaks = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0]
        breaths_per_s = len(peaks) / max(len(y) / self.fs, 1e-6)
        return float(breaths_per_s * 60.0)

    # 1) Hilfsfunktion ergänzen (in die Klasse einfügen)
    def _resample_to_fs(self, df: pd.DataFrame, target_fs: int) -> pd.DataFrame:
        t_old = df["time_s"].to_numpy(dtype=np.float64)
        t0, t1 = float(t_old[0]), float(t_old[-1])
        step = 1.0 / target_fs
        t_new = np.arange(t0, t1 + 1e-9, step, dtype=np.float64)

        out = {"time_s": t_new}
        # numerische Spalten linear, video per nearest
        num_cols = [c for c in df.columns if c not in ("video",)]
        for c in num_cols:
            y = df[c].to_numpy(dtype=np.float64)
            out[c] = np.interp(t_new, t_old, y).astype(np.float32)

        if "video" in df.columns:
            v = df["video"].to_numpy()
            idx = np.searchsorted(t_old, t_new, side="left")
            idx = np.clip(idx, 0, len(t_old) - 1)
            left = np.clip(idx - 1, 0, len(t_old) - 1)
            use_left = (np.abs(t_new - t_old[left]) <= np.abs(t_new - t_old[idx]))
            nearest_idx = np.where(use_left, left, idx)
            out["video"] = np.asarray(v[nearest_idx], dtype=np.int16)

        return pd.DataFrame(out)

    # ---------- Kernmethoden ----------

    def load_subject(self, subject_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Lädt physiological + annotations (interpolated) und gibt beide DataFrames zurück.
        Fügt time_s-Spalten (Sekunden) hinzu.
        """
        p_phys = self.base_path / "case_dataset-master" / "data" / "interpolated" / "physiological" / f"sub_{subject_id}.csv"
        p_ann = self.base_path / "case_dataset-master" / "data" / "interpolated" / "annotations" / f"sub_{subject_id}.csv"

        phys = pd.read_csv(p_phys)
        ann = pd.read_csv(p_ann)

        # Zeitachsen in Sekunden
        if "daqtime" not in phys.columns or "jstime" not in ann.columns:
            raise ValueError("Erwarte Spalten 'daqtime' in phys bzw. 'jstime' in ann (Millisekunden).")

        phys["time_s"] = phys["daqtime"] / 1000.0
        ann["time_s"] = ann["jstime"] / 1000.0

        return phys, ann

    def interpolate_annotations(self, phys: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
        """
        Interpoliert Valence & Arousal (und Video-ID diskret) auf die Physiologie-Zeitachse.
        Gibt eine *kopierte* DataFrame mit Spalten valence, arousal zurück.
        """
        out = phys.copy()
        for col in ("valence", "arousal"):
            if col not in ann.columns:
                raise ValueError(f"'{col}' fehlt in annotations CSV.")
            out[col] = np.interp(out["time_s"].to_numpy(), ann["time_s"].to_numpy(), ann[col].to_numpy())

        # Video-ID: diskrete Werte → nächstliegendes Label per nearest-neighbour-Interpolation
        if "video" in ann.columns:
            # wir nehmen hier 'pad' Logik: Video-ID an den nächsten Zeitstempel andocken
            out["video"] = np.interp(
                out["time_s"].to_numpy(),
                ann["time_s"].to_numpy(),
                ann["video"].to_numpy(),
            ).round().astype(int)
        elif "video" in phys.columns:
            out["video"] = phys["video"]
        else:
            out["video"] = -1  # unbekannt

        return out

    def _zscore_per_subject(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Z-Normalisierung pro Person auf ausgewählten Spalten (robust gegenüber NaNs)."""
        zdf = df.copy()
        for c in cols:
            mu = np.nanmean(zdf[c].to_numpy())
            sd = np.nanstd(zdf[c].to_numpy())
            if not np.isfinite(sd) or sd == 0:
                zdf[c + "_z"] = 0.0
            else:
                zdf[c + "_z"] = (zdf[c] - mu) / sd
        return zdf

    def build_windows(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Gibt eine Liste von (start_idx, end_idx)-Indices für Sliding Windows zurück.
        """
        n = len(df)
        win = self.window_size * self.fs
        step = self.step_size * self.fs
        idx_pairs = []
        i = 0
        while i + win <= n:
            idx_pairs.append((i, i + win))
            i += step
        return idx_pairs

    def _time_normalize_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Linearly time-warp each continuous block of the same video ID to a common length.
        Target length in seconds is self.target_video_len_s if set, else the subject's
        median video duration. Works on already resampled data (uniform 'time_s').
        """
        if "video" not in df.columns:
            return df

        v = df["video"].to_numpy()
        t = df["time_s"].to_numpy(dtype=np.float64)

        # Identify contiguous blocks of equal video id (and ignore -1)
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

        # Decide target length (seconds)
        target_len_s = self.target_video_len_s
        if target_len_s is None:
            target_len_s = float(np.median(durations)) if len(durations) else durations[0]

        # Resample each block to the target sample count at fs
        out_rows = []
        t_cursor = float(t[0])

        num_cols = [c for c in df.columns if c not in ("video",)]
        for s, e, vid in blocks:
            seg = df.iloc[s:e].copy()
            t0, t1 = float(seg["time_s"].iloc[0]), float(seg["time_s"].iloc[-1])
            seg_t_rel = seg["time_s"].to_numpy(dtype=np.float64) - t0

            n_target = max(int(round(target_len_s * self.fs)), 2)
            new_t_rel = np.linspace(0.0, target_len_s, n_target, dtype=np.float64)
            new_t_abs = t_cursor + new_t_rel

            resampled = {"time_s": new_t_abs, "video": np.full(n_target, vid, dtype=np.int16)}
            # linear interp for numeric columns
            for c in num_cols:
                y = seg[c].to_numpy(dtype=np.float64)
                # guard against zero duration (shouldn’t happen post-resample)
                if seg_t_rel[-1] <= 0:
                    resampled[c] = np.full(n_target, float(y[-1]) if len(y) else 0.0, dtype=np.float32)
                else:
                    resampled[c] = np.interp(new_t_rel, seg_t_rel, y).astype(np.float32)

            out_rows.append(pd.DataFrame(resampled))
            t_cursor = new_t_abs[-1] + (1.0 / self.fs)  # small gapless advance

        out = pd.concat(out_rows, axis=0, ignore_index=True)

        # If there were leading/trailing non-labeled samples (-1), we drop them.
        return out

    def extract_features(self, window: pd.DataFrame) -> Dict[str, float]:
        """
        NeuroKit2-basierte Feature-Extraktion pro Fenster.
        - ECG: HRV-Zeit/Frequenz (so weit im kurzen Fenster möglich)
        - BVP (PPG): Pulsrate + Zeitdomänen-PPG-Features
        - EDA (GSR): tonisch/phasic + SCR-Features
        - RSP: Atemrate + Amplituden-/Perioden-Features
        - EMG: gefilterte Hüllkurvenstatistiken (NeuroKit-Prozessierung)
        - SKT: einfache Statistiken (Temp → keine Peak-Modelle)
        Fallback: Basale Stats/Slope, wenn ein Kanal fehlschlägt.
        """
        feats: Dict[str, float] = {}

        def _add_stats(sig_name: str, y: np.ndarray) -> None:
            # Basale Fallback-Stats (werden immer ergänzt, nützlich für sehr kurze/fehlerhafte Fenster)
            if len(y) == 0:
                feats[f"{sig_name}_mean"] = 0.0
                feats[f"{sig_name}_std"] = 0.0
                feats[f"{sig_name}_min"] = 0.0
                feats[f"{sig_name}_max"] = 0.0
                feats[f"{sig_name}_slope_per_s"] = 0.0
                return
            feats[f"{sig_name}_mean"] = float(np.nanmean(y))
            feats[f"{sig_name}_std"] = float(np.nanstd(y))
            feats[f"{sig_name}_min"] = float(np.nanmin(y))
            feats[f"{sig_name}_max"] = float(np.nanmax(y))
            feats[f"{sig_name}_slope_per_s"] = self._slope_per_second(y)

        # ---------- ECG / HRV ----------
        if "ecg" in window.columns:
            y = window["ecg"].to_numpy(dtype=float)
            _add_stats("ecg", y)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # R-Peak-Detektion
                    ecg_clean = nk.ecg_clean(y, sampling_rate=self.fs)
                    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=self.fs)
                    # HRV-Features (Zeit)
                    hrv_time = nk.hrv_time(rpeaks, sampling_rate=self.fs, show=False)
                    for k, v in hrv_time.items():
                        feats[f"ecg_{k}"] = float(v) if np.isfinite(v).all() else np.nan
                    # HRV-Features (Frequenz) – bei sehr kurzen Fenstern ggf. leer
                    try:
                        hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=self.fs, psd_method="welch", show=False)
                        for k, v in hrv_freq.items():
                            feats[f"ecg_{k}"] = float(v) if np.isfinite(v).all() else np.nan
                    except Exception:
                        pass
                    # Momentane Herzrate (Mittel/Std)
                    # (Aus R-Peaks ableitbar, robust gegenüber kurzem Fenster)
                    if "ECG_R_Peaks" in rpeaks:
                        rate = nk.signal_rate(rpeaks["ECG_R_Peaks"], sampling_rate=self.fs, desired_length=len(y))
                        feats["ecg_rate_mean"] = float(np.nanmean(rate))
                        feats["ecg_rate_std"] = float(np.nanstd(rate))
            except Exception:
                # Keine harten Fehler – basale Stats sind schon gesetzt
                pass

        # ---------- BVP (PPG) ----------
        if "bvp" in window.columns:
            y = window["bvp"].to_numpy(dtype=float)
            _add_stats("bvp", y)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ppg_clean = nk.ppg_clean(y, sampling_rate=self.fs)
                    # Peaks & Pulsrate
                    peaks = nk.ppg_peaks(ppg_clean, sampling_rate=self.fs)[1]
                    # Zeitdomänen-Features des Pulses
                    # (NeuroKit hat kein ppg_intervalrelated wie EDA/RSP; wir nehmen Rate + Interbeat-Zeiten)
                    rate = nk.signal_rate(peaks["PPG_Peaks"], sampling_rate=self.fs, desired_length=len(y))
                    feats["ppg_rate_mean"] = float(np.nanmean(rate))
                    feats["ppg_rate_std"] = float(np.nanstd(rate))
                    # Interbeat-Interval (IBI) Zeitstatistiken
                    ibi = nk.events_to_interval(peaks["PPG_Peaks"])  # in Samples
                    if len(ibi) > 0:
                        ibi_s = ibi / self.fs
                        feats["ppg_ibi_mean_s"] = float(np.nanmean(ibi_s))
                        feats["ppg_ibi_std_s"] = float(np.nanstd(ibi_s))
            except Exception:
                pass

        # ---------- EDA / GSR ----------
        # Hinweis: Dein CSV nennt den Kanal "gsr". NeuroKit verwendet "eda".
        if "gsr" in window.columns:
            y = window["gsr"].to_numpy(dtype=float)
            _add_stats("gsr", y)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    signals, info = nk.eda_process(y, sampling_rate=self.fs)
                    # Intervall-bezogene Features (SCR-Peaks etc.)
                    eda_feats = nk.eda_intervalrelated(signals)
                    for k, v in eda_feats.iloc[0].items():
                        feats[f"eda_{k}"] = float(v) if np.isfinite(v) else np.nan
                    # Ein paar nützliche Ableitungen
                    feats["eda_tonic_mean"] = float(np.nanmean(signals.get("EDA_Tonic", pd.Series(dtype=float))))
                    feats["eda_phasic_mean"] = float(np.nanmean(signals.get("EDA_Phasic", pd.Series(dtype=float))))
            except Exception:
                pass

        # ---------- RSP ----------
        if "rsp" in window.columns:
            y = window["rsp"].to_numpy(dtype=float)
            _add_stats("rsp", y)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    signals, info = nk.rsp_process(y, sampling_rate=self.fs)
                    rsp_feats = nk.rsp_intervalrelated(signals)
                    for k, v in rsp_feats.iloc[0].items():
                        feats[f"rsp_{k}"] = float(v) if np.isfinite(v) else np.nan
                    # Kompakte Metriken
                    if "RSP_Rate" in signals:
                        feats["rsp_rate_mean"] = float(np.nanmean(signals["RSP_Rate"]))
                        feats["rsp_rate_std"] = float(np.nanstd(signals["RSP_Rate"]))
            except Exception:
                pass

        # ---------- EMG (Zygo, Corr, Trap): Hüllkurve/Amplitude ----------
        for emg_col in ["emg_zygo", "emg_coru", "emg_trap"]:
            if emg_col in window.columns:
                y = window[emg_col].to_numpy(dtype=float)
                _add_stats(emg_col, y)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Bandpass + Hüllkurve
                        emg_clean = nk.emg_clean(y, sampling_rate=self.fs)
                        emg_ampl = nk.emg_amplitude(emg_clean, sampling_rate=self.fs)
                        emg_act = nk.emg_activation(emg_ampl, sampling_rate=self.fs)
                        feats[f"{emg_col}_ampl_mean"] = float(np.nanmean(emg_ampl))
                        feats[f"{emg_col}_ampl_std"] = float(np.nanstd(emg_ampl))
                        feats[f"{emg_col}_act_mean"] = float(np.nanmean(emg_act))
                        feats[f"{emg_col}_act_std"] = float(np.nanstd(emg_act))
                except Exception:
                    pass

        # ---------- SKT (Hauttemperatur): einfache Stats ----------
        if "skt" in window.columns:
            y = window["skt"].to_numpy(dtype=float)
            _add_stats("skt", y)

        # ---------- Ein paar generische Dynamik-Features (derivatives) ----------
        # (Ergänzend; nützlich, falls NeuroKit wegen zu kurzer Fenster wenig liefern kann.)
        for sig in self.phys_cols:
            if sig in window.columns:
                y = window[sig].to_numpy(dtype=float)
                if len(y) > 1:
                    dy = np.diff(y) * self.fs
                    feats[f"{sig}_diff_mean"] = float(np.nanmean(dy))
                    feats[f"{sig}_diff_std"] = float(np.nanstd(dy))
                    feats[f"{sig}_pos_diff_ratio"] = float(np.mean(dy > 0))
                else:
                    feats[f"{sig}_diff_mean"] = 0.0
                    feats[f"{sig}_diff_std"] = 0.0
                    feats[f"{sig}_pos_diff_ratio"] = 0.0

        # Optional: Video-ID (wie bisher)
        if self.use_video_as_feature and "video" in window.columns:
            vid = int(np.round(window["video"].mode(dropna=False).iloc[0]))
            feats["video_id"] = float(vid)

        return feats

    def _window_labels(self, window: pd.DataFrame) -> Tuple[float, float]:
        """
        Mittelwert von Valence/Arousal im Fenster als Label.
        (Optionaler Label-Shift wurde bereits vor dem Windowing angewendet.)
        """
        v = float(window["valence"].mean())
        a = float(window["arousal"].mean())
        return v, a

    def _apply_label_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verschiebt Valence/Arousal um label_shift_s in die Zukunft (→ Reaktion verzögert)."""
        if abs(self.label_shift_s) < 1e-9:
            return df
        shift_samples = int(round(self.label_shift_s * self.fs))
        out = df.copy()
        out["valence"] = out["valence"].shift(-shift_samples)
        out["arousal"] = out["arousal"].shift(-shift_samples)
        # Ränder entfernen, damit keine NaNs in Labels landen:
        valid = out["valence"].notna() & out["arousal"].notna()
        return out.loc[valid].reset_index(drop=True)

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

            df = self.interpolate_annotations(phys, ann)
            df = self._resample_to_fs(df, self.fs)
            df = self._zscore_per_subject(df, self.phys_cols)

            if self.normalize_video_lengths:
                df = self._time_normalize_videos(df)

            df = self._apply_label_shift(df)
            idx_pairs = self.build_windows(df)

            n_windows = len(idx_pairs)
            if not n_windows:
                print(f"[WARN] Keine Fenster für Subject {sid} – überspringe.")
                continue

            print(f"[INFO] Subject {sid}: {n_windows} Fenster – Starte Verarbeitung...")

            for i, (s_idx, e_idx) in enumerate(idx_pairs, start=1):
                w = df.iloc[s_idx:e_idx]
                feats = self.extract_features(w)
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

                # Fortschritt in 10-%-Schritten anzeigen
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

        print("[ALL DONE] Verarbeitung aller Subjekte abgeschlossen.")
        return X_df, y_val, y_aro, meta_df


# ----------------------------- Beispiel-Nutzung -----------------------------
if __name__ == "__main__":
    """
    Beispiel:
    base_path = "/path/to"  (Ordner, der 'case_dataset-master' enthält)
    """
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 0)  # alle Spalten anzeigen
    pd.set_option("display.max_rows", 20)

    base_path = ".."  # anpassen

    prep = CaseDataPreprocessor(
        base_path=base_path,
        fs=20,
        window_size=10,
        step_size=10,
        subjects=list(range(1, 31)),
        label_shift_s=0.0,
        use_video_as_feature=False,
        normalize_video_lengths=False,  # ← enable normalization
        target_video_len_s=None  # ← optional; if None uses per-subject median
    )

    print("[INFO] Starte Vorbereitung...")
    X, yv, ya, meta = prep.prepare_all()

    # ---------------- Speichern: alle Features + Labels + Meta ----------------
    out_dir = Path("features_case")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Problematische Werte bereinigen
    X = X.replace([np.inf, -np.inf], np.nan)

    # 2) Spalten, die komplett NaN sind, verwerfen (manchmal liefern sehr kurze Fenster keine NK-Features)
    all_na_cols = [c for c in X.columns if X[c].isna().all()]
    if all_na_cols:
        print(f"[CLEANUP] Entferne {len(all_na_cols)} komplett leere Feature-Spalten.")
        X = X.drop(columns=all_na_cols)

    # 3) Optional: verbleibende NaNs füllen (je nach Downstream-ML)
    #    Für reines Abspeichern nicht zwingend, aber oft praktisch:
    X_filled = X.fillna(0.0)

    # 4) Labels sinnvoll benennen und zusammenbauen
    y_val_named = yv.rename("label_valence")
    y_aro_named = ya.rename("label_arousal")

    combined = pd.concat([meta, y_val_named, y_aro_named, X_filled], axis=1)

    # 5) Schreiben: Parquet (schnell/kompakt) und CSV (kompatibel)
    #    a) Gesamtpaket
    combined.to_parquet(out_dir / "combined.parquet", index=False)
    combined.to_csv(out_dir / "combined.csv.gz", index=False, compression="gzip")

    #    b) Einzelteile
    X_filled.to_parquet(out_dir / "X_features.parquet", index=False)
    X_filled.to_csv(out_dir / "X_features.csv.gz", index=False, compression="gzip")

    y_val_named.to_csv(out_dir / "y_valence.csv", index=False)
    y_aro_named.to_csv(out_dir / "y_arousal.csv", index=False)
    meta.to_csv(out_dir / "meta.csv", index=False)

    # 6) Featureliste sichern
    with open(out_dir / "feature_names.txt", "w", encoding="utf-8") as f:
        for col in X_filled.columns:
            f.write(f"{col}\n")

    print("\n[GESPEICHERT]")
    print(f"  → {(out_dir / 'combined.parquet').resolve()}")
    print(f"  → {(out_dir / 'combined.csv.gz').resolve()}")
    print(f"  → {(out_dir / 'X_features.parquet').resolve()}")
    print(f"  → {(out_dir / 'X_features.csv.gz').resolve()}")
    print(f"  → {(out_dir / 'y_valence.csv').resolve()}")
    print(f"  → {(out_dir / 'y_arousal.csv').resolve()}")
    print(f"  → {(out_dir / 'meta.csv').resolve()}")
    print(f"  → {(out_dir / 'feature_names.txt').resolve()}")

    # (Optional) Kurzer Überblick
    print("\n[FEATURES] Anzahl Spalten (nach Cleanup):", X_filled.shape[1])
    print("[BEISPIEL-FEATURES]:", list(X_filled.columns[:10]))

    # ---------------- Anzeigen / Zusammenfassen ----------------
    print("\n[SHAPES]")
    print(f"X:    {X.shape}  (Fenster × Features)")
    print(f"y_v:  {yv.shape}  (Valence)")
    print(f"y_a:  {ya.shape}  (Arousal)")
    print(f"meta: {meta.shape} (subject, start_s, end_s, video)")

    print("\n[FEATURE SPALTEN — erste 15]")
    print(list(X.columns[:15]))

    # Vorschau: Features + Labels + Meta nebeneinander
    preview_n = 15
    preview = pd.concat(
        [meta.head(preview_n), yv.head(preview_n).rename("valence"), ya.head(preview_n).rename("arousal"),
         X.head(preview_n)],
        axis=1
    )
    print(f"\n[VORSCHAU: erste {preview_n} Fenster]")
    print(preview.to_string(index=False))

    # Einfache Statistiken
    print("\n[STATISTIK Valence/Arousal]")
    stats = pd.DataFrame({"valence": yv.describe(), "arousal": ya.describe()})
    print(stats.to_string())

    # Verteilung nach Videos (falls vorhanden)
    if "video" in meta.columns:
        print("\n[FENSTER pro Video — Top 10]")
        vc = meta["video"].value_counts().sort_values(ascending=False).head(10)
        print(vc.to_string())

    # Verteilung nach Subjekt
    print("\n[FENSTER pro Subjekt — Top 10]")
    sub_vc = meta["subject"].value_counts().sort_index()
    print(sub_vc.head(10).to_string())

    # ---------------- Speichern einer kleinen Vorschau ----------------
    out_path = Path("preview_windows_sample.csv")
    preview.to_csv(out_path, index=False)
    print(f"\n[GESPEICHERT] Vorschau als CSV: {out_path.resolve()}")
