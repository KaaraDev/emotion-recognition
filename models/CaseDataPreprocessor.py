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
    Angepasst:
    - Kein hartes globales Resampling mehr auf fs (z.B. 20 Hz)
    - Stattdessen arbeiten wir auf der nativen Samplingrate aus den CSVs.
    - Für jedes Fenster schätzen wir die effektive fs_local aus time_s
      und geben die an NeuroKit weiter.
    """

    def __init__(
            self,
            base_path: str | Path,
            window_size: int = 5,  # Sekunden Fensterlänge
            step_size: int = 1,  # Sekunden Schritt
            subjects: Optional[List[int]] = None,
            label_shift_s: float = 0.0,
            use_video_as_feature: bool = False,
            normalize_video_lengths: bool = False,
            target_video_len_s: Optional[float] = None
    ):
        self.base_path = Path(base_path)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.subjects = subjects or list(range(1, 30))
        self.label_shift_s = float(label_shift_s)
        self.use_video_as_feature = use_video_as_feature
        self.normalize_video_lengths = bool(normalize_video_lengths)
        self.target_video_len_s = target_video_len_s

        # physiologische Kanäle, wie gehabt
        self.phys_cols = [
            "ecg", "bvp", "gsr", "rsp", "skt",
            "emg_zygo", "emg_coru", "emg_trap"
        ]

    # ---------- Hilfsfunktionen ----------

    @staticmethod
    def _slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 2 or np.allclose(y.std(), 0):
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if den == 0:
            return 0.0
        return float(num / den)

    def _slope_per_second(self, y: np.ndarray, fs_local: float) -> float:
        # vorher war's * self.fs, jetzt * fs_local
        if fs_local <= 0:
            return 0.0
        return self._slope(y) * fs_local

    @staticmethod
    def _safe_std(y: np.ndarray) -> float:
        return float(np.std(y)) if len(y) else 0.0

    @staticmethod
    def _zero_cross_rate(y: np.ndarray, fs_local: float) -> float:
        if len(y) < 2:
            return 0.0
        s = np.signbit(y - np.mean(y))
        zc = np.count_nonzero(s[1:] != s[:-1])
        dur_s = max(len(y) / fs_local, 1e-6) if fs_local > 0 else 1.0
        return float(zc / dur_s)

    def _rsp_rate_per_min(self, y: np.ndarray, fs_local: float) -> float:
        if len(y) < 3 or fs_local <= 0:
            return 0.0
        dy = np.diff(y)
        peaks = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0]
        duration_s = max(len(y) / fs_local, 1e-6)
        breaths_per_s = len(peaks) / duration_s
        return float(breaths_per_s * 60.0)

    def _estimate_fs(self, window: pd.DataFrame) -> float:
        """
        Schätzt die effektive Samplingrate aus time_s.
        Annahme: ungefähr gleichmäßig gesampelt.
        """
        t = window["time_s"].to_numpy(dtype=float)
        if len(t) < 2:
            return 0.0
        diffs = np.diff(t)
        # median statt mean, damit Ausreißer nicht alles zerstören
        dt = np.median(diffs)
        if dt <= 0:
            return 0.0
        return float(1.0 / dt)

    # ---------- Kernmethoden ----------

    def load_subject(self, subject_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p_phys = self.base_path / "case_dataset-master" / "data" / "interpolated" / "physiological" / f"sub_{subject_id}.csv"
        p_ann = self.base_path / "case_dataset-master" / "data" / "interpolated" / "annotations" / f"sub_{subject_id}.csv"

        phys = pd.read_csv(p_phys)
        ann = pd.read_csv(p_ann)

        if "daqtime" not in phys.columns or "jstime" not in ann.columns:
            raise ValueError("Erwarte Spalten 'daqtime' (phys) bzw. 'jstime' (ann) in Millisekunden.")

        phys["time_s"] = phys["daqtime"] / 1000.0
        ann["time_s"] = ann["jstime"] / 1000.0

        return phys, ann

    def interpolate_annotations(self, phys: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
        """
        Wir bleiben auf der physiologischen Zeitachse (die native fs des Sensors),
        und interpolieren valence/arousal/video dahin.
        Kein globales Resampling mehr!
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
            mu = np.nanmean(zdf[c].to_numpy())
            sd = np.nanstd(zdf[c].to_numpy())
            if not np.isfinite(sd) or sd == 0:
                zdf[c + "_z"] = 0.0
            else:
                zdf[c + "_z"] = (zdf[c] - mu) / sd
        return zdf

    def build_windows(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        NEU:
        Wir bauen Fenster über Zeit, nicht über Samples.
        D.h. wir holen uns Startzeiten, dann nehmen wir alle Zeilen,
        deren time_s in [start, start+window_size).
        Wir geben trotzdem Indexbereiche zurück, damit downstream gleich bleibt.
        """
        times = df["time_s"].to_numpy(dtype=float)
        if len(times) == 0:
            return []

        idx_pairs: List[Tuple[int, int]] = []

        t_start_global = times[0]
        t_end_global = times[-1]

        cur_start_t = t_start_global
        w = float(self.window_size)
        step = float(self.step_size)

        while cur_start_t + w <= t_end_global + 1e-9:
            # Start/End-Bereich in Zeit
            t_lo = cur_start_t
            t_hi = cur_start_t + w

            # alle Indizes in diesem Zeitbereich
            in_window = np.where((times >= t_lo) & (times < t_hi))[0]
            if len(in_window) > 1:
                s_idx = in_window[0]
                e_idx = in_window[-1] + 1  # slice-exclusive
                idx_pairs.append((s_idx, e_idx))

            cur_start_t += step

        return idx_pairs

    def _time_normalize_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        WARNUNG:
        Diese Funktion 'warpt' Signale zeitlich. Das zerstört Morphologie (ECG Peaks etc.).
        Für echte Physiologie sollte das normalerweise AUS bleiben.
        Wir lassen sie drin, aber du hast normalize_video_lengths=False gesetzt, also kein Aufruf.
        """
        # Wir lassen die Implementierung aus deinem Code unverändert,
        # aber in prepare_all rufen wir das standardmäßig NICHT mehr auf,
        # es sei denn du setzt normalize_video_lengths=True.
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

            # wir approximieren eine lokale fs über diesen Block
            fs_block = self._estimate_fs(seg)
            if fs_block <= 0:
                fs_block = 20.0  # Fallback

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
            # advance cursor (lückenlos)
            new_dt = np.median(np.diff(new_t_abs)) if len(new_t_abs) > 1 else 1.0 / fs_block
            t_cursor = new_t_abs[-1] + new_dt

        out = pd.concat(out_rows, axis=0, ignore_index=True)
        return out

    def extract_features(self, window: pd.DataFrame, fs_local: float) -> Dict[str, float]:
        """
        Angepasst:
        - Wir benutzen fs_local (geschätzte Fenster-Samplingrate),
          nicht self.fs.
        - Sonst wie gehabt, ABER wir setzen bei fehlenden Peaks 0 statt NaN,
          damit es downstream keine Löcher reißt.
        """
        feats: Dict[str, float] = {}

        def _add_stats(sig_name: str, y: np.ndarray) -> None:
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
            feats[f"{sig_name}_slope_per_s"] = self._slope_per_second(y, fs_local)

        # ---------- ECG / HRV ----------
        if "ecg" in window.columns:
            y = window["ecg"].to_numpy(dtype=float)
            _add_stats("ecg", y)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ecg_clean = nk.ecg_clean(y, sampling_rate=fs_local)
                    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs_local)

                    # Herzrate
                    if "ECG_R_Peaks" in rpeaks:
                        rate = nk.signal_rate(
                            rpeaks["ECG_R_Peaks"],
                            sampling_rate=fs_local,
                            desired_length=len(y)
                        )
                        feats["ecg_rate_mean"] = float(np.nanmean(rate))
                        feats["ecg_rate_std"] = float(np.nanstd(rate))
                    else:
                        feats["ecg_rate_mean"] = 0.0
                        feats["ecg_rate_std"] = 0.0

                    # HRV Zeit
                    try:
                        hrv_time = nk.hrv_time(rpeaks, sampling_rate=fs_local, show=False)
                        for k, v in hrv_time.items():
                            feats[f"ecg_{k}"] = float(v) if np.isfinite(v).all() else 0.0
                    except Exception:
                        pass

                    # HRV Frequenz
                    try:
                        hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs_local, psd_method="welch", show=False)
                        for k, v in hrv_freq.items():
                            feats[f"ecg_{k}"] = float(v) if np.isfinite(v).all() else 0.0
                    except Exception:
                        pass
            except Exception:
                # Stats sind schon gesetzt
                pass

        # ---------- BVP (PPG) ----------
        if "bvp" in window.columns:
            y = window["bvp"].to_numpy(dtype=float)
            _add_stats("bvp", y)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ppg_clean = nk.ppg_clean(y, sampling_rate=fs_local)
                    peaks = nk.ppg_peaks(ppg_clean, sampling_rate=fs_local)[1]

                    rate = nk.signal_rate(
                        peaks["PPG_Peaks"],
                        sampling_rate=fs_local,
                        desired_length=len(y)
                    )
                    feats["ppg_rate_mean"] = float(np.nanmean(rate))
                    feats["ppg_rate_std"] = float(np.nanstd(rate))

                    ibi = nk.events_to_interval(peaks["PPG_Peaks"])
                    if len(ibi) > 0:
                        ibi_s = ibi / fs_local
                        feats["ppg_ibi_mean_s"] = float(np.nanmean(ibi_s))
                        feats["ppg_ibi_std_s"] = float(np.nanstd(ibi_s))
                    else:
                        feats["ppg_ibi_mean_s"] = 0.0
                        feats["ppg_ibi_std_s"] = 0.0
            except Exception:
                pass

        # ---------- EDA / GSR ----------
        if "gsr" in window.columns:
            y = window["gsr"].to_numpy(dtype=float)
            _add_stats("gsr", y)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    signals, info = nk.eda_process(y, sampling_rate=fs_local)
                    eda_feats = nk.eda_intervalrelated(signals)
                    for k, v in eda_feats.iloc[0].items():
                        feats[f"eda_{k}"] = float(v) if np.isfinite(v) else 0.0
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
                    signals, info = nk.rsp_process(y, sampling_rate=fs_local)
                    rsp_feats = nk.rsp_intervalrelated(signals)
                    for k, v in rsp_feats.iloc[0].items():
                        feats[f"rsp_{k}"] = float(v) if np.isfinite(v) else 0.0

                    if "RSP_Rate" in signals:
                        feats["rsp_rate_mean"] = float(np.nanmean(signals["RSP_Rate"]))
                        feats["rsp_rate_std"] = float(np.nanstd(signals["RSP_Rate"]))
                    else:
                        feats["rsp_rate_mean"] = 0.0
                        feats["rsp_rate_std"] = 0.0
            except Exception:
                pass

        # ---------- EMG ----------
        for emg_col in ["emg_zygo", "emg_coru", "emg_trap"]:
            if emg_col in window.columns:
                y = window[emg_col].to_numpy(dtype=float)
                _add_stats(emg_col, y)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        emg_clean = nk.emg_clean(y, sampling_rate=fs_local)
                        emg_ampl = nk.emg_amplitude(emg_clean, sampling_rate=fs_local)
                        emg_act = nk.emg_activation(emg_ampl, sampling_rate=fs_local)
                        feats[f"{emg_col}_ampl_mean"] = float(np.nanmean(emg_ampl))
                        feats[f"{emg_col}_ampl_std"] = float(np.nanstd(emg_ampl))
                        feats[f"{emg_col}_act_mean"] = float(np.nanmean(emg_act))
                        feats[f"{emg_col}_act_std"] = float(np.nanstd(emg_act))
                except Exception:
                    pass

        # ---------- SKT ----------
        if "skt" in window.columns:
            y = window["skt"].to_numpy(dtype=float)
            _add_stats("skt", y)

        # ---------- generische Dynamik-Features ----------
        for sig in self.phys_cols:
            if sig in window.columns:
                y = window[sig].to_numpy(dtype=float)
                if len(y) > 1 and fs_local > 0:
                    dy = np.diff(y) * fs_local
                    feats[f"{sig}_diff_mean"] = float(np.nanmean(dy))
                    feats[f"{sig}_diff_std"] = float(np.nanstd(dy))
                    feats[f"{sig}_pos_diff_ratio"] = float(np.mean(dy > 0))
                else:
                    feats[f"{sig}_diff_mean"] = 0.0
                    feats[f"{sig}_diff_std"] = 0.0
                    feats[f"{sig}_pos_diff_ratio"] = 0.0

        # Video-ID als Feature (optional)
        if self.use_video_as_feature and "video" in window.columns:
            vid = int(np.round(window["video"].mode(dropna=False).iloc[0]))
            feats["video_id"] = float(vid)

        return feats

    def _window_labels(self, window: pd.DataFrame) -> Tuple[float, float]:
        v = float(window["valence"].mean())
        a = float(window["arousal"].mean())
        return v, a

    def _apply_label_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vorsicht: Wir haben jetzt kein fixes fs mehr, also ist "shift in Samples"
        nicht mehr sauber definiert. Wir lösen das zeitbasiert:
        """
        if abs(self.label_shift_s) < 1e-9:
            return df

        shift_s = self.label_shift_s
        out = df.copy()

        # neue Spalten initial kopieren
        out["valence_shifted"] = np.nan
        out["arousal_shifted"] = np.nan

        t = out["time_s"].to_numpy(dtype=float)
        for i, ti in enumerate(t):
            tgt_t = ti + shift_s
            # Index des nächsten Zeitpunkts >= tgt_t
            j = np.searchsorted(t, tgt_t)
            if j < len(t):
                out.at[i, "valence_shifted"] = out["valence"].iloc[j]
                out.at[i, "arousal_shifted"] = out["arousal"].iloc[j]

        out["valence"] = out["valence_shifted"]
        out["arousal"] = out["arousal_shifted"]
        out = out.drop(columns=["valence_shifted", "arousal_shifted"])

        # gültige Zeilen behalten
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

            # 1. Annotationen auf native physiologische Zeitachse legen
            df = self.interpolate_annotations(phys, ann)

            # 2. (Optional) Video-Zeitausgleich -> normalerweise AUS lassen!
            if self.normalize_video_lengths:
                df = self._time_normalize_videos(df)

            # 3. pro-Subjekt z-Normalisierung der physio-Kanäle
            df = self._zscore_per_subject(df, self.phys_cols)

            # 4. Label shift zeitbasiert
            df = self._apply_label_shift(df)

            # 5. Fenster über Zeit bauen
            idx_pairs = self.build_windows(df)
            n_windows = len(idx_pairs)
            if not n_windows:
                print(f"[WARN] Keine Fenster für Subject {sid} – überspringe.")
                continue

            print(f"[INFO] Subject {sid}: {n_windows} Fenster – Starte Verarbeitung...")

            for i, (s_idx, e_idx) in enumerate(idx_pairs, start=1):
                w = df.iloc[s_idx:e_idx]

                # lokale fs schätzen aus time_s
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

        print("[ALL DONE] Verarbeitung aller Subjekte abgeschlossen.")
        return X_df, y_val, y_aro, meta_df


# ----------------------------- Beispiel-Nutzung -----------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 0)
    pd.set_option("display.max_rows", 20)

    base_path = ".."  # anpassen

    prep = CaseDataPreprocessor(
        base_path=base_path,
        window_size=60,
        step_size=60,
        subjects=list(range(1, 31)),
        label_shift_s=0.0,
        use_video_as_feature=False,
        normalize_video_lengths=False,
        target_video_len_s=None
    )

    print("[INFO] Starte Vorbereitung...")
    X, yv, ya, meta = prep.prepare_all()

    # ---------------- Cleanup & Feature-Selektion ----------------
    out_dir = Path("features_case")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Inf-Werte in NaN umwandeln
    X = X.replace([np.inf, -np.inf], np.nan)

    # 2) Spalten, die komplett NaN sind, verwerfen
    all_na_cols = [c for c in X.columns if X[c].isna().all()]
    if all_na_cols:
        print(f"[CLEANUP] Entferne {len(all_na_cols)} komplett NaN-Spalten.")
        X = X.drop(columns=all_na_cols)

    # 3) Verbleibende NaNs auffüllen (z. B. wenn mal ein Fenster kein Peak hatte)
    X = X.fillna(0.0)

    # 4) Spalten, die KEINE Varianz haben, entfernen
    #    -> also Features, die in allen Fenstern exakt denselben Wert haben (z.B. nur 0.0)
    nunique_per_col = X.nunique(dropna=False)
    constant_cols = nunique_per_col[nunique_per_col <= 1].index.tolist()

    if constant_cols:
        print(f"[CLEANUP] Entferne {len(constant_cols)} konstant wertlose Spalten.")
        print("          Beispiele:", constant_cols[:10])
        X = X.drop(columns=constant_cols)

    # 5) Jetzt finalisierte Feature-Matrix
    X_final = X.copy()

    # 6) Labels sinnvoll benennen und zusammenbauen
    y_val_named = yv.rename("label_valence")
    y_aro_named = ya.rename("label_arousal")

    combined = pd.concat([meta, y_val_named, y_aro_named, X_final], axis=1)

    # ---------------- Speichern ----------------
    # Gesamtpaket
    combined.to_parquet(out_dir / "combined.parquet", index=False)
    combined.to_csv(out_dir / "combined.csv.gz", index=False, compression="gzip")

    # Nur Features (gefiltert!)
    X_final.to_parquet(out_dir / "X_features.parquet", index=False)
    X_final.to_csv(out_dir / "X_features.csv.gz", index=False, compression="gzip")

    # Labels einzeln
    y_val_named.to_csv(out_dir / "y_valence.csv", index=False)
    y_aro_named.to_csv(out_dir / "y_arousal.csv", index=False)

    # Meta
    meta.to_csv(out_dir / "meta.csv", index=False)

    # Featureliste sichern (nur die guten Features!)
    with open(out_dir / "feature_names.txt", "w", encoding="utf-8") as f:
        for col in X_final.columns:
            f.write(f"{col}\n")

    # ---------------- Logs / Überblick ----------------
    print("\n[GESPEICHERT]")
    print(f"  → {(out_dir / 'combined.parquet').resolve()}")
    print(f"  → {(out_dir / 'combined.csv.gz').resolve()}")
    print(f"  → {(out_dir / 'X_features.parquet').resolve()}")
    print(f"  → {(out_dir / 'X_features.csv.gz').resolve()}")
    print(f"  → {(out_dir / 'y_valence.csv').resolve()}")
    print(f"  → {(out_dir / 'y_arousal.csv').resolve()}")
    print(f"  → {(out_dir / 'meta.csv').resolve()}")
    print(f"  → {(out_dir / 'feature_names.txt').resolve()}")

    print("\n[FEATURES] Anzahl Spalten nach Cleanup:", X_final.shape[1])
    print("[BEISPIEL-FEATURES]:", list(X_final.columns[:10]))

    print("\n[SHAPES]")
    print(f"X_final: {X_final.shape}  (Fenster × Features)")
    print(f"y_v:     {yv.shape}  (Valence)")
    print(f"y_a:     {ya.shape}  (Arousal)")
    print(f"meta:    {meta.shape} (subject, start_s, end_s, video)")

    preview_n = 15
    preview = pd.concat(
        [
            meta.head(preview_n),
            yv.head(preview_n).rename("valence"),
            ya.head(preview_n).rename("arousal"),
            X_final.head(preview_n),
        ],
        axis=1
    )
    print(f"\n[VORSCHAU: erste {preview_n} Fenster]")
    print(preview.to_string(index=False))

    print("\n[STATISTIK Valence/Arousal]")
    stats = pd.DataFrame({"valence": yv.describe(), "arousal": ya.describe()})
    print(stats.to_string())

    if "video" in meta.columns:
        print("\n[FENSTER pro Video — Top 10]")
        vc = meta["video"].value_counts().sort_values(ascending=False).head(10)
        print(vc.to_string())

    print("\n[FENSTER pro Subjekt — Top 10]")
    sub_vc = meta["subject"].value_counts().sort_index()
    print(sub_vc.head(10).to_string())

    out_path = Path("preview_windows_sample.csv")
    preview.to_csv(out_path, index=False)
    print(f"\n[GESPEICHERT] Vorschau als CSV: {out_path.resolve()}")