from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
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
    ):
        self.base_path = Path(base_path)
        self.fs = int(fs)
        self.window_size = int(window_size)  # Sekunden
        self.step_size = int(step_size)  # Sekunden
        self.subjects = subjects or list(range(1, 30 + 1))
        self.label_shift_s = float(label_shift_s)
        self.use_video_as_feature = use_video_as_feature

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

    def extract_features(self, window: pd.DataFrame) -> Dict[str, float]:
        """
        Berechnet Basis-Features pro Fenster über die typischen CASE-Kanäle.
        Naming: {signal}_{feature}.
        """
        feats: Dict[str, float] = {}

        # Allgemeine Featurefamilie pro Kanal
        for sig in self.phys_cols:
            if sig not in window.columns:
                continue
            y = window[sig].to_numpy(dtype=float)

            feats[f"{sig}_mean"] = float(np.mean(y)) if len(y) else 0.0
            feats[f"{sig}_std"] = self._safe_std(y)
            feats[f"{sig}_min"] = float(np.min(y)) if len(y) else 0.0
            feats[f"{sig}_max"] = float(np.max(y)) if len(y) else 0.0
            feats[f"{sig}_slope_per_s"] = self._slope_per_second(y)

            # einfache Dynamik-Features
            if len(y) > 1:
                dy = np.diff(y) * self.fs  # approx. erste Ableitung pro Sekunde
                feats[f"{sig}_diff_mean"] = float(np.mean(dy))
                feats[f"{sig}_diff_std"] = self._safe_std(dy)
                feats[f"{sig}_pos_diff_ratio"] = float(np.mean(dy > 0))
            else:
                feats[f"{sig}_diff_mean"] = 0.0
                feats[f"{sig}_diff_std"] = 0.0
                feats[f"{sig}_pos_diff_ratio"] = 0.0

        # GSR-spezifische Extras (Peak-Proxy, AUC)
        if "gsr" in window.columns and len(window) > 1:
            g = window["gsr"].to_numpy(dtype=float)
            dg = np.diff(g)
            peaks = np.where((dg[:-1] > 0) & (dg[1:] <= 0))[0]
            feats["gsr_peak_rate_per_s"] = len(peaks) / max(len(g) / self.fs, 1e-6)
            feats["gsr_auc"] = float(np.trapezoid(g, dx=1 / self.fs))

        # RSP: grobe Atemfrequenz
        if "rsp" in window.columns:
            feats["rsp_rate_per_min"] = self._rsp_rate_per_min(window["rsp"].to_numpy(dtype=float))

        # Optional: Video-ID als One-Hot (nur wenn explizit gewünscht)
        if self.use_video_as_feature and "video" in window.columns:
            vid = int(np.round(window["video"].mode(dropna=False).iloc[0]))
            feats[f"video_id"] = float(vid)

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
# ----------------------------- Beispiel-Nutzung -----------------------------
if __name__ == "__main__":
    """
    Beispiel:
    base_path = "/path/to"  (Ordner, der 'case_dataset-master' enthält)
    """
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 0)   # alle Spalten anzeigen
    pd.set_option("display.max_rows", 20)

    base_path = ".."  # anpassen

    prep = CaseDataPreprocessor(
        base_path=base_path,
        fs=20,
        window_size=5,
        step_size=2,
        subjects=list(range(1, 31)),
        label_shift_s=0.0,
        use_video_as_feature=False
    )

    print("[INFO] Starte Vorbereitung...")
    X, yv, ya, meta = prep.prepare_all()

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
        [meta.head(preview_n), yv.head(preview_n).rename("valence"), ya.head(preview_n).rename("arousal"), X.head(preview_n)],
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

