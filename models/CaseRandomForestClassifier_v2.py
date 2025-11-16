# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib


# ----------------------------- Import Preprocessor -----------------------------
from models.CaseDataPreprocessor import CaseDataPreprocessor


# ====================== Random-Forest-Classifier-Wrapper ======================

@dataclass
class RFConfig:
    n_estimators: int = 400
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    n_jobs: int = -1
    random_state: int = 42


class CaseRandomForestClassifier:
    """
    Klassifiziert Emotionen (z.B. Scary, Amusement, Boredom, …) auf Intervall-Ebene
    und evaluiert zusätzlich auf Video-Ebene, indem alle Intervalle eines Videos
    aggregiert werden (Summieren der Klassenwahrscheinlichkeiten → argmax).

    Labels kommen über ein Mapping von Video-ID → Emotionsname (EMO_MAP).
    """

    def __init__(
        self,
        preprocessor: CaseDataPreprocessor | None,
        rf_config: RFConfig | None = None,
        emo_map: Optional[Dict[int, str]] = None,
        remove_videos: Optional[Set[int]] = None,   # global aus den Daten entfernen (z.B. Start/Blue/End)
        exclude_subjects: Optional[Set[int]] = None,  # echter Holdout auf Subjektebene (z.B. {30})
        class_weight: Optional[str] = "balanced",
    ):
        self.prep = preprocessor
        self.config = rf_config or RFConfig()
        self.emo_map: Dict[int, str] = emo_map or {}
        self.remove_videos: Set[int] = set(remove_videos or set())
        self.exclude_subjects: Set[int] = set(exclude_subjects or set())
        self.class_weight = class_weight

        # Model & Encoder
        self.model: Optional[RandomForestClassifier] = None
        self.le_: Optional[LabelEncoder] = None

        # Data containers (Training/CV)
        self.X_: Optional[pd.DataFrame] = None
        self.y_int_: Optional[np.ndarray] = None  # integer-encoded labels
        self.y_str_: Optional[np.ndarray] = None  # string labels
        self.meta_: Optional[pd.DataFrame] = None

        # Holdout (abgetrennte Subjekte)
        self.X_hold_: Optional[pd.DataFrame] = None
        self.y_hold_int_: Optional[np.ndarray] = None
        self.y_hold_str_: Optional[np.ndarray] = None
        self.meta_hold_: Optional[pd.DataFrame] = None

    # --------------------------- Hilfsfunktionen ---------------------------

    def _labels_from_meta(self, meta: pd.DataFrame) -> np.ndarray:
        assert "video" in meta.columns, "meta muss eine Spalte 'video' besitzen."
        if not self.emo_map:
            raise ValueError(
                "emo_map ist leer. Übergib ein Dict {video_id: 'Emotion'} in den Konstruktor."
            )
        # Mappe Video-ID → Emotionsname
        y_str = meta["video"].map(self.emo_map)
        if y_str.isna().any():
            missing = sorted(set(meta.loc[y_str.isna(), "video"].unique()))
            raise ValueError(
                f"Für folgende Video-IDs fehlt ein Label in emo_map: {missing}"
            )
        return y_str.values.astype(str)

    def _apply_global_video_filter(
        self, X: pd.DataFrame, meta: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Entfernt unerwünschte Videos (z.B. Start/Blue/End) global aus X und meta.
        Wichtig: Labels werden erst NACH diesem Schritt erzeugt/encodiert.
        """
        if not self.remove_videos:
            return X, meta

        mask_keep = ~meta["video"].isin(self.remove_videos)
        removed = (~mask_keep).sum()
        if removed > 0:
            print(
                f"[FILTER] Entferne {removed} Intervalle mit Videos={sorted(self.remove_videos)} "
                f"({len(set(meta.loc[~mask_keep, 'video']))} Video-IDs)."
            )
        X_f = X.loc[mask_keep].reset_index(drop=True)
        meta_f = meta.loc[mask_keep].reset_index(drop=True)
        return X_f, meta_f

    def _split_holdout_by_subject(
        self, X: pd.DataFrame, y_int: np.ndarray, y_str: np.ndarray, meta: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
        """Trennt Subjekte als Holdout ab (echtes Test-Set)."""
        if not self.exclude_subjects:
            return X, y_int, y_str, meta

        mask_hold = meta["subject"].isin(self.exclude_subjects).values
        self.X_hold_ = X.loc[mask_hold].reset_index(drop=True)
        self.y_hold_int_ = y_int[mask_hold]
        self.y_hold_str_ = y_str[mask_hold]
        self.meta_hold_ = meta.loc[mask_hold].reset_index(drop=True)

        X = X.loc[~mask_hold].reset_index(drop=True)
        y_int = y_int[~mask_hold]
        y_str = y_str[~mask_hold]
        meta = meta.loc[~mask_hold].reset_index(drop=True)

        print(
            f"[HOLDOUT] Subjekt(e) entfernt: {sorted(self.exclude_subjects)} | "
            f"Intervalle={len(self.X_hold_)} | Videos={sorted(set(self.meta_hold_['video']))}"
        )
        return X, y_int, y_str, meta

    def print_data_summary(self) -> None:
        """Kleiner Überblick zu Train- und Holdout-Daten."""
        def _summ(meta: Optional[pd.DataFrame], name: str) -> None:
            if meta is None or len(meta) == 0:
                print(f"[SUMMARY] {name}: leer")
                return
            n_int = len(meta)
            n_subj = meta["subject"].nunique()
            vids = sorted(meta["video"].unique())
            print(f"[SUMMARY] {name}: Intervalle={n_int} | Subjekte={n_subj} | Videos={vids}")

        _summ(self.meta_, "TRAIN/CV")
        _summ(self.meta_hold_, "HOLDOUT")

        if self.y_int_ is not None and self.le_ is not None:
            cls, cnt = np.unique(self.y_int_, return_counts=True)
            dist = {self.le_.classes_[i]: int(c) for i, c in zip(cls, cnt)}
            print(f"[SUMMARY] Klassenverteilung (TRAIN/CV): {dist}")
        if self.y_hold_int_ is not None and self.le_ is not None and len(self.y_hold_int_) > 0:
            cls, cnt = np.unique(self.y_hold_int_, return_counts=True)
            dist = {self.le_.classes_[i]: int(c) for i, c in zip(cls, cnt)}
            print(f"[SUMMARY] Klassenverteilung (HOLDOUT): {dist}")

    # --------------------------- Daten vorbereiten ---------------------------

    def prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Nutzt den Preprocessor, entfernt unerwünschte Videos global,
        encodiert Labels NACH dem Filter und trennt anschließend Subjekt-Holdout ab.
        """
        assert self.prep is not None, "Preprocessor fehlt."
        X, yv, ya, meta = self.prep.prepare_all()  # yv/ya ungenutzt

        # 1) Unerwünschte Videos global entfernen
        _ = self._labels_from_meta(meta)  # nur Validierung
        X, meta = self._apply_global_video_filter(X, meta)

        # 2) Labels NACH dem Filter neu bilden + encoden
        y_str = self._labels_from_meta(meta)
        le = LabelEncoder()
        y_int = le.fit_transform(y_str)

        # 3) Subjekt-Holdout abtrennen
        X, y_int, y_str, meta = self._split_holdout_by_subject(X, y_int, y_str, meta)

        self.X_, self.y_int_, self.y_str_, self.meta_ = X, y_int, y_str, meta
        self.le_ = le
        return X, y_int, meta

    # --------------------------- CV (Intervall + Video) ---------------------------

    def cross_validate(self, n_splits: int = 5, verbose: bool = True) -> Dict[str, float]:
        """
        GroupKFold: Subjekt-weise K-Fold (mehrere Subjekte im Test pro Fold).
        Für 'immer genau EIN Subjekt als Test' nutze cross_validate_loso().
        """
        assert self.X_ is not None and self.y_int_ is not None and self.meta_ is not None
        groups = self.meta_["subject"].values
        n_unique = np.unique(groups).size
        if n_splits > n_unique:
            raise ValueError(
                f"n_splits={n_splits} > einzigartige Subjekte im Train={n_unique}. "
                f"Bitte n_splits ≤ {n_unique} wählen."
            )

        gkf = GroupKFold(n_splits=n_splits)

        interval_metrics = []  # (acc, bacc, f1)
        video_metrics = []  # (acc, f1_macro)

        for fold, (tr, te) in enumerate(gkf.split(self.X_.values, self.y_int_, groups=groups), 1):
            clf = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                class_weight=self.class_weight,
            )
            Xtr, Xte = self.X_.values[tr], self.X_.values[te]
            ytr, yte = self.y_int_[tr], self.y_int_[te]
            met_te = self.meta_.iloc[te]

            clf.fit(Xtr, ytr)
            y_pred = clf.predict(Xte)
            y_proba = clf.predict_proba(Xte)

            # Intervall-Ebene
            acc = accuracy_score(yte, y_pred)
            bacc = balanced_accuracy_score(yte, y_pred)
            f1m = f1_score(yte, y_pred, average="macro")
            interval_metrics.append((acc, bacc, f1m))

            # Video-Ebene (pro (subject, video) aggregieren)
            acc_v, f1m_v = self._video_level_scores(yte, y_proba, met_te)
            video_metrics.append((acc_v, f1m_v))

            if verbose:
                subj_test = sorted(met_te["subject"].unique())
                print(
                    f"Fold {fold}: Intervalle  acc={acc:.3f}  bacc={bacc:.3f}  f1_macro={f1m:.3f} | "
                    f"Videos  acc={acc_v:.3f}  f1_macro={f1m_v:.3f} | Test-Subjects={subj_test}"
                )

        # Mittelwerte
        interval_arr = np.array(interval_metrics)
        video_arr = np.array(video_metrics)
        out = {
            "interval_accuracy": float(interval_arr[:, 0].mean()),
            "interval_bal_acc": float(interval_arr[:, 1].mean()),
            "interval_f1_macro": float(interval_arr[:, 2].mean()),
            "video_accuracy": float(video_arr[:, 0].mean()),
            "video_f1_macro": float(video_arr[:, 1].mean()),
        }
        if verbose:
            print("\n[CV] Mittelwerte:", {k: round(v, 4) for k, v in out.items()})
        return out

    def cross_validate_loso(self, verbose: bool = True) -> Dict[str, float]:
        """
        Leave-One-Subject-Out-CV (LOSO):
        In jedem Fold wird genau EIN Subjekt als Testset gelassen.
        Aggregiert Intervall- und Video-Level-Metriken über alle Folds.
        """
        assert self.X_ is not None and self.y_int_ is not None and self.meta_ is not None
        groups = self.meta_["subject"].values
        logo = LeaveOneGroupOut()

        interval_metrics = []  # (acc, bacc, f1_macro)
        video_metrics = []     # (acc, f1_macro)

        for fold, (tr, te) in enumerate(logo.split(self.X_.values, self.y_int_, groups=groups), 1):
            clf = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                class_weight=self.class_weight,
            )
            Xtr, Xte = self.X_.values[tr], self.X_.values[te]
            ytr, yte = self.y_int_[tr], self.y_int_[te]
            met_te = self.meta_.iloc[te]

            clf.fit(Xtr, ytr)
            y_pred = clf.predict(Xte)
            y_proba = clf.predict_proba(Xte)

            # Intervall-Ebene
            acc = accuracy_score(yte, y_pred)
            bacc = balanced_accuracy_score(yte, y_pred)
            f1m = f1_score(yte, y_pred, average="macro")
            interval_metrics.append((acc, bacc, f1m))

            # Video-Ebene (pro (subject, video) aggregieren)
            acc_v, f1m_v = self._video_level_scores(yte, y_proba, met_te)
            video_metrics.append((acc_v, f1m_v))

            if verbose:
                subj_test = int(met_te["subject"].iloc[0]) if len(met_te) else -1
                print(
                    f"Fold {fold:02d} (Test-Subject {subj_test}): "
                    f"Intervalle acc={acc:.3f} bacc={bacc:.3f} f1_macro={f1m:.3f} | "
                    f"Videos acc={acc_v:.3f} f1_macro={f1m_v:.3f}"
                )

        interval_arr = np.array(interval_metrics)
        video_arr = np.array(video_metrics)
        out = {
            "interval_accuracy": float(interval_arr[:, 0].mean()),
            "interval_bal_acc": float(interval_arr[:, 1].mean()),
            "interval_f1_macro": float(interval_arr[:, 2].mean()),
            "video_accuracy": float(video_arr[:, 0].mean()),
            "video_f1_macro": float(video_arr[:, 1].mean()),
        }
        if verbose:
            print("\n[LOSO] Mittelwerte:", {k: round(v, 4) for k, v in out.items()})
        return out

    def _video_level_scores(
        self,
        y_true_int: np.ndarray,
        y_proba: np.ndarray,
        meta_subset: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Aggregiert per (subject, video) die Klassenwahrscheinlichkeiten (Summe),
        sagt die Video-Emotion als argmax voraus und vergleicht mit dem wahren Label
        (wir nehmen das häufigste Intervall-Label in diesem Video als Ground Truth).
        Gibt (accuracy, f1_macro) zurück.
        """
        # Index pro Gruppe
        grp_keys = list(zip(meta_subset["subject"].values, meta_subset["video"].values))
        df = pd.DataFrame({
            "subject": [s for s, _ in grp_keys],
            "video": [v for _, v in grp_keys],
            "y_true": y_true_int,
        })

        # Wahrscheinlichkeiten aufsummieren
        proba_df = pd.DataFrame(y_proba)
        proba_df["subject"] = df["subject"].values
        proba_df["video"] = df["video"].values

        proba_sum = proba_df.groupby(["subject", "video"]).sum(numeric_only=True)
        y_pred_video = proba_sum.values.argmax(axis=1)

        # True-Label pro Video: Modus der Intervalle
        y_true_video = (
            df.groupby(["subject", "video"])["y_true"]
              .agg(lambda x: np.bincount(x).argmax())
              .values
        )

        acc = accuracy_score(y_true_video, y_pred_video)
        f1m = f1_score(y_true_video, y_pred_video, average="macro")
        return acc, f1m

    # ------------------------------ Training ------------------------------

    def fit(self, verbose: bool = True) -> RandomForestClassifier:
        assert self.X_ is not None and self.y_int_ is not None
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            class_weight=self.class_weight,
        )
        self.model.fit(self.X_.values, self.y_int_)
        if verbose:
            n, d = self.X_.shape
            print(f"[FIT] RandomForestClassifier trainiert auf {n} Fenstern mit {d} Features.")
        return self.model

    # --------------------------- Inferenz / Utils ---------------------------

    def predict(self, X_new: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self.model is not None
        Xv = X_new.values if isinstance(X_new, pd.DataFrame) else X_new
        return self.model.predict(Xv)

    def predict_proba(self, X_new: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self.model is not None
        Xv = X_new.values if isinstance(X_new, pd.DataFrame) else X_new
        return self.model.predict_proba(Xv)

    def feature_importance(self, top_k: int | None = 20) -> pd.Series:
        assert self.model is not None and self.X_ is not None
        fi = pd.Series(self.model.feature_importances_, index=self.X_.columns).sort_values(ascending=False)
        return fi.head(top_k) if top_k is not None else fi

    # ------------------------------ Holdout-Test ------------------------------

    def evaluate_holdout(self, verbose_report: bool = True) -> Dict[str, float]:
        assert self.model is not None, "Bitte zuerst fit() aufrufen."
        assert self.X_hold_ is not None and len(self.X_hold_) > 0, "Kein Holdout vorhanden."

        y_pred = self.predict(self.X_hold_)
        y_proba = self.predict_proba(self.X_hold_)

        # Intervall
        acc = accuracy_score(self.y_hold_int_, y_pred)
        bacc = balanced_accuracy_score(self.y_hold_int_, y_pred)
        f1m = f1_score(self.y_hold_int_, y_pred, average="macro")

        # Video
        acc_v, f1m_v = self._video_level_scores(self.y_hold_int_, y_proba, self.meta_hold_)

        out = {
            "interval_accuracy": float(acc),
            "interval_bal_acc": float(bacc),
            "interval_f1_macro": float(f1m),
            "video_accuracy": float(acc_v),
            "video_f1_macro": float(f1m_v),
        }
        print("\n[HOLDOUT] Scores:", {k: round(v, 4) for k, v in out.items()})

        if verbose_report and self.le_ is not None:
            print("\n[HOLDOUT] classification_report (Intervalle):")
            print(classification_report(self.y_hold_int_, y_pred, target_names=list(self.le_.classes_)))
            print("[HOLDOUT] confusion_matrix (Intervalle):")
            print(confusion_matrix(self.y_hold_int_, y_pred))

        return out

    # ------------------------------ Persistenz ------------------------------

    def save(self, path: str | bytes | "os.PathLike[str]") -> None:
        assert self.model is not None and self.le_ is not None
        joblib.dump(
            {
                "model": self.model,
                "feature_names": None if self.X_ is None else list(self.X_.columns),
                "rf_config": self.config,
                "classes_": self.le_.classes_.tolist(),
                "emo_map": self.emo_map,
            },
            path,
        )
        print(f"[SAVE] Klassifikationsmodell gespeichert unter: {path}")

    @staticmethod
    def load(path: str | bytes | "os.PathLike[str]") -> "CaseRandomForestClassifier":
        bundle = joblib.load(path)
        obj = CaseRandomForestClassifier(
            preprocessor=None,
            rf_config=bundle.get("rf_config", RFConfig()),
            emo_map=bundle.get("emo_map", {}),
        )
        obj.model = bundle["model"]
        if "feature_names" in bundle:
            obj.X_ = pd.DataFrame(columns=bundle["feature_names"])  # nur Namen parken
        le = LabelEncoder()
        if "classes_" in bundle:
            le.fit(bundle["classes_"])  # nur zum Transport der Klassenreihenfolge
        obj.le_ = le
        print(f"[LOAD] Klassifikationsmodell geladen von: {path}")
        return obj

    # ---------------------- Laden aus gespeicherten Features ----------------------

    def load_prepared_from_dir(
        self,
        features_dir: str = "features_case_10w1s",
        combined_basename: str = "combined",
        prefer_parquet: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Lädt gespeicherte Features (+Labels+Meta) aus features_dir/combined.(parquet|csv.gz),
        entfernt unerwünschte Videos global, erstellt y aus meta['video'] via self.emo_map,
        trennt Subjekt-Holdout ab. Rückgabe: (X, y_int, meta) – bereit für CV/Training.
        """
        import os
        dir_path = os.path.abspath(features_dir)
        pq = os.path.join(dir_path, f"{combined_basename}.parquet")
        gz = os.path.join(dir_path, f"{combined_basename}.csv.gz")

        # 1) Combined-Datei laden
        if prefer_parquet and os.path.exists(pq):
            df = pd.read_parquet(pq)
            print(f"[LOAD] Loaded Parquet: {pq}  shape={df.shape}")
        elif os.path.exists(gz):
            df = pd.read_csv(gz)
            print(f"[LOAD] Loaded CSV.GZ:  {gz}  shape={df.shape}")
        else:
            raise FileNotFoundError(
                f"Keine Combined-Datei gefunden in {dir_path} "
                f"(erwartet: {combined_basename}.parquet oder {combined_basename}.csv.gz)."
            )

        # 2) Spalten trennen
        required_meta = ["subject", "start_s", "end_s", "video"]
        for c in required_meta:
            if c not in df.columns:
                raise ValueError(f"Spalte '{c}' fehlt in der Combined-Datei.")

        label_cols = ["label_valence", "label_arousal"]
        for c in label_cols:
            if c not in df.columns:
                raise ValueError(f"Spalte '{c}' (Label) fehlt in der Combined-Datei.")

        meta = df[required_meta].copy()
        # Features = alles außer Meta + Labelspalten
        drop_cols = set(required_meta + label_cols)
        X = df.drop(columns=[c for c in df.columns if c in drop_cols]).copy()

        # 3) Videos global entfernen (Blue/Start/End etc.)
        _ = self._labels_from_meta(meta)  # nur Validierung
        X, meta = self._apply_global_video_filter(X, meta)

        # 4) Klassenlabels aus Video→Emotion (emo_map) NACH Filterung
        y_str = self._labels_from_meta(meta)
        le = LabelEncoder()
        y_int = le.fit_transform(y_str)

        # 5) Subjekt-Holdout abtrennen
        X, y_int, y_str, meta = self._split_holdout_by_subject(X, y_int, y_str, meta)

        # 6) Im Wrapper parken
        self.X_, self.y_int_, self.y_str_, self.meta_ = X, y_int, y_str, meta
        self.le_ = le
        print(f"[READY] TRAIN: X={X.shape}, y={y_int.shape}, meta={meta.shape}")
        return X, y_int, meta


# ============================== Beispiel-Usage ==============================
if __name__ == "__main__":
    # Mapping: Video-ID → Emotionsname (bitte ggf. anpassen!)
    EMO_MAP = {
        1: "Amusement", 2: "Amusement",
        3: "Boredom",   4: "Boredom",
        5: "Relaxed",   6: "Relaxed",
        7: "Scary",     8: "Scary",
        10: "Start", 11: "Blue", 12: "End",
        # Falls weitere Video-IDs vorkommen, hier ergänzen.
    }

    rf_cfg = RFConfig(
        n_estimators=400,
        min_samples_leaf=2,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    # Preprocessor ist NICHT nötig – wir laden aus Dateien
    clfw = CaseRandomForestClassifier(
        preprocessor=None,
        rf_config=rf_cfg,
        emo_map=EMO_MAP,
        remove_videos={10, 11, 12},  # global raus (Blue/Start/End)
        exclude_subjects={30},       # echter Holdout (Subjekt 30); wird nicht in CV genutzt
        class_weight="balanced",
    )

    print("[INFO] Lade gespeicherte Features …")
    X, y, meta = clfw.load_prepared_from_dir(
        features_dir="features_case_10w1s",     # ggf. Pfad anpassen
        combined_basename="combined",     # falls du anders benannt hast
        prefer_parquet=True               # auf False setzen, wenn kein pyarrow installiert
    )

    clfw.print_data_summary()

    # ---------- Leave-One-Subject-Out (LOSO) ----------
    print("[INFO] Starte Leave-One-Subject-Out (LOSO) …")
    clfw.cross_validate_loso(verbose=True)

    # Optional zusätzlich: klassische GroupKFold-5-Fold CV
    # print("[INFO] Starte subjekt-weise 5-Fold GroupKFold CV …")
    # clfw.cross_validate(n_splits=5, verbose=True)

    print("[INFO] Trainiere finales Modell auf TRAIN/CV …")
    clfw.fit()

    print("\n[TOP-Feature-Importances]")
    print(clfw.feature_importance(top_k=20).to_string())

    # Eval nur auf Subjekt-Holdout (falls vorhanden)
    if clfw.X_hold_ is not None and len(clfw.X_hold_) > 0:
        clfw.evaluate_holdout(verbose_report=True)

    clfw.save("rf_emotion_classifier_from_saved_features.joblib")
