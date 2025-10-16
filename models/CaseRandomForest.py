# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
import joblib

# ----------------------------- Import Preprocessor -----------------------------
from models.CaseDataPreprocessor import CaseDataPreprocessor


# ----------------------------- Hilfsmetrik -----------------------------
def _pearson_multi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mittlerer Pearson r über (Valence, Arousal). Erwartet y in Form (N,2)."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    r_list = []
    for j in range(y_true.shape[1]):
        t = y_true[:, j]
        p = y_pred[:, j]
        if np.std(t) == 0 or np.std(p) == 0:
            r_list.append(0.0)
        else:
            r = np.corrcoef(t, p)[0, 1]
            if not np.isfinite(r):
                r = 0.0
            r_list.append(float(r))
    return float(np.mean(r_list))


# ====================== Random-Forest-Wrapper-Klasse ======================

@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 3
    n_jobs: int = -1
    random_state: int = 42


class CaseRandomForest:
    """
    Kapselt Training/Evaluierung eines Multi-Output-Random-Forest
    auf den von CaseDataPreprocessor erzeugten Daten.
    Variante A: bestimmte Videos werden komplett vom Training/CV ausgeschlossen
    und optional als Holdout-Test genutzt.
    """

    def __init__(
            self,
            preprocessor: CaseDataPreprocessor | None,
            rf_config: RFConfig | None = None,
            exclude_videos: Optional[Set[int]] = None,
    ):
        self.prep = preprocessor
        self.config = rf_config or RFConfig()
        self.exclude_videos: Set[int] = set(exclude_videos or [])
        self.model: Optional[RandomForestRegressor] = None

        # Daten-Container (Training/CV)
        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[np.ndarray] = None
        self.meta_: Optional[pd.DataFrame] = None

        # Holdout-Container (ausgeschlossene Videos)
        self.X_hold_: Optional[pd.DataFrame] = None
        self.y_hold_: Optional[np.ndarray] = None
        self.meta_hold_: Optional[pd.DataFrame] = None

    # --------------------------- Daten laden ---------------------------

    def prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Ruft den Preprocessor auf, filtert ausgeschlossene Videos heraus
        und legt (X, y, meta) intern ab.
        """
        assert self.prep is not None, "Preprocessor fehlt."
        X, yv, ya, meta = self.prep.prepare_all()
        y = np.column_stack([yv.values, ya.values]).astype(np.float32)

        if self.exclude_videos:
            mask_hold = meta["video"].isin(self.exclude_videos)
            # Holdout separat ablegen
            self.X_hold_ = X.loc[mask_hold].copy()
            self.y_hold_ = y[mask_hold].copy()
            self.meta_hold_ = meta.loc[mask_hold].copy()
            # Trainings-/CV-Daten: ohne die ausgeschlossenen Videos
            X = X.loc[~mask_hold].copy()
            y = y[~mask_hold].copy()
            meta = meta.loc[~mask_hold].copy()

        self.X_, self.y_, self.meta_ = X, y, meta
        return X, y, meta

    # ---------------------------- Evaluieren ---------------------------

    def cross_validate(
            self,
            n_splits: int = 5,
            verbose: bool = True
    ) -> Dict[str, float]:
        """
        Subjekt-weise Cross-Validation (GroupKFold).
        Gibt pro Fold und Durchschnitts-Scores aus.
        """
        assert self.X_ is not None and self.y_ is not None and self.meta_ is not None, \
            "Bitte zuerst prepare_data() aufrufen."

        groups = self.meta_["subject"].values

        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )

        scoring = {
            "R2": "r2",
            "MAE": "neg_mean_absolute_error",
            "Pearson": make_scorer(_pearson_multi, greater_is_better=True),
        }

        cv = GroupKFold(n_splits=n_splits)

        res = cross_validate(
            rf,
            self.X_.values,
            self.y_,
            cv=cv,
            groups=groups,
            scoring=scoring,
            n_jobs=self.config.n_jobs,
            return_train_score=False,
        )

        # Fold-Ergebnisse ausgeben
        if verbose:
            print("\n[CV] Einzel-Fold-Ergebnisse:")
            for i in range(n_splits):
                print(f"  Fold {i + 1}: "
                      f"R2={res['test_R2'][i]:.4f}, "
                      f"MAE={-res['test_MAE'][i]:.4f}, "
                      f"Pearson={res['test_Pearson'][i]:.4f}")

        # Durchschnitt berechnen
        out = {
            "R2": float(np.mean(res["test_R2"])),
            "MAE": float(-np.mean(res["test_MAE"])),
            "Pearson": float(np.mean(res["test_Pearson"]))
        }

        if verbose:
            print("\n[CV] Mittelwerte über alle Folds:",
                  {k: round(v, 4) for k, v in out.items()})
        return out

    # ------------------------------ Training ------------------------------

    def fit(self, verbose: bool = True) -> RandomForestRegressor:
        assert self.X_ is not None and self.y_ is not None, "Bitte zuerst prepare_data() aufrufen."
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )
        self.model.fit(self.X_.values, self.y_)
        if verbose:
            print("[FIT] RandomForest trainiert auf", self.X_.shape[0],
                  "Fenstern mit", self.X_.shape[1], "Features.")
        return self.model

    # --------------------------- Inferenz / Utils ---------------------------

    def predict(self, X_new: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self.model is not None, "Modell ist nicht trainiert. fit() aufrufen."
        Xv = X_new.values if isinstance(X_new, pd.DataFrame) else X_new
        return self.model.predict(Xv)

    def feature_importance(self, top_k: int | None = 20) -> pd.Series:
        assert self.model is not None, "Modell ist nicht trainiert."
        assert self.X_ is not None, "Keine Feature-Namen verfügbar."
        fi = pd.Series(self.model.feature_importances_, index=self.X_.columns).sort_values(ascending=False)
        return fi.head(top_k) if top_k is not None else fi

    # ------------------------------ Holdout-Test ------------------------------

    def evaluate_holdout(self) -> Dict[str, float]:
        assert self.model is not None, "Bitte zuerst fit() aufrufen."
        assert self.X_hold_ is not None and len(self.X_hold_) > 0, "Kein Holdout vorhanden."
        y_pred = self.predict(self.X_hold_)
        r2 = float(r2_score(self.y_hold_, y_pred))
        mae = float(mean_absolute_error(self.y_hold_, y_pred))
        pear = _pearson_multi(self.y_hold_, y_pred)
        out = {"R2": r2, "MAE": mae, "Pearson": pear}
        print("\n[HOLDOUT] Scores:", {k: round(v, 4) for k, v in out.items()})
        return out

    # ------------------------------ Persistenz ------------------------------

    def save(self, path: str | bytes | "os.PathLike[str]") -> None:
        assert self.model is not None, "Kein Modell zum Speichern vorhanden."
        joblib.dump(
            {
                "model": self.model,
                "feature_names": None if self.X_ is None else list(self.X_.columns),
                "rf_config": self.config,
            },
            path,
        )
        print(f"[SAVE] Modell gespeichert unter: {path}")

    @staticmethod
    def load(path: str | bytes | "os.PathLike[str]") -> "CaseRandomForest":
        bundle = joblib.load(path)
        obj = CaseRandomForest(preprocessor=None, rf_config=bundle.get("rf_config", RFConfig()))
        obj.model = bundle["model"]
        if "feature_names" in bundle:
            obj.X_ = pd.DataFrame(columns=bundle["feature_names"])
        print(f"[LOAD] Modell geladen von: {path}")
        return obj


# ============================== Beispiel-Usage ==============================
if __name__ == "__main__":
    base_path = ".."
    prep = CaseDataPreprocessor(
        base_path=base_path,
        fs=20,
        window_size=10,
        step_size=1,
        subjects=list(range(1, 29)),
        label_shift_s=0.0,
        use_video_as_feature=False,
    )

    rf_cfg = RFConfig(n_estimators=300, min_samples_leaf=3, max_depth=None, n_jobs=-1, random_state=42)
    rfw = CaseRandomForest(preprocessor=prep, rf_config=rf_cfg, exclude_videos={10, 11, 12})

    print("[INFO] Erzeuge Feature-Matrix & Labels ...")
    X, y, meta = rfw.prepare_data()
    print(f"[INFO] Shapes (TRAIN): X={X.shape}, y={y.shape}")
    if rfw.X_hold_ is not None:
        print(f"[INFO] Holdout: X_hold={rfw.X_hold_.shape}")

    print("[INFO] Starte subjekt-weise 5-Fold CV (ohne Videos 10/11/12) ...")
    cv_scores = rfw.cross_validate(n_splits=5)

    print("[INFO] Trainiere finales Modell ...")
    rfw.fit()

    print("\n[TOP-Feature-Importances]")
    print(rfw.feature_importance(top_k=20).to_string())

    if rfw.X_hold_ is not None and len(rfw.X_hold_) > 0:
        rfw.evaluate_holdout()

    rfw.save("rf_va_model.joblib")
