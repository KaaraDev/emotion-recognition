# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


@dataclass
class TrainConfig:
    features_csv: Path = Path("outputs/features_case_bvp_gsr_skt.csv")
    out_dir: Path = Path("outputs/model_random_tree")
    # Videos, die GAR NICHT verwendet werden sollen (z.B. Pausen/Blue)
    excluded_videos: List[int] = None  # z.B. [0, 99] – nach Bedarf füllen
    # Optional: falls du DOCH einen Hold-out möchtest
    holdout_videos: Optional[List[int]] = None
    random_state: int = 42
    n_estimators: int = 500
    min_samples_leaf: int = 3
    max_depth: int | None = None
    n_splits_cv: int = 5


def _load_data(cfg: TrainConfig) -> pd.DataFrame:
    if not cfg.features_csv.exists():
        raise FileNotFoundError(
            f"Features-Datei fehlt: {cfg.features_csv}. "
            "Bitte vorher extract_features_case() ausführen."
        )
    df = pd.read_csv(cfg.features_csv)

    # Safety: erwartete Metaspalten prüfen (werden unten genutzt)
    needed = {"subject", "video", "t_start_s", "t_end_s"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in Features: {missing}")

    return df


def train_random_tree(cfg: TrainConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Laden
    df = _load_data(cfg)
    n0 = len(df)

    # ------------------ Exkludierte Videos raus
    if cfg.excluded_videos:
        df = df[~df["video"].isin(cfg.excluded_videos)].copy()
    n1 = len(df)

    # ------------------ Features / Labels
    meta_cols = ["subject", "video", "t_start_s", "t_end_s"]
    num_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("Keine numerischen Feature-Spalten gefunden. Prüfe deine Feature-CSV.")

    # WICHTIG: KEIN dropna – Imputer macht das
    df_clean = df.copy()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_all = le.fit_transform(df_clean["video"].astype(str))
    X_all = df_clean[num_cols].to_numpy(dtype=float)
    groups_all = df_clean["subject"].to_numpy()

    # ------------------ Optionaler Holdout
    if cfg.holdout_videos:
        is_holdout = df_clean["video"].isin(cfg.holdout_videos).to_numpy()
    else:
        is_holdout = np.zeros(len(df_clean), dtype=bool)

    X_train, y_train, groups_train = X_all[~is_holdout], y_all[~is_holdout], groups_all[~is_holdout]
    X_test, y_test = X_all[is_holdout], y_all[is_holdout]

    # ------------------ Diagnostics
    print("[INFO] Zeilen: initial =", n0,
          "| nach exclude =", n1,
          "| train =", X_train.shape[0],
          "| holdout =", X_test.shape[0])
    print("[INFO] Verteilung Videos (train):")
    print(pd.Series(df_clean.loc[~is_holdout, "video"]).value_counts().sort_index().to_string())
    print("[INFO] Verteilung Subjekte (train):")
    print(pd.Series(df_clean.loc[~is_holdout, "subject"]).value_counts().sort_index().to_string())

    # ------------------ Safety-Checks
    if X_train.shape[0] == 0:
        raise ValueError(
            "Nach Filtern/Holdout sind keine Trainings-Samples übrig.\n"
            f"- excluded_videos: {cfg.excluded_videos}\n"
            f"- holdout_videos:  {cfg.holdout_videos}\n"
            "- Tipp: Entferne Holdout oder reduziere excluded_videos."
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Trainingsdaten enthalten nur eine Klasse (ein Video). "
            "Für Klassifikation werden mindestens 2 Klassen benötigt."
        )

    # ------------------ Pipeline
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=cfg.random_state
        ))
    ])

    # ------------------ CV (nur wenn genug Subjekte & Samples)
    uniq_groups = np.unique(groups_train)
    n_unique_groups = len(uniq_groups)
    do_cv = (n_unique_groups >= 2) and (X_train.shape[0] >= 2)

    if do_cv:
        n_splits = min(cfg.n_splits_cv, n_unique_groups)
        cv = GroupKFold(n_splits=n_splits)
        scoring = {
            "bal_acc": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted"
        }
        cv_res = cross_validate(
            pipe, X_train, y_train, groups=groups_train,
            scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False
        )
        print("\n=== CV (gruppiert nach subject) ===")
        for k, v in cv_res.items():
            if k.startswith("test_"):
                name = k.replace("test_", "")
                print(f"{name:>12s}:  mean={np.mean(v):.4f}  std={np.std(v):.4f}")
    else:
        cv_res = {}
        print("\n[WARN] CV übersprungen (zu wenige Subjekte oder Samples). "
              f"unique subjects = {n_unique_groups}, n_train = {X_train.shape[0]}")

    # ------------------ Fit auf Train
    pipe.fit(X_train, y_train)

    # ------------------ Feature Importances
    importances = pipe.named_steps["clf"].feature_importances_
    fi = pd.DataFrame({"feature": num_cols, "importance": importances}).sort_values("importance", ascending=False)
    fi_path = cfg.out_dir / "feature_importances.csv"
    fi.to_csv(fi_path, index=False)

    # ------------------ Optional: Holdout
    if X_test.size:
        y_pred = pipe.predict(X_test)
        y_pred_proba = pipe.predict_proba(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        print("\n=== Hold-out ===")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print("\nKlassifikationsbericht:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in le.classes_], columns=[f"pred_{c}" for c in le.classes_])
        cm_path = cfg.out_dir / "confusion_matrix_holdout.csv"
        cm_df.to_csv(cm_path)

        test_rows = df_clean[is_holdout].copy()
        test_rows["y_true"] = le.inverse_transform(y_test)
        test_rows["y_pred"] = le.inverse_transform(y_pred)
        for i, cls in enumerate(le.classes_):
            test_rows[f"proba_{cls}"] = y_pred_proba[:, i]
        pred_path = cfg.out_dir / "predictions_holdout.csv"
        test_rows.to_csv(pred_path, index=False)
    else:
        bal_acc = np.nan
        cm_path = cfg.out_dir / "confusion_matrix_holdout.csv"
        pred_path = cfg.out_dir / "predictions_holdout.csv"
        pd.DataFrame().to_csv(cm_path, index=False)
        pd.DataFrame().to_csv(pred_path, index=False)

    # ------------------ Persist
    model_path = cfg.out_dir / "random_forest_model.joblib"
    le_path = cfg.out_dir / "label_encoder.joblib"
    joblib.dump(pipe, model_path)
    joblib.dump(le, le_path)

    print("\n=== Artefakte gespeichert ===")
    print(f"Model:            {model_path.resolve()}")
    print(f"LabelEncoder:     {le_path.resolve()}")
    print(f"Feature Import.:  {fi_path.resolve()}")
    if X_test.size:
        print(f"Confusion Matrix: {cm_path.resolve()}")
        print(f"Predictions:      {pred_path.resolve()}")

    return {
        "model_path": model_path,
        "label_encoder_path": le_path,
        "feature_importances_path": fi_path,
        "confusion_matrix_path": cm_path,
        "predictions_path": pred_path,
        "cv_scores": {k.replace('test_', ''): float(np.mean(v)) for k, v in getattr(cv_res, "items", lambda: [])() if
                      k.startswith('test_')},
        "holdout_bal_acc": float(bal_acc) if not np.isnan(bal_acc) else None
    }


if __name__ == "__main__":
    cfg = TrainConfig(
        # Beispiel: Pausen/blau o.ä. rauswerfen
        excluded_videos=[0],  # <- HIER deine irrelevanten Videos eintragen
        holdout_videos=None  # <- None = kein Hold-out
    )
    train_random_tree(cfg)
