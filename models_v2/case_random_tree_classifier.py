# -*- coding: utf-8 -*-
"""
Train Random Forest Classifier (Masterarbeit-Setup) auf CASE-Features
mit umfangreichem Logging.

Siehe: python train_random_forest.py --help
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import json
import time
import logging
import sys
import traceback
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

# ------------------------- Config -------------------------
@dataclass
class TrainConfig:
    csv_path: Path
    out_dir: Path
    label_col: Optional[str] = None
    map_from_video_id: bool = False
    n_iter: int = 60
    cv_folds: int = 5
    random_state: int = 42
    log_level: str = "INFO"


# Default-Mapping: Video-ID -> Label (bei Bedarf anpassen)
VIDEOID_TO_LABEL: Dict[str, str] = {
    "1": "amusement",
    "2": "amusement",
    "3": "boring",
    "4": "boring",
    "5": "relaxed",
    "6": "relaxed",
    "7": "scary",
    "8": "scary",
    "10": "start",
    "11": "blue",
    "12": "end"
}

# ------------------------- Logging Helpers -------------------------

def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    logger = logging.getLogger("rf_train")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logger initialisiert. Level=%s, Datei=%s", level.upper(), log_path)
    return logger


def log_env_versions(logger: logging.Logger):
    try:
        import sklearn, imblearn
        logger.info("Versionen: scikit-learn=%s | imbalanced-learn=%s | pandas=%s | numpy=%s",
                    sklearn.__version__, imblearn.__version__, pd.__version__, np.__version__)
    except Exception:  # pragma: no cover
        logger.warning("Konnte Versionen nicht loggen: %s", traceback.format_exc(limit=1))


def log_dataset_overview(df: pd.DataFrame, label_col: str, logger: logging.Logger):
    logger.info("Dataset-Shape: %s", df.shape)
    if 'subject' in df.columns:
        n_subj = df['subject'].nunique()
        logger.info("Subjects: %d | Samples: %d", n_subj, len(df))
        subj_sizes = df['subject'].value_counts().sort_index()
        logger.debug("Samples pro Subject (Top 10): %s", subj_sizes.head(10).to_dict())
    class_counts = df[label_col].value_counts().sort_index()
    logger.info("Klassenverteilung: %s", class_counts.to_dict())

    num_cols = [c for c in df.columns if c not in (label_col, 'subject')]
    na_rate = df[num_cols].isna().mean().sort_values(ascending=False)
    if na_rate.max() > 0:
        logger.warning("NaN-Anteile pro Feature (Top 10): %s", na_rate.head(10).to_dict())
    inf_any = np.isinf(df[num_cols].to_numpy()).any()
    if inf_any:
        logger.warning("INF/-INF in numerischen Features gefunden!")

    desc = df[num_cols].describe().T[['mean', 'std', 'min', 'max']]
    logger.debug("Feature-Stats (Auszug):\n%s", desc.head(10).to_string())


def log_cv_folds(gkf: GroupKFold,
                 X: pd.DataFrame,
                 y: np.ndarray,
                 groups: np.ndarray,
                 logger: logging.Logger):
    logger.info("GroupKFold mit %d Folds (Subject-Transfer).", gkf.n_splits)
    for i, (_, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        y_val = y[val_idx]
        g_val = groups[val_idx]
        cls_dist = pd.Series(y_val).value_counts().sort_index().to_dict()
        logger.info("Fold %d: val_samples=%d | val_subjects=%d | Klassen=%s",
                    i, len(val_idx), len(np.unique(g_val)), cls_dist)


# ------------------------- Loading & Prep -------------------------

def load_dataset(cfg: TrainConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, str]:
    t0 = time.perf_counter()
    df = pd.read_csv(cfg.csv_path)
    logger.info("CSV geladen: %s (%.2fs)", cfg.csv_path, time.perf_counter() - t0)

    # Label-Spalte ermitteln/erstellen
    if cfg.label_col and cfg.label_col in df.columns:
        label_col = cfg.label_col
        logger.info("Label-Spalte verwendet: '%s'", label_col)
    elif cfg.map_from_video_id and 'video_id' in df.columns:
        if not VIDEOID_TO_LABEL:
            raise ValueError("VIDEOID_TO_LABEL ist leer. Bitte Mapping definieren oder --label_col angeben.")
        df['label'] = df['video_id'].map(lambda x: VIDEOID_TO_LABEL.get(str(x), np.nan))
        n_missing = int(df['label'].isna().sum())
        if n_missing > 0:
            bad = df.loc[df['label'].isna(), 'video_id'].astype(str).value_counts().to_dict()
            raise ValueError(f"{n_missing} Zeilen ohne Mapping von video_id -> label. Fehlende IDs: {bad}")
        label_col = 'label'
        logger.info("Label aus video_id gemappt. Beispiel-Mapping: %s -> %s",
                    next(iter(VIDEOID_TO_LABEL.keys())), next(iter(VIDEOID_TO_LABEL.values())))
    else:
        if 'label' in df.columns:
            label_col = 'label'
            logger.info("Label-Spalte automatisch erkannt: 'label'")
        else:
            cols = ', '.join(df.columns[:10])
            raise ValueError(
                f"Keine Label-Spalte gefunden. Entweder --label_col angeben oder --map_from_video_id nutzen. "
                f"Verfügbare Spalten: {cols} …")

    # Feature-Spalten
    feat_cols = [
        'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
        'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate',
        'gsr_mean', 'gsr_slope', 'skt_mean', 'skt_slope'
    ]
    missing_feats = [c for c in feat_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Fehlende Feature-Spalten in CSV: {missing_feats}")

    if 'subject' not in df.columns:
        raise ValueError("Spalte 'subject' fehlt (wird für GroupKFold benötigt).")

    # Nur benötigte Spalten behalten
    keep_cols = ['subject', label_col] + feat_cols
    df = df[keep_cols].copy()

    # Label normalisieren
    df[label_col] = df[label_col].astype(str)

    # Überblick loggen
    log_dataset_overview(df, label_col, logger)

    return df.rename(columns={label_col: 'label'}), 'label'


# ------------------------- Model Pipeline -------------------------

def make_pipeline_and_space(random_state: int):
    numeric_features = [
        'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
        'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate',
        'gsr_mean', 'gsr_slope', 'skt_mean', 'skt_slope'
    ]

    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler())
        ]), numeric_features)
    ], remainder='drop')

    clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    pipe = ImbPipeline(steps=[
        ('pre', pre),
        ('smote', BorderlineSMOTE(kind='borderline-1', random_state=random_state)),
        ('clf', clf)
    ])

    param_space = {
        'smote__k_neighbors': [3, 4, 5, 7],
        'clf__n_estimators': [200, 400, 600, 800, 1200, 1600],
        'clf__max_depth': [None, 20, 30, 40, 60, 80],
        'clf__min_samples_split': [2, 4, 6, 10],
        'clf__min_samples_leaf': [1, 2, 3, 4],
        'clf__max_features': ['sqrt', 'log2', 0.5, 0.7],
        'clf__criterion': ['gini', 'entropy']
    }
    return pipe, param_space


# ------------------------- Train & Evaluate -------------------------

def train_and_evaluate(df: pd.DataFrame, cfg: TrainConfig, logger: logging.Logger):
    X = df.drop(columns=['label', 'subject'])
    y = df['label'].values
    groups = df['subject'].values

    pipe, param_space = make_pipeline_and_space(cfg.random_state)
    gkf = GroupKFold(n_splits=cfg.cv_folds)

    log_cv_folds(gkf, X, y, groups, logger)

    logger.info("Starte RandomizedSearchCV: n_iter=%d, scoring=accuracy", cfg.n_iter)
    t0 = time.perf_counter()
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_space,
        n_iter=cfg.n_iter,
        scoring='accuracy',
        n_jobs=-1,
        cv=gkf.split(X, y, groups=groups),
        refit=True,
        random_state=cfg.random_state,
        verbose=1,
        return_train_score=True
    )

    search.fit(X, y)
    dt = time.perf_counter() - t0
    logger.info("RandomizedSearchCV fertig in %.2fs", dt)
    logger.info("Beste Params: %s", json.dumps(search.best_params_, ensure_ascii=False))
    logger.info("Bestes CV-Score (mean_test_score): %.4f", search.best_score_)

    best_model = search.best_estimator_

    logger.info("Erzeuge OOF-Vorhersagen (GroupKFold) für Report …")
    t1 = time.perf_counter()
    y_pred = cross_val_predict(best_model, X, y,
                               cv=gkf.split(X, y, groups=groups),
                               n_jobs=-1, method='predict')
    logger.info("OOF-Vorhersagen fertig in %.2fs", time.perf_counter() - t1)

    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average='macro')
    logger.info("OOF Accuracy: %.4f | OOF F1-macro: %.4f", acc, f1m)

    report = classification_report(y, y_pred, digits=3)
    logger.debug("Classification Report:\n%s", report)

    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    logger.info("Konfusionsmatrix (Zeile=Ist, Spalte=Vorhersage):\n%s", cm_df.to_string())

    # Feature Importances
    rf: RandomForestClassifier = best_model.named_steps['clf']
    feat_cols = list(X.columns)
    importances = pd.DataFrame({
        'feature': feat_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info("Top-10 Feature Importances: %s",
                importances.head(10).round(6).to_dict(orient='records'))

    # Ergebnisse speichern
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.out_dir / 'best_params.json', 'w', encoding='utf-8') as f:
        json.dump(search.best_params_, f, indent=2)
    with open(cfg.out_dir / 'cv_results.json', 'w', encoding='utf-8') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in search.cv_results_.items()}, f)

    with open(cfg.out_dir / 'metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-macro: {f1m:.4f}\n\n")
        f.write(report)

    cm_df.to_csv(cfg.out_dir / 'confusion_matrix.csv')
    importances.to_csv(cfg.out_dir / 'feature_importance.csv', index=False)

    joblib.dump(best_model, cfg.out_dir / 'random_forest_model.joblib')
    logger.info("Artefakte gespeichert unter: %s", cfg.out_dir.resolve())

    # Kurzer Abschluss
    print("\n===== Ergebnisse =====")
    print(f"Best Params: {search.best_params_}")
    print(f"Accuracy (OOF): {acc:.4f}")
    print(f"F1-macro (OOF): {f1m:.4f}")
    print("\nClassification Report:\n", report)
    print("Gespeichert unter:", cfg.out_dir)


# ------------------------- CLI -------------------------

def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train Random Forest (Masterarbeit-Setup) auf CASE-Features (mit Logging)")
    ap.add_argument('--csv', required=True, help='Pfad zur Feature-CSV (aus der Extraktion)')
    ap.add_argument('--out_dir', default='outputs/rf_master', help='Ausgabeordner')
    ap.add_argument('--label_col', default=None, help='Name der Labelspalte (falls vorhanden)')
    ap.add_argument('--map_from_video_id', action='store_true', help='Label aus video_id per Mapping erzeugen')
    ap.add_argument('--n_iter', type=int, default=60, help='Anzahl RandomizedSearch-Iterationen')
    ap.add_argument('--cv', type=int, default=5, help='Anzahl Folds für GroupKFold')
    ap.add_argument('--seed', type=int, default=42, help='Random Seed')
    ap.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging-Stufe')
    args = ap.parse_args()

    return TrainConfig(
        csv_path=Path(args.csv),
        out_dir=Path(args.out_dir),
        label_col=args.label_col,
        map_from_video_id=args.map_from_video_id,
        n_iter=args.n_iter,
        cv_folds=args.cv,
        random_state=args.seed,
        log_level=args.log_level
    )


if __name__ == '__main__':
    cfg = parse_args()
    logger = setup_logger(cfg.out_dir, cfg.log_level)
    log_env_versions(logger)
    np.random.seed(cfg.random_state)
    try:
        df, label_col = load_dataset(cfg, logger)
        train_and_evaluate(df, cfg, logger)
    except Exception as e:
        logger.error("Training fehlgeschlagen: %s", e)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        sys.exit(1)
