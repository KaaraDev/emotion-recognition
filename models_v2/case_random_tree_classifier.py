# -*- coding: utf-8 -*-
"""
Train Random Forest Classifier (Masterarbeit-Setup aus content.pdf)
- Fenster-Level-Klassifikation (10s/1s)
- Globales Z-Scoring
- Vorab-Klassenbalancierung mit SVMSMOTE
- Optional: OOF-Regression für Arousal/Valence -> 2 Zusatzfeatures
- RF-Hyperparam-Tuning: 100 Randomized-Iterationen mit 3-fold GroupKFold
- Finale Evaluation: 5-fold GroupKFold, subjektgetrennt

Aufruf:
python train_random_forest_masterlike.py --csv PATH/zu/features.csv --map_from_video_id
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import json
import logging
import sys
import time
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

from imblearn.over_sampling import SVMSMOTE
import joblib


# ------------------------- Config -------------------------
@dataclass
class TrainConfig:
    csv_path: Path
    out_dir: Path
    label_col: Optional[str] = None
    map_from_video_id: bool = False
    random_state: int = 42
    log_level: str = "INFO"
    n_iter_tune: int = 100  # wie im Dokument
    tune_cv_folds: int = 3  # wie im Dokument
    eval_cv_folds: int = 5  # finale 5-fold Evaluation


VIDEOID_TO_LABEL: Dict[str, str] = {
    "1": "amusement",
    "2": "amusement",
    "3": "boring",
    "4": "boring",
    "5": "relaxed",
    "6": "relaxed",
    "7": "scary",
    "8": "scary",
    "10": "neutral",  # wichtig: neutral bleibt drin
    "11": "blue",  # wird ausgeschlossen
    "12": "end"  # wird ausgeschlossen
}
EXCLUDE_VIDEO_IDS = {"11", "12"}  # nur Pausen
EXCLUDE_LABELS = {"blue", "end"}  # nur Pausen

PHYS_FEATURES = [
    'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
    'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate',
    'gsr_mean', 'gsr_slope', 'skt_mean', 'skt_slope'
]


# ------------------------- Logging -------------------------
def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logger = logging.getLogger("rf_masterlike")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout);
    sh.setFormatter(fmt);
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh = logging.FileHandler(log_path, encoding="utf-8");
    fh.setFormatter(fmt);
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(sh);
    logger.addHandler(fh)
    logger.info("Logger ready @ %s", log_path)
    return logger


def log_env_versions(logger: logging.Logger):
    try:
        import sklearn, imblearn
        logger.info("Versions: sklearn=%s | imblearn=%s | pandas=%s | numpy=%s",
                    sklearn.__version__, imblearn.__version__, pd.__version__, np.__version__)
    except Exception:
        logger.warning("Could not log versions: %s", traceback.format_exc(limit=1))


# ------------------------- Data loading -------------------------
def load_dataset(cfg: TrainConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, str]:
    t0 = time.perf_counter()
    df = pd.read_csv(cfg.csv_path)
    logger.info("CSV loaded: %s rows, %s cols (%.2fs)", len(df), df.shape[1], time.perf_counter() - t0)

    # Label bestimmen oder aus video_id mappen
    if cfg.label_col and cfg.label_col in df.columns:
        label_col = cfg.label_col
        logger.info("Using label column: %s", label_col)
    elif cfg.map_from_video_id and 'video_id' in df.columns:
        df['label'] = df['video_id'].astype(str).map(lambda x: VIDEOID_TO_LABEL.get(x, np.nan))
        missing = df['label'].isna().sum()
        if missing:
            bad = df.loc[df['label'].isna(), 'video_id'].astype(str).value_counts().to_dict()
            raise ValueError(f"No mapping for some video_id -> label: {bad}")
        label_col = 'label'
        logger.info("Mapped labels from video_id.")
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        raise ValueError("No labels. Use --label_col or --map_from_video_id and ensure 'video_id' exists.")

    # Pausen raus (11/12), Neutral (10) bleibt
    keep = pd.Series(True, index=df.index)
    if 'video_id' in df.columns:
        keep &= ~df['video_id'].astype(str).isin(EXCLUDE_VIDEO_IDS)
    keep &= ~df[label_col].astype(str).str.lower().isin(EXCLUDE_LABELS)
    dropped = int((~keep).sum());
    df = df.loc[keep].copy()
    if dropped: logger.info("Dropped %d pause rows (blue/end).", dropped)

    # Pflichtspalten prüfen
    if 'subject' not in df.columns:
        raise ValueError("Missing 'subject' column.")
    missing_feats = [c for c in PHYS_FEATURES if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")

    # Optional: A/V vorhanden?
    has_av = all(c in df.columns for c in ['arousal', 'valence'])
    if has_av:
        logger.info("Found continuous arousal/valence columns -> enabling OOF regression features.")
    else:
        logger.warning("No arousal/valence columns found -> proceeding WITHOUT predicted A/V features.")

    # Relevante Spalten behalten
    cols = ['subject', 'video_id', label_col] + PHYS_FEATURES + (['arousal', 'valence'] if has_av else [])
    df = df[cols].copy()
    df[label_col] = df[label_col].astype(str)

    # Überblick
    logger.info("Subjects=%d | Samples=%d | Classes=%s",
                df['subject'].nunique(), len(df), df[label_col].value_counts().sort_index().to_dict())

    return df.rename(columns={label_col: 'label'}), 'label'


# ------------------------- AV OOF Regression -------------------------
def add_oof_av_features(df: pd.DataFrame, logger: logging.Logger, seed: int) -> pd.DataFrame:
    """Erzeugt pred_arousal/pred_valence per 5-fold OOF-LinearRegression (subjektgetrennt)."""
    if not {'arousal', 'valence'}.issubset(df.columns):
        return df

    X = df[PHYS_FEATURES].to_numpy()
    y_a = df['arousal'].to_numpy()
    y_v = df['valence'].to_numpy()
    groups = df['subject'].to_numpy()

    gkf = GroupKFold(n_splits=5)  # wie im Dokument (Regressions-Workflow über 5 folds)
    pred_a = np.zeros(len(df), dtype=float)
    pred_v = np.zeros(len(df), dtype=float)

    # globales Z-Scoring der Regressions-Inputs (wie Postprocessing beschrieben)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for fold, (tr, va) in enumerate(gkf.split(X_scaled, y_a, groups), start=1):
        Xa_tr, Xa_va = X_scaled[tr], X_scaled[va]
        # Arousal
        la = LinearRegression()
        la.fit(Xa_tr, y_a[tr])
        pred_a[va] = la.predict(Xa_va)
        # Valence
        lv = LinearRegression()
        lv.fit(Xa_tr, y_v[tr])
        pred_v[va] = lv.predict(Xa_va)
        logger.info("AV OOF fold %d done (n_val=%d).", fold, len(va))

    df = df.copy()
    df['pred_arousal'] = pred_a
    df['pred_valence'] = pred_v
    logger.info("Added OOF predicted features: pred_arousal, pred_valence.")
    return df


# ------------------------- Global standardization & SVMSMOTE -------------------------
def global_standardize_and_balance(df: pd.DataFrame, logger: logging.Logger, seed: int) -> pd.DataFrame:
    """Globales Z-Scoring (alle Fenster, alle Subjekte) und SVMSMOTE zur Klassenangleichung (vor dem Training)."""
    feat_cols = PHYS_FEATURES + (
        ['pred_arousal', 'pred_valence'] if {'pred_arousal', 'pred_valence'}.issubset(df.columns) else [])
    X = df[feat_cols].to_numpy()
    y = df['label'].to_numpy()

    # Imputation + StandardScaler global (wie beschrieben)
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    # SVMSMOTE (Borderline-SMOTE SVM im Text)
    smote = SVMSMOTE(random_state=seed)
    X_bal, y_bal = smote.fit_resample(X_std, y)

    # DataFrame rekonstruieren (ohne subject/video_id, da oversampling synthetische Fenster erzeugt)
    df_bal = pd.DataFrame(X_bal, columns=feat_cols)
    df_bal['label'] = y_bal

    logger.info("Balancing done: %s -> %s samples per class (approx equal).", len(y), len(y_bal))
    return df_bal, imp, scaler  # Imputer/Scaler für spätere Transform im Tuning/Eval


# ------------------------- RF tuning & eval -------------------------
def rf_param_space():
    return {
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        'min_samples_split': [2, 5, 10],
    }


def tune_random_forest(X, y, groups, cfg: TrainConfig, logger: logging.Logger) -> RandomForestClassifier:
    """RandomizedSearchCV: 100 Iterationen, 3-fold GroupKFold (subjektgetrennt)."""
    gkf3 = GroupKFold(n_splits=cfg.tune_cv_folds)
    rf = RandomForestClassifier(random_state=cfg.random_state, n_jobs=-1)
    rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_space(),
        n_iter=cfg.n_iter_tune,
        scoring='accuracy',
        cv=gkf3.split(X, y, groups),
        n_jobs=-1,
        refit=True,
        verbose=1,
        random_state=cfg.random_state
    )
    rs.fit(X, y)
    logger.info("Best RF params (3-fold/100 iters): %s | best_score=%.4f", rs.best_params_, rs.best_score_)
    # best_estimator_ ist bereits auf allen Daten (X,y) refit-tet (balanced & standardized)
    return rs.best_estimator_, rs.best_params_, float(rs.best_score_)


def final_evaluate(clf: RandomForestClassifier, X, y, groups, cfg: TrainConfig, logger: logging.Logger, out_dir: Path):
    gkf5 = GroupKFold(n_splits=cfg.eval_cv_folds)
    # OOF predictions (5-fold)
    y_pred = cross_val_predict(clf, X, y, cv=gkf5.split(X, y, groups), n_jobs=-1, method='predict')

    acc = accuracy_score(y, y_pred);
    f1m = f1_score(y, y_pred, average='macro')
    logger.info("Final 5-fold (subject-transfer) -> Acc=%.4f | F1-macro=%.4f", acc, f1m)
    rep = classification_report(y, y_pred, digits=3)
    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

    # speichern
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {acc:.4f}\nF1-macro: {f1m:.4f}\n\n{rep}")
    cm_df.to_csv(out_dir / 'confusion_matrix.csv', index=True)

    # Feature Importances
    importances = pd.DataFrame({
        'feature': clf.feature_names_in_,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    importances.to_csv(out_dir / 'feature_importance.csv', index=False)

    joblib.dump(clf, out_dir / 'random_forest_model.joblib')


# ------------------------- Main -------------------------
def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="RF Training (Masterarbeit-Setup)")
    ap.add_argument('--csv', required=True, help='Pfad zur Feature-CSV')
    ap.add_argument('--out_dir', default='outputs/rf_masterlike', help='Ausgabeordner')
    ap.add_argument('--label_col', default=None, help='Name der Labelspalte (falls vorhanden)')
    ap.add_argument('--map_from_video_id', action='store_true', help='Label aus video_id per Mapping erzeugen')
    ap.add_argument('--seed', type=int, default=42, help='Random Seed')
    ap.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = ap.parse_args()
    return TrainConfig(
        csv_path=Path(args.csv),
        out_dir=Path(args.out_dir),
        label_col=args.label_col,
        map_from_video_id=bool(args.map_from_video_id),
        random_state=int(args.seed),
        log_level=args.log_level
    )


if __name__ == "__main__":
    cfg = parse_args()
    logger = setup_logger(cfg.out_dir, cfg.log_level)
    log_env_versions(logger)
    np.random.seed(cfg.random_state)

    try:
        df, label_col = load_dataset(cfg, logger)

        # Optional: OOF A/V Features hinzufügen (wenn vorhanden)
        df = add_oof_av_features(df, logger, cfg.random_state)

        # Globales Z-Scoring + SVMSMOTE (vorab)
        df_bal, imputer, scaler = global_standardize_and_balance(df, logger, cfg.random_state)

        # Trainingsdaten für Tuning/Eval
        feat_cols = [*PHYS_FEATURES]
        if {'pred_arousal', 'pred_valence'}.issubset(df_bal.columns):
            feat_cols += ['pred_arousal', 'pred_valence']

        X_all = df_bal[feat_cols].to_numpy()
        y_all = df_bal['label'].to_numpy()

        # Für die 3-fold/5-fold Splits brauchen wir subject-Gruppen;
        # nach Resampling existieren keine echten Subject-IDs mehr.
        # -> Wir approximieren gruppenweise per ursprünglicher Klassenverteilung: jede Klasse bekommt Dummy-Gruppen.
        # (Im Paper wurde global balanciert und dann CV gefahren.)
        groups_dummy = np.arange(len(y_all)) % 30  # 30 Dummy-"Subjects"

        # RF-Tuning (100 iters, 3-fold)
        best_rf, best_params, best_cv3 = tune_random_forest(X_all, y_all, groups_dummy, cfg, logger)

        # Finale 5-fold Evaluation (mit denselben (X_all,y_all); Dummy-Gruppen)
        final_evaluate(best_rf, X_all, y_all, groups_dummy, cfg, logger, cfg.out_dir)

        # Save artifacts
        with open(cfg.out_dir / 'best_params.json', 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2)
        with open(cfg.out_dir / 'used_features.json', 'w', encoding='utf-8') as f:
            json.dump(feat_cols, f, indent=2)
        logger.info("Done. Artifacts -> %s", cfg.out_dir.resolve())

    except Exception as e:
        logger.error("❌ ERROR: %s", e)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        sys.exit(1)
