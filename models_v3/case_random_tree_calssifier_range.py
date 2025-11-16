# -*- coding: utf-8 -*-
"""
Train Random Forest Classifier (Video-ID -> feste Emotionslabels)
Klassen kommen direkt aus dem Video-Typ, NICHT mehr aus Valence/Arousal.

Mapping:
    video 1,2   -> amused
    video 3,4   -> bored
    video 5,6   -> relaxed
    video 7,8   -> scary
    video 10,11,12 werden entfernt

Wichtige Eigenschaften:
- GroupKFold über subjects für Evaluation (kein Nested CV, kein Tuning)
- Pipeline ohne Leakage:
    Imputer -> optional QuantileTransformer -> AdaptiveSMOTE(train-only) -> RF
- Optional:
    * Downsampling der Mehrheitsklasse ("bored") vor CV
    * OOF-Regression (arousal/valence aus Physio schätzen, als Zusatzfeatures)
    * Macro-Aggregation (lange Fenster, z.B. 120s/60s)
    * Video-Level-Evaluation (Mehrheitsvote pro (subject,video_id))
- Logging, Confusion-Matrix, classification_report etc.

Aufruf:
python train_rf_from_video_labels.py --csv PATH/zu/features.csv --exclude_pauses \
    --use_macro_agg --macro_seconds 120 --macro_hop 60 --eval_video_level --add_oof_av
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging
import sys
import time
import traceback
import random
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import clone

from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.base import BaseSampler

import joblib


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class TrainConfig:
    csv_path: Path
    out_dir: Path
    random_state: int = 42
    log_level: str = "INFO"

    exclude_pauses: bool = True  # video_id 10/11/12 raus
    add_oof_av: bool = False

    smote_variant: str = "svm"  # "svm" oder "borderline"
    no_quantile: bool = False
    no_class_weight: bool = False

    # Downsampling für 'bored' (Mehrheitsklasse)
    downsample_bored: bool = True
    bored_target: str = "median"  # 'min' oder 'median'

    # Makro-Aggregation
    use_macro_agg: bool = False
    macro_seconds: int = 120
    macro_hop: int = 60

    # Evaluation
    eval_cv_folds: int = 5
    eval_video_level: bool = False

    # Permutation Importance (optional)
    perm_importance: bool = False
    perm_importance_n_repeats: int = 10


PHYS_FEATURES = [
    'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
    'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate',
    'gsr_mean', 'gsr_slope', 'skt_mean', 'skt_slope'
]

# Welche Videos behalten?
VIDEO_TO_LABEL = {
    "1": "amused",
    "2": "amused",
    "3": "bored",
    "4": "bored",
    "5": "relaxed",
    "6": "relaxed",
    "7": "scary",
    "8": "scary",
    # 10,11,12 werden entfernt
}
EXCLUDE_VIDEO_IDS = {"10", "11", "12"}

EMOTIONS = ["amused", "bored", "relaxed", "scary"]


# ------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------

def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logger = logging.getLogger("rf_video_labels")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(sh)
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


# ------------------------------------------------------------------
# Daten laden + Label pro Video setzen
# ------------------------------------------------------------------

def load_and_label(cfg: TrainConfig, logger: logging.Logger) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_csv(cfg.csv_path)
    logger.info("CSV loaded: %s rows, %s cols (%.2fs)", len(df), df.shape[1], time.perf_counter() - t0)

    req_cols = ['subject', 'video_id']
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    # Pausen rauswerfen (10/11/12)
    before = len(df)
    df = df.loc[~df['video_id'].astype(str).isin(EXCLUDE_VIDEO_IDS)].copy()
    logger.info("Excluded pauses %s: %d -> %d rows", sorted(list(EXCLUDE_VIDEO_IDS)), before, len(df))

    # Video->Label Mapping anwenden, Zeilen ohne Mapping droppen
    df['label'] = df['video_id'].astype(str).map(VIDEO_TO_LABEL)
    before_map = len(df)
    df = df[df['label'].notna()].copy()
    logger.info("Applied video->label mapping. Dropped unmapped: %d -> %d rows", before_map, len(df))

    # Feature-Prüfung
    missing_feats = [c for c in PHYS_FEATURES if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")

    # Basic stats
    logger.info("Subjects=%d | Samples=%d | Classes=%s",
                df['subject'].nunique(), len(df), df['label'].value_counts().sort_index().to_dict())

    # Labelverteilung pro Subject (erste 5 Subjects nur fürs Log)
    try:
        subj_stats = []
        for sid, g in df.groupby('subject'):
            subj_stats.append((sid, g['label'].value_counts().to_dict()))
        logger.info("Label per subject (first 5): %s", dict(subj_stats[:5]))
    except Exception:
        pass

    keep_cols = ['subject', 'video_id'] + \
                (['window_start_ms'] if 'window_start_ms' in df.columns else []) + \
                PHYS_FEATURES + \
                (['arousal', 'valence'] if {'arousal', 'valence'}.issubset(df.columns) else []) + \
                ['label']
    return df[keep_cols].copy()


# ------------------------------------------------------------------
# Downsampling 'bored'
# ------------------------------------------------------------------

def downsample_class_df(df: pd.DataFrame, class_name: str, logger: logging.Logger, seed: int,
                        target: str = "median") -> pd.DataFrame:
    counts = df['label'].value_counts().to_dict()
    logger.info("[Downsampling:%s] Before: %s", class_name, counts)
    if class_name not in counts:
        logger.info("[Downsampling:%s] Klasse nicht vorhanden -> skip.", class_name)
        return df

    other = {k: v for k, v in counts.items() if k != class_name}
    if not other:
        logger.warning("[Downsampling:%s] Nur diese Klasse vorhanden -> skip.", class_name)
        return df

    if target == "min":
        target_n = int(min(other.values()))
    elif target == "median":
        target_n = int(np.median(list(other.values())))
    else:
        raise ValueError(f"Unknown target '{target}'. Supported: 'min','median'.")

    n_curr = counts[class_name]
    if n_curr <= target_n:
        logger.info("[Downsampling:%s] %d <= target %d. Nichts zu tun.", class_name, n_curr, target_n)
        return df

    df_tgt = df[df['label'] == class_name].sample(n=target_n, random_state=seed)
    df_other = df[df['label'] != class_name]
    df_new = pd.concat([df_other, df_tgt], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info("[Downsampling:%s] After: %s (%s: %d -> %d)",
                class_name,
                df_new['label'].value_counts().to_dict(),
                class_name, n_curr, target_n)
    return df_new


# ------------------------------------------------------------------
# Makro-Aggregation (lange Fenster mitteln, Label=Mehrheit)
# ------------------------------------------------------------------

def _mode_label(labels: pd.Series) -> str:
    cnt = Counter(labels)
    if not cnt:
        return "bored"
    most = cnt.most_common()
    if len(most) == 1 or (len(most) > 1 and most[0][1] > most[1][1]):
        return most[0][0]
    tied_labels = [l for l, n in most if n == most[0][1]]
    tied_labels.sort()
    return tied_labels[0]


def aggregate_to_macro_windows(df: pd.DataFrame, seconds: int, hop: int,
                               logger: logging.Logger) -> pd.DataFrame:
    """
    Aggregiert kurze Fenster zu längeren Makro-Fenstern (z.B. 120s/60s):
    - Mittelwert über PHYS_FEATURES (+ arousal/valence falls vorhanden)
    - Label = Mehrheit im Fenster
    - Gruppiert pro (subject, video_id); zeitlich über window_start_ms falls vorhanden
    """
    use_time = 'window_start_ms' in df.columns
    rows = []
    group_cols = ['subject', 'video_id']

    for _, g in df.sort_values(group_cols + (['window_start_ms'] if use_time else [])).groupby(group_cols):
        if use_time:
            t0 = g['window_start_ms'].min()
            win = seconds * 1000
            hop_ms = hop * 1000
            start = int(t0)
            tmax = int(g['window_start_ms'].max())
            while start <= tmax:
                end = start + win
                sub = g[(g['window_start_ms'] >= start) & (g['window_start_ms'] < end)]
                if len(sub) >= max(3, win // 10000):
                    row = {c: sub[c].mean() for c in PHYS_FEATURES if c in sub.columns}
                    if {'arousal', 'valence'}.issubset(sub.columns):
                        row['arousal'] = sub['arousal'].mean()
                        row['valence'] = sub['valence'].mean()
                    row['subject'] = sub['subject'].iloc[0]
                    row['video_id'] = sub['video_id'].iloc[0]
                    row['macro_start_ms'] = start
                    row['macro_end_ms'] = end
                    row['label'] = _mode_label(sub['label'])
                    rows.append(row)
                start += hop_ms
        else:
            # Fallback ohne window_start_ms -> chunkweise
            n = len(g)
            approx_frames = max(seconds, 60)
            step = max(hop, 30)
            idx = 0
            while idx < n:
                sub = g.iloc[idx: idx + approx_frames]
                if len(sub) >= max(10, approx_frames // 5):
                    row = {c: sub[c].mean() for c in PHYS_FEATURES if c in sub.columns}
                    if {'arousal', 'valence'}.issubset(sub.columns):
                        row['arousal'] = sub['arousal'].mean()
                        row['valence'] = sub['valence'].mean()
                    row['subject'] = sub['subject'].iloc[0]
                    row['video_id'] = sub['video_id'].iloc[0]
                    row['macro_start_ms'] = np.nan
                    row['macro_end_ms'] = np.nan
                    row['label'] = _mode_label(sub['label'])
                    rows.append(row)
                idx += max(step, approx_frames // 2)

    if not rows:
        logger.warning("[MacroAgg] Keine aggregierten Fenster erzeugt. Rückfall auf Originaldaten.")
        return df

    df_macro = pd.DataFrame(rows)
    logger.info("[MacroAgg] Aggregation %ds/%ds: %d -> %d Zeilen",
                seconds, hop, len(df), len(df_macro))

    keep = ['subject', 'video_id', 'macro_start_ms', 'macro_end_ms'] + PHYS_FEATURES
    if 'arousal' in df_macro.columns and 'valence' in df_macro.columns:
        keep += ['arousal', 'valence']
    keep += ['label']
    return df_macro[keep].copy()


# ------------------------------------------------------------------
# OOF A/V Features (gleich wie vorher)
# ------------------------------------------------------------------

def add_oof_av_features(df: pd.DataFrame, logger: logging.Logger, seed: int) -> pd.DataFrame:
    # Braucht arousal/valence Spalten
    if not {'arousal', 'valence'}.issubset(df.columns):
        logger.warning("OOF A/V requested but arousal/valence not in df. Skipping.")
        return df.copy()

    X_all = df[PHYS_FEATURES].to_numpy(dtype=float)
    y_a_all = df['arousal'].to_numpy(dtype=float)
    y_v_all = df['valence'].to_numpy(dtype=float)
    groups = df['subject'].to_numpy()

    gkf = GroupKFold(n_splits=5)
    pred_a = np.full(len(df), np.nan, dtype=float)
    pred_v = np.full(len(df), np.nan, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_a_all, groups), start=1):
        pipe_a = SkPipeline(steps=[
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('reg', RandomForestRegressor(
                n_estimators=400, random_state=seed, n_jobs=-1,
                max_depth=None, min_samples_leaf=2))
        ])
        mask_tr_a = np.isfinite(y_a_all[tr_idx])
        if mask_tr_a.sum() >= 10:
            pipe_a.fit(X_all[tr_idx][mask_tr_a], y_a_all[tr_idx][mask_tr_a])
            pred_a[va_idx] = pipe_a.predict(X_all[va_idx])

        pipe_v = SkPipeline(steps=[
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('reg', RandomForestRegressor(
                n_estimators=400, random_state=seed + 1, n_jobs=-1,
                max_depth=None, min_samples_leaf=2))
        ])
        mask_tr_v = np.isfinite(y_v_all[tr_idx])
        if mask_tr_v.sum() >= 10:
            pipe_v.fit(X_all[tr_idx][mask_tr_v], y_v_all[tr_idx][mask_tr_v])
            pred_v[va_idx] = pipe_v.predict(X_all[va_idx])

        logging.getLogger("rf_video_labels").info(
            "OOF AV fold %d done (n_val=%d, train A:%d, V:%d).",
            fold, len(va_idx), int(mask_tr_a.sum()), int(mask_tr_v.sum())
        )

    if np.isnan(pred_a).any():
        med_a = np.nanmedian(pred_a)
        pred_a = np.where(np.isnan(pred_a), med_a, pred_a)
        logging.getLogger("rf_video_labels").warning(
            "Filled NaNs in pred_arousal with median %.3f.", med_a)
    if np.isnan(pred_v).any():
        med_v = np.nanmedian(pred_v)
        pred_v = np.where(np.isnan(pred_v), med_v, pred_v)
        logging.getLogger("rf_video_labels").warning(
            "Filled NaNs in pred_valence with median %.3f.", med_v)

    df = df.copy()
    df['pred_arousal'] = pred_a
    df['pred_valence'] = pred_v

    try:
        ca = np.corrcoef(df['arousal'].to_numpy(), df['pred_arousal'].to_numpy())[0, 1]
        cv = np.corrcoef(df['valence'].to_numpy(), df['pred_valence'].to_numpy())[0, 1]
        logging.getLogger("rf_video_labels").info(
            "OOF corr(arousal,pred)=%.3f | corr(valence,pred)=%.3f", ca, cv)
    except Exception:
        pass

    logging.getLogger("rf_video_labels").info(
        "Added OOF predicted features: pred_arousal, pred_valence.")
    return df


# ------------------------------------------------------------------
# Adaptive SMOTE (Train-Fold-only Oversampling mit Fail-Safe)
# ------------------------------------------------------------------

from imblearn.base import BaseSampler
from sklearn.base import clone
from sklearn.utils._param_validation import Interval
from numbers import Integral


class AdaptiveSMOTE(BaseSampler):
    """
    SMOTE/BorderlineSMOTE/SVMSMOTE mit adaptivem k und sicheren Fallbacks:
    - Wenn im Train-Fold < 2 Klassen vorhanden: no-op (kein Resampling).
    - Wenn in der kleinsten Klasse <= 1 Samples: no-op.
    - k_neighbors = min(min_k, min_class_count-1)
    - Falls der interne Sampler trotzdem wirft: no-op mit Warnung.
    """
    _parameter_constraints = {
        "base_smote": [object],
        "min_k": [Interval(Integral, 1, None, closed="left")],
        "sampling_strategy": [object],
    }
    _sampling_type = "over-sampling"  # WICHTIG für imblearn.BaseSampler

    def __init__(self, base_smote, min_k=3, sampling_strategy="auto"):
        self.base_smote = base_smote
        self.min_k = min_k
        self.sampling_strategy = sampling_strategy
        self._effective_smote_ = None
        self._disabled_ = False

    def _fit_resample(self, X, y):
        import numpy as np
        y_arr = np.asarray(y)

        # 1) Wenn <2 Klassen: no-op
        classes, counts = np.unique(y_arr, return_counts=True)
        if classes.size < 2:
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y

        min_count = int(counts.min())
        # 2) Kleinste Klasse hat <=1 Sample: no-op
        if min_count <= 1:
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y

        # 3) k/m/n so wählen, dass sie zur kleinsten Klasse passen
        #    (m_neighbors existiert z.B. bei SVMSMOTE; n_neighbors bei manchen SMOTE-Varianten)
        max_k = max(1, min_count - 1)
        k = max(1, min(self.min_k, max_k))

        smote = clone(self.base_smote)
        set_params = {"k_neighbors": k, "sampling_strategy": self.sampling_strategy}

        # Falls vorhanden, m_neighbors und n_neighbors mit anpassen
        if hasattr(smote, "m_neighbors"):
            set_params["m_neighbors"] = min(getattr(smote, "m_neighbors", 10), k, max_k)
        if hasattr(smote, "n_neighbors"):
            set_params["n_neighbors"] = min(getattr(smote, "n_neighbors", 5), k, max_k)

        smote.set_params(**set_params)

        # 4) Defensive Absicherung
        try:
            X_res, y_res = smote.fit_resample(X, y)
            self._effective_smote_ = smote
            self._disabled_ = False
            return X_res, y_res
        except Exception as e:
            import logging
            logging.getLogger("rf_ranges_fixed").warning(
                "AdaptiveSMOTE fallback to no-op (k=%d, classes=%s, counts=%s): %r",
                k, classes.tolist(), counts.tolist(), e
            )
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y


def build_pipeline(cfg: TrainConfig) -> ImbPipeline:
    if cfg.smote_variant.lower() == "borderline":
        base = BorderlineSMOTE(random_state=cfg.random_state, k_neighbors=3, sampling_strategy="auto")
    else:
        base = SVMSMOTE(random_state=cfg.random_state, k_neighbors=3, sampling_strategy="auto")

    smote = AdaptiveSMOTE(base_smote=base, min_k=3, sampling_strategy="auto")

    steps = [('imp', SimpleImputer(strategy='median'))]
    if not cfg.no_quantile:
        steps.append(('qt', QuantileTransformer(output_distribution="normal",
                                                subsample=200000,
                                                random_state=cfg.random_state)))
    class_weight = None if cfg.no_class_weight else "balanced_subsample"
    steps.append(('smote', smote))
    steps.append(('rf', RandomForestClassifier(
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight=class_weight,
        n_estimators=800,
        max_depth=40,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        criterion="gini"
    )))
    return ImbPipeline(steps=steps)


# ------------------------------------------------------------------
# Video-Level-Metriken (Majority Vote pro (subject,video_id))
# ------------------------------------------------------------------

def video_level_metrics(df_va: pd.DataFrame, y_hat_va: np.ndarray,
                        out_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if 'video_id' not in df_va.columns:
        return {}
    tmp = df_va[['subject', 'video_id', 'label']].copy()
    tmp['pred'] = y_hat_va
    grouped = []
    for (_, vid), g in tmp.groupby(['subject', 'video_id']):
        y_true = _mode_label(g['label'])
        y_pred = _mode_label(g['pred'])
        grouped.append((y_true, y_pred))
    if not grouped:
        return {}
    y_true_g, y_pred_g = zip(*grouped)
    rep = classification_report(y_true_g, y_pred_g, digits=3, output_dict=False)
    acc = accuracy_score(y_true_g, y_pred_g)
    bacc = balanced_accuracy_score(y_true_g, y_pred_g)
    f1m = f1_score(y_true_g, y_pred_g, average='macro')
    with open(out_dir / 'report_video_level.txt', 'w', encoding='utf-8') as f:
        f.write(rep)
    logger.info("[Eval-Video] acc=%.4f | bAcc=%.4f | f1_macro=%.4f", acc, bacc, f1m)
    return {"video_level_acc": acc, "video_level_bacc": bacc, "video_level_f1m": f1m}


# ------------------------------------------------------------------
# Cross-Validation (kein Hyperparam-Tuning!)
# ------------------------------------------------------------------

def simple_group_cv_eval(df: pd.DataFrame, cfg: TrainConfig,
                         logger: logging.Logger, out_dir: Path) -> Dict[str, Any]:
    feat_cols = [*PHYS_FEATURES]
    if {'pred_arousal', 'pred_valence'}.issubset(df.columns):
        feat_cols += ['pred_arousal', 'pred_valence']

    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df['label'].to_numpy()
    groups_all = df['subject'].to_numpy()

    gkf = GroupKFold(n_splits=cfg.eval_cv_folds)

    y_true_all = []
    y_pred_all = []
    fold_metrics_rows = []

    start_all = time.perf_counter()
    logger.info("[Eval] Simple GroupKFold CV mit %d Folds (kein Tuning).", cfg.eval_cv_folds)

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups_all), start=1):
        fold_t0 = time.perf_counter()
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        pipe = build_pipeline(cfg)
        pipe.fit(X_tr, y_tr)

        y_hat = pipe.predict(X_va)

        acc = accuracy_score(y_va, y_hat)
        bacc = balanced_accuracy_score(y_va, y_hat)
        f1m = f1_score(y_va, y_hat, average='macro')

        rep = classification_report(y_va, y_hat, digits=3, output_dict=False)
        labels_sorted = sorted(np.unique(np.concatenate([y_va, y_hat])))
        cm = confusion_matrix(y_va, y_hat, labels=labels_sorted)
        pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted) \
            .to_csv(out_dir / f'cm_fold{fold_idx}.csv', index=True)

        with open(out_dir / f'report_fold{fold_idx}.txt', 'w', encoding='utf-8') as f:
            f.write(rep)

        y_true_all.append(y_va)
        y_pred_all.append(y_hat)

        df_va = df.iloc[va_idx].copy()
        if cfg.eval_video_level:
            fold_dir = out_dir / f'fold{fold_idx}_'
            video_level_metrics(df_va, y_hat, fold_dir, logger)

        fold_metrics_rows.append({
            'fold': fold_idx,
            'accuracy': acc,
            'balanced_accuracy': bacc,
            'f1_macro': f1m
        })

        elapsed_total = time.perf_counter() - start_all
        avg_per_fold = elapsed_total / fold_idx
        remaining = max(cfg.eval_cv_folds - fold_idx, 0) * avg_per_fold
        logger.info("[Eval] Fold %d/%d done in %.1fs | elapsed=%.1fs | ETA=%.1fs",
                    fold_idx, cfg.eval_cv_folds,
                    time.perf_counter() - fold_t0,
                    elapsed_total, remaining)

    # pooled
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    acc_all = accuracy_score(y_true_all, y_pred_all)
    bacc_all = balanced_accuracy_score(y_true_all, y_pred_all)
    f1m_all = f1_score(y_true_all, y_pred_all, average='macro')

    fold_df = pd.DataFrame(fold_metrics_rows)
    fold_df.to_csv(out_dir / 'metrics_per_fold.csv', index=False)

    pooled_report_txt = classification_report(y_true_all, y_pred_all, digits=3, output_dict=False)
    with open(out_dir / 'classification_report_pooled.txt', 'w', encoding='utf-8') as f:
        f.write(pooled_report_txt)

    labels_sorted = sorted(np.unique(np.concatenate([y_true_all, y_pred_all])))
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=labels_sorted)
    pd.DataFrame(cm_all, index=labels_sorted, columns=labels_sorted) \
        .to_csv(out_dir / 'confusion_matrix_pooled.csv', index=True)

    agg = {
        'pooled_accuracy': float(acc_all),
        'pooled_balanced_accuracy': float(bacc_all),
        'pooled_f1_macro': float(f1m_all),
        'accuracy_mean': float(fold_df['accuracy'].mean()),
        'accuracy_std': float(fold_df['accuracy'].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        'balanced_accuracy_mean': float(fold_df['balanced_accuracy'].mean()),
        'balanced_accuracy_std': float(fold_df['balanced_accuracy'].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        'f1_macro_mean': float(fold_df['f1_macro'].mean()),
        'f1_macro_std': float(fold_df['f1_macro'].std(ddof=1)) if len(fold_df) > 1 else 0.0,
    }
    with open(out_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)

    if cfg.eval_video_level and 'video_id' in df.columns:
        video_metrics = video_level_metrics(df, y_pred_all, out_dir / 'video_level_total', logger)
        if video_metrics:
            with open(out_dir / 'metrics_video_level.json', 'w', encoding='utf-8') as f:
                json.dump(video_metrics, f, indent=2)

    return {'feat_cols': feat_cols}


# ------------------------------------------------------------------
# Finaler Refit auf allen Daten mit fixen RF-Params
# ------------------------------------------------------------------

def refit_on_all_data(df: pd.DataFrame, cfg: TrainConfig,
                      logger: logging.Logger, out_dir: Path):
    feat_cols = [*PHYS_FEATURES]
    if {'pred_arousal', 'pred_valence'}.issubset(df.columns):
        feat_cols += ['pred_arousal', 'pred_valence']

    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df['label'].to_numpy()

    pipe_final = build_pipeline(cfg)
    pipe_final.fit(X_all, y_all)

    rf_step = pipe_final.named_steps['rf']
    importances = pd.DataFrame({
        'feature': feat_cols,
        'importance': rf_step.feature_importances_,
    }).sort_values('importance', ascending=False)
    importances.to_csv(out_dir / 'feature_importance_refit.csv', index=False)

    # Optional permutation importance (grobe Variante)
    if cfg.perm_importance:
        logger.info("[PermutationImportance] Computing (n_repeats=%d) on 20%% subset.",
                    cfg.perm_importance_n_repeats)
        rng = np.random.RandomState(cfg.random_state)
        subset_size = max(2000, int(0.2 * len(X_all))) if len(X_all) > 0 else 0
        if subset_size > 0:
            idx = rng.choice(len(X_all), size=min(subset_size, len(X_all)), replace=False)
            from sklearn.inspection import permutation_importance
            try:
                result = permutation_importance(
                    pipe_final, X_all[idx], y_all[idx],
                    n_repeats=cfg.perm_importance_n_repeats,
                    random_state=cfg.random_state,
                    n_jobs=-1,
                    scoring='f1_macro'
                )
                perm_df = pd.DataFrame({
                    'feature': feat_cols,
                    'perm_importance_mean': result.importances_mean,
                    'perm_importance_std': result.importances_std
                }).sort_values('perm_importance_mean', ascending=False)
                perm_df.to_csv(out_dir / 'permutation_importance_refit.csv', index=False)
                logger.info("[PermutationImportance] Done.")
            except Exception as e:
                logger.warning("Permutation importance failed: %s", e)
        else:
            logger.warning("Permutation importance skipped (empty dataset).")

    joblib.dump(pipe_final, out_dir / 'final_model_pipeline.joblib')
    logger.info("Final model refit on all data stored to %s", out_dir)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(
        description="RF Training mit Video-Labels (1/2 amused, 3/4 bored, 5/6 relaxed, 7/8 scary)."
    )
    ap.add_argument('--csv', required=True,
                    help='Pfad zur Feature-CSV (braucht subject, video_id, phys-Features)')
    ap.add_argument('--out_dir', default='outputs_10w1s/rf_video_labels',
                    help='Ausgabeordner')
    ap.add_argument('--seed', type=int, default=42, help='Random Seed')
    ap.add_argument('--log_level', default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    ap.add_argument('--exclude_pauses', action='store_true',
                    help='Videos 10/11/12 entfernen (Standard: an)')
    ap.add_argument('--add_oof_av', action='store_true',
                    help='OOF-Prediction von A/V als Zusatzfeatures nutzen (falls arousal/valence da sind)')

    ap.add_argument('--smote_variant', default='svm',
                    choices=['svm', 'borderline'],
                    help='SMOTE-Variante')

    ap.add_argument('--no_quantile', action='store_true',
                    help='QuantileTransformer deaktivieren')
    ap.add_argument('--no_class_weight', action='store_true',
                    help='class_weight im RF deaktivieren')

    # Downsampling bored
    ap.add_argument('--downsample_bored', action='store_true', default=True,
                    help="Reduziert 'bored' vor dem Training (Ziel via --bored_target). Standard: an.")
    ap.add_argument('--bored_target', default='median',
                    choices=['min', 'median'],
                    help="Zielgröße für 'bored' (nach Downsampling). Default: median.")

    # Makro-Aggregation
    ap.add_argument('--use_macro_agg', action='store_true',
                    help="Aggregiert kurze Fenster zu Makrofenstern (z.B. 120s/60s) VOR dem Training.")
    ap.add_argument('--macro_seconds', type=int, default=120,
                    help='Länge der Makrofenster in Sekunden.')
    ap.add_argument('--macro_hop', type=int, default=60,
                    help='Schrittweite der Makrofenster in Sekunden.')

    # Evaluation
    ap.add_argument('--eval_cv_folds', type=int, default=5,
                    help='Folds für GroupKFold')
    ap.add_argument('--eval_video_level', action='store_true',
                    help="Video-Level-Metriken (Majority Vote pro (subject,video_id)).")

    # Permutation Importance
    ap.add_argument('--perm_importance', action='store_true',
                    help='Permutation Importances nach finalem Refit berechnen.')
    ap.add_argument('--perm_importance_n_repeats', type=int, default=10)

    args = ap.parse_args()
    return TrainConfig(
        csv_path=Path(args.csv),
        out_dir=Path(args.out_dir),
        random_state=int(args.seed),
        log_level=args.log_level,
        exclude_pauses=bool(args.exclude_pauses),
        add_oof_av=bool(args.add_oof_av),
        smote_variant=str(args.smote_variant),
        no_quantile=bool(args.no_quantile),
        no_class_weight=bool(args.no_class_weight),
        downsample_bored=bool(args.downsample_bored),
        bored_target=str(args.bored_target),
        use_macro_agg=bool(args.use_macro_agg),
        macro_seconds=int(args.macro_seconds),
        macro_hop=int(args.macro_hop),
        eval_cv_folds=int(args.eval_cv_folds),
        eval_video_level=bool(args.eval_video_level),
        perm_importance=bool(args.perm_importance),
        perm_importance_n_repeats=int(args.perm_importance_n_repeats),
    )


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()
    logger = setup_logger(cfg.out_dir, cfg.log_level)
    logger.info("CONFIG: %s", vars(cfg))
    log_env_versions(logger)

    np.random.seed(cfg.random_state)
    random.seed(cfg.random_state)

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Laden + Video→Label
    df = load_and_label(cfg, logger)

    # 2) Optional Makro-Aggregation
    if cfg.use_macro_agg:
        df = aggregate_to_macro_windows(
            df,
            seconds=cfg.macro_seconds,
            hop=cfg.macro_hop,
            logger=logger
        )
        logger.info("After MacroAgg: Subjects=%d | Samples=%d | Classes=%s",
                    df['subject'].nunique(),
                    len(df),
                    df['label'].value_counts().sort_index().to_dict())

    # 3) Downsampling bored
    if cfg.downsample_bored:
        df = downsample_class_df(
            df,
            class_name="bored",
            logger=logger,
            seed=cfg.random_state,
            target=cfg.bored_target
        )
        logger.info("Class counts AFTER downsampling(bored): %s",
                    df['label'].value_counts().to_dict())

    # 4) Optional: OOF A/V Features
    if cfg.add_oof_av:
        df = add_oof_av_features(df, logger, cfg.random_state)

    # 5) CV ohne Tuning
    results = simple_group_cv_eval(df, cfg, logger, out_dir)

    # 6) Final Refit
    refit_on_all_data(df, cfg, logger, out_dir)

    # 7) Artefakte sichern
    feat_cols = [*PHYS_FEATURES]
    if {'pred_arousal', 'pred_valence'}.issubset(df.columns):
        feat_cols += ['pred_arousal', 'pred_valence']

    with open(out_dir / 'used_features.json', 'w', encoding='utf-8') as f:
        json.dump(feat_cols, f, indent=2)

    logger.info("Done. See %s for metrics and artifacts.", out_dir.resolve())
