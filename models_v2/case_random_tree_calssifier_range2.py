# -*- coding: utf-8 -*-
"""
Train Random Forest Classifier (Valence–Arousal -> 4 Klassen via Ranges: amused, relaxed, scary, bored)
Robustes Setup mit Nested GroupKFold, Leakage-freier Pipeline & optionalen Features:

- Subjekt-getrennte Nested GroupKFold (inner: Tuning, outer: Evaluation; optional StratifiedGroupKFold)
- Kein Leakage: Imputer, optional QuantileTransformer, SMOTE in Pipeline (nur auf Train-Folds)
- Tuning-Scoring: f1_macro; Validierung auf Originalverteilung (ohne Resampling)
- Optional OOF-Regression (RFRegressor) für A/V als Zusatzfeatures (gruppen-bewusst)
- Optionales Makro-Aggregieren (z.B. 120s/60s) der Fenster -> Rauschreduktion
- Optional zusätzliche Video-Level-Evaluation (Mehrheitsvote pro video_id)
- Umfassendes Logging + ETA pro Tuning-Iteration
- Verbesserte Parameterauswahl fürs Refit (Mehrheits-/Score-basierte Auswahl; optionales Re-Tuning)
- Pooled classification_report + Confusion Matrix
- Optionale Permutation Importances

Spezifische Anpassung:
- 'neutral' wird strikt nur im Mittelrechteck vergeben und anschließend mit 'bored' zusammengelegt (neutral→bored).
- Downsampling: Klasse 'bored' wird auf 'median' der anderen Klassen reduziert (optional, per Flag standardmäßig AN).

Beispiel:
python train_rf_from_va_ranges_fixed.py --csv PATH/zu/features.csv --exclude_pauses --add_oof_av \
  --use_macro_agg --macro_seconds 120 --macro_hop 60 --eval_video_level
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import json
import logging
import sys
import time
import traceback
import random
from collections import Counter
from sklearn.utils._param_validation import Interval
from numbers import Integral

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, ParameterSampler, cross_validate
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, balanced_accuracy_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import clone

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGF = True
except Exception:
    HAS_SGF = False

from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.base import BaseSampler

import joblib


# ------------------------- Config -------------------------
@dataclass
class TrainConfig:
    csv_path: Path
    out_dir: Path
    random_state: int = 42
    log_level: str = "INFO"
    n_iter_tune: int = 50
    tune_cv_folds: int = 3
    eval_cv_folds: int = 5
    exclude_pauses: bool = True  # video_id 10/11/12 raus
    add_oof_av: bool = False
    smote_variant: str = "svm"  # "svm" oder "borderline"
    v_low: float = 4.0
    v_high: float = 6.0
    a_low: float = 4.0
    a_high: float = 6.0
    # Downsampling für 'bored' (neutral wurde in bored gemappt)
    downsample_bored: bool = True
    bored_target: str = "median"  # 'min' oder 'median'
    # Makro-Aggregation
    use_macro_agg: bool = False
    macro_seconds: int = 120
    macro_hop: int = 60
    # Video-Level-Evaluation
    eval_video_level: bool = False
    # Verbesserungen/Flags
    no_quantile: bool = False
    no_class_weight: bool = False
    use_stratified_groupkfold: bool = False  # optional, falls verfügbar
    # Refit-Verbesserung
    refit_retune_n_iter: int = 0  # >0 = kleines Re-Tuning auf allen Daten
    refit_retune_folds: int = 3
    # Permutation Importance (optional, teuer)
    perm_importance: bool = False
    perm_importance_n_repeats: int = 10


PHYS_FEATURES = [
    'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
    'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate',
    'gsr_mean', 'gsr_slope', 'skt_mean', 'skt_slope'
]
EXCLUDE_VIDEO_IDS = {"10", "11", "12"}  # Pausen konsistent
NEUTRAL_SYM = "neutral"  # interne Zwischenklasse beim Mapping

# Finale Emotionsliste (4 Klassen, neutral->bored zusammengelegt)
EMOTIONS = ["amused", "bored", "relaxed", "scary"]


# ------------------------- Label mapping (strikt) -------------------------
def va_to_label(valence: float, arousal: float,
                v_low: float, v_high: float,
                a_low: float, a_high: float) -> str:
    """
    Neutral NUR wenn:
      v_low <= valence <= v_high  UND  a_low <= arousal <= a_high
    Sonst: logische Zuordnung zu {amused, relaxed, scary, bored}.

    Logik außerhalb des Neutral-Bandes:
      - Beide Achsen 'high/low' -> Quadrant
      - Nur eine Achse außerhalb -> wähle anhand des Mittelpunkts der anderen Achse
    """
    if not np.isfinite(valence) or not np.isfinite(arousal):
        return NEUTRAL_SYM

    in_v = (v_low <= valence <= v_high)
    in_a = (a_low <= arousal <= a_high)

    # Neutral nur, wenn BEIDE innerhalb sind
    if in_v and in_a:
        return NEUTRAL_SYM

    # Bestimme Seite(n) außerhalb des Neutralbands
    v_side = ('low' if valence < v_low else ('high' if valence > v_high else None))
    a_side = ('low' if arousal < a_low else ('high' if arousal > a_high else None))

    # 1) Klassischer Quadrant: beide Achsen eindeutig
    if v_side and a_side:
        if v_side == 'low' and a_side == 'high':
            return "scary"
        if v_side == 'low' and a_side == 'low':
            return "bored"
        if v_side == 'high' and a_side == 'high':
            return "amused"
        if v_side == 'high' and a_side == 'low':
            return "relaxed"

    # 2) Nur EINE Achse außerhalb → entscheide über Mittelpunkt der anderen Achse
    mid_v = (v_low + v_high) / 2.0
    mid_a = (a_low + a_high) / 2.0

    if v_side and not a_side:
        return "amused" if (v_side == 'high' and arousal >= mid_a) else \
            "relaxed" if (v_side == 'high') else \
                "scary" if (arousal >= mid_a) else "bored"

    if a_side and not v_side:
        return "amused" if (a_side == 'high' and valence >= mid_v) else \
            "scary" if (a_side == 'high') else \
                "relaxed" if (valence >= mid_v) else "bored"

    # Fallback
    return "bored"


# ------------------------- Logging -------------------------
def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logger = logging.getLogger("rf_ranges_fixed")
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
def load_and_label(cfg: TrainConfig, logger: logging.Logger) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_csv(cfg.csv_path)
    logger.info("CSV loaded: %s rows, %s cols (%.2fs)", len(df), df.shape[1], time.perf_counter() - t0)

    req_cols = ['subject', 'arousal', 'valence']
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    if cfg.exclude_pauses and 'video_id' in df.columns:
        before = len(df)
        df = df.loc[~df['video_id'].astype(str).isin(EXCLUDE_VIDEO_IDS)].copy()
        logger.info("Excluded pauses (video_id in %s): %d -> %d rows", sorted(list(EXCLUDE_VIDEO_IDS)), before, len(df))

    # Erstes Mapping (inkl. Neutralrechteck)
    labels = [
        va_to_label(v, a, cfg.v_low, cfg.v_high, cfg.a_low, cfg.a_high)
        for v, a in zip(df['valence'].to_numpy(dtype=float), df['arousal'].to_numpy(dtype=float))
    ]
    df['label'] = pd.Series(labels, index=df.index)

    # 'neutral' -> 'bored' zusammenlegen
    before_counts = df['label'].value_counts().to_dict()
    df['label'] = df['label'].replace({NEUTRAL_SYM: 'bored'})
    after_counts = df['label'].value_counts().to_dict()
    logger.info("Merged neutral→bored | before=%s | after=%s", before_counts, after_counts)

    missing_feats = [c for c in PHYS_FEATURES if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")

    logger.info("Subjects=%d | Samples=%d | Classes=%s",
                df['subject'].nunique(), len(df), df['label'].value_counts().sort_index().to_dict())

    # Labelverteilung pro Subject (erste 5)
    try:
        subj_stats = []
        for sid, g in df.groupby('subject'):
            subj_stats.append((sid, g['label'].value_counts().to_dict()))
        logger.info("Label per subject (first 5): %s", dict(subj_stats[:5]))
    except Exception:
        pass

    keep_cols = ['subject'] + (['video_id'] if 'video_id' in df.columns else []) \
                + (['window_start_ms'] if 'window_start_ms' in df.columns else []) \
                + PHYS_FEATURES + ['arousal', 'valence', 'label']
    return df[keep_cols].copy()


# ------------------------- Downsampling 'bored' -------------------------
def downsample_class_df(df: pd.DataFrame, class_name: str, logger: logging.Logger, seed: int,
                        target: str = "median") -> pd.DataFrame:
    counts = df['label'].value_counts().to_dict()
    logger.info("[Downsampling:%s] Before: %s", class_name, counts)
    if class_name not in counts:
        logger.info("[Downsampling:%s] Keine Samples vorhanden. Überspringe.", class_name)
        return df

    other = {k: v for k, v in counts.items() if k != class_name}
    if not other:
        logger.warning("[Downsampling:%s] Nur Zielklasse vorhanden – kein Training möglich. Überspringe.", class_name)
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
                class_name, df_new['label'].value_counts().to_dict(), class_name, n_curr, target_n)
    return df_new


# ------------------------- Makro-Aggregation -------------------------
def _mode_label(labels: pd.Series) -> str:
    cnt = Counter(labels)
    if not cnt:
        return "bored"  # neutral existiert final nicht mehr
    most = cnt.most_common()
    if len(most) == 1 or (len(most) > 1 and most[0][1] > most[1][1]):
        return most[0][0]
    # Tie-break: ohne Neutral-Prior – wähle das erste der Top-Kandidaten deterministisch
    tied_labels = [l for l, n in most if n == most[0][1]]
    tied_labels.sort()  # deterministisch
    return tied_labels[0]


def aggregate_to_macro_windows(df: pd.DataFrame, seconds: int, hop: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Aggregiert 10s/1s-Fenster zu längeren Chunks (z.B. 120s/60s).
    - Mittelwert über PHYS_FEATURES (+ arousal/valence falls vorhanden)
    - Label = Mehrheit (Tie-break: deterministisch)
    - Gruppierung pro (subject, video_id); falls 'window_start_ms' vorhanden, zeitbasiert,
      sonst gleichmäßig in Blöcke partitioniert.
    """
    if 'video_id' not in df.columns:
        logger.warning("[MacroAgg] Keine video_id-Spalte vorhanden. Aggregation wird nur pro Subject gemacht.")
    use_time = 'window_start_ms' in df.columns

    rows = []
    group_cols = ['subject'] + (['video_id'] if 'video_id' in df.columns else [])
    for _, g in df.sort_values(group_cols + (['window_start_ms'] if use_time else [])).groupby(group_cols,
                                                                                               dropna=False):
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
                    row = {c: sub[c].mean() for c in PHYS_FEATURES + ['arousal', 'valence'] if c in sub.columns}
                    for c in group_cols: row[c] = sub[c].iloc[0]
                    row['macro_start_ms'] = start;
                    row['macro_end_ms'] = end
                    row['label'] = _mode_label(sub['label'])
                    rows.append(row)
                start += hop_ms
        else:
            n = len(g);
            approx_frames = max(seconds, 60);
            step = max(hop, 30);
            idx = 0
            while idx < n:
                sub = g.iloc[idx: idx + approx_frames]
                if len(sub) >= max(10, approx_frames // 5):
                    row = {c: sub[c].mean() for c in PHYS_FEATURES + ['arousal', 'valence'] if c in sub.columns}
                    for c in group_cols: row[c] = sub[c].iloc[0]
                    row['macro_start_ms'] = np.nan;
                    row['macro_end_ms'] = np.nan
                    row['label'] = _mode_label(sub['label']);
                    rows.append(row)
                idx += max(step, approx_frames // 2)

    if not rows:
        logger.warning("[MacroAgg] Keine aggregierten Fenster erzeugt. Rückfall auf Originaldaten.")
        return df

    df_macro = pd.DataFrame(rows)
    logger.info("[MacroAgg] Aggregation %ds/%ds: %d -> %d Zeilen", seconds, hop, len(df), len(df_macro))
    keep = ['subject'] + (['video_id'] if 'video_id' in df_macro.columns else []) + \
           ['macro_start_ms', 'macro_end_ms'] + PHYS_FEATURES + \
           (['arousal', 'valence'] if 'arousal' in df_macro.columns else []) + ['label']
    return df_macro[keep].copy()


# ------------------------- Optional OOF A/V as features -------------------------
def add_oof_av_features(df: pd.DataFrame, logger: logging.Logger, seed: int) -> pd.DataFrame:
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
                n_estimators=400, random_state=seed, n_jobs=-1, max_depth=None, min_samples_leaf=2))
        ])
        mask_tr_a = np.isfinite(y_a_all[tr_idx])
        if mask_tr_a.sum() >= 10:
            pipe_a.fit(X_all[tr_idx][mask_tr_a], y_a_all[tr_idx][mask_tr_a])
            pred_a[va_idx] = pipe_a.predict(X_all[va_idx])

        pipe_v = SkPipeline(steps=[
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('reg', RandomForestRegressor(
                n_estimators=400, random_state=seed + 1, n_jobs=-1, max_depth=None, min_samples_leaf=2))
        ])
        mask_tr_v = np.isfinite(y_v_all[tr_idx])
        if mask_tr_v.sum() >= 10:
            pipe_v.fit(X_all[tr_idx][mask_tr_v], y_v_all[tr_idx][mask_tr_v])
            pred_v[va_idx] = pipe_v.predict(X_all[va_idx])

        logging.getLogger("rf_ranges_fixed").info(
            "OOF AV fold %d done (n_val=%d, train A:%d, V:%d).",
            fold, len(va_idx), int(mask_tr_a.sum()), int(mask_tr_v.sum())
        )

    if np.isnan(pred_a).any():
        med_a = np.nanmedian(pred_a);
        pred_a = np.where(np.isnan(pred_a), med_a, pred_a)
        logging.getLogger("rf_ranges_fixed").warning("Filled NaNs in pred_arousal with median %.3f.", med_a)
    if np.isnan(pred_v).any():
        med_v = np.nanmedian(pred_v);
        pred_v = np.where(np.isnan(pred_v), med_v, pred_v)
        logging.getLogger("rf_ranges_fixed").warning("Filled NaNs in pred_valence with median %.3f.", med_v)

    df = df.copy()
    df['pred_arousal'] = pred_a;
    df['pred_valence'] = pred_v

    try:
        ca = np.corrcoef(df['arousal'].to_numpy(), df['pred_arousal'].to_numpy())[0, 1]
        cv = np.corrcoef(df['valence'].to_numpy(), df['pred_valence'].to_numpy())[0, 1]
        logging.getLogger("rf_ranges_fixed").info("OOF corr(arousal,pred)=%.3f | corr(valence,pred)=%.3f", ca, cv)
    except Exception:
        pass

    logging.getLogger("rf_ranges_fixed").info("Added OOF predicted features: pred_arousal, pred_valence.")
    return df


# ------------------------- RF Pipeline & Param Space -------------------------

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
                                                subsample=200000, random_state=cfg.random_state)))
    class_weight = None if cfg.no_class_weight else "balanced_subsample"
    steps.append(('smote', smote))
    steps.append(('rf', RandomForestClassifier(random_state=cfg.random_state, n_jobs=-1, class_weight=class_weight)))
    return ImbPipeline(steps=steps)


def param_distributions(cfg: TrainConfig) -> Dict[str, List]:
    return {
        'rf__n_estimators': [400, 600, 800, 1000, 1200],
        'rf__criterion': ['gini', 'entropy', 'log_loss'],
        'rf__max_features': ['sqrt', 'log2', None],
        'rf__max_depth': [20, 30, 40, 60, 80, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [2, 3, 4],
        'rf__bootstrap': [True, False],
        # 'smote__k_neighbors'  <- ENTFERNT (adaptive gesetzt)
    }


# ------------------------- Random Search mit Logs -------------------------
def random_search_with_logs(
        pipe, param_space: Dict[str, List], n_iter: int, scoring: str, cv, X, y, groups, logger, seed=42,
        label="[Tuning]"
):
    """
    Loggt pro Iteration f1_macro + accuracy. Akzeptiert als 'cv' entweder:
    - einen Splitter (mit .split) ODER
    - ein Iterable von (train_idx, test_idx)
    Wir materialisieren die Splits als Liste, damit sie in jeder Iteration wiederverwendbar sind.
    """
    # --- NEU: CV-Splits einmalig materialisieren ---
    if hasattr(cv, "split"):
        cv_splits = list(cv.split(X, y, groups=groups))
    else:
        cv_splits = list(cv)
    if len(cv_splits) == 0:
        raise RuntimeError("Inner-CV liefert 0 Splits – prüfe deine Gruppen/Folds.")

    # Immer beide Metriken evaluieren
    primary = scoring if isinstance(scoring, str) else "f1_macro"
    scoring_list = [primary, "accuracy"] if primary != "accuracy" else ["accuracy", "f1_macro"]

    sampler = list(ParameterSampler(param_space, n_iter=n_iter, random_state=seed))
    best_score = -np.inf
    best_params = None
    t_start = time.perf_counter()

    for i, params in enumerate(sampler, start=1):
        t0 = time.perf_counter()
        model = clone(pipe).set_params(**params)

        cv_res = cross_validate(
            model, X, y,
            groups=groups,
            cv=cv_splits,  # <- WICHTIG: die LISTE, nicht der Generator
            scoring=scoring_list,
            n_jobs=-1,
            return_train_score=False,
            error_score=np.nan  # robust: falls mal ein Fold scheitert
        )

        def _mean(key):
            import numpy as np
            vals = cv_res.get(f"test_{key}", [])
            return float(np.nanmean(vals)) if len(vals) else float("nan")

        f1 = _mean("f1_macro")
        acc = _mean("accuracy")
        score = f1 if primary == "f1_macro" else acc

        iter_time = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_start
        remain = (n_iter - i) * (elapsed / i)

        is_best = score > best_score
        if is_best:
            best_score = score
            best_params = params

        logger.info(
            f"{label} {i:3d}/{n_iter:<3d} ({100 * i / n_iter:.1f}%) | "
            f"f1={f1:.4f} | acc={acc:.4f}"
            f"{' (NEW BEST)' if is_best else ''} | best={best_score:.4f} "
            f"| iter={iter_time:.1f}s | elapsed={elapsed:.1f}s | ETA={remain:.1f}s"
        )

    best_est = clone(pipe).set_params(**best_params).fit(X, y)
    return best_est, best_params, best_score


# ------------------------- Helpers: Parameterauswahl -------------------------
def choose_consensus_params(best_params_per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not best_params_per_fold:
        raise RuntimeError("No best params available.")
    serialized = [json.dumps(item['params'], sort_keys=True) for item in best_params_per_fold]
    counts = Counter(serialized);
    top_ser, top_count = counts.most_common(1)[0]
    tied = [s for s, c in counts.items() if c == top_count]
    if len(tied) == 1:
        return json.loads(top_ser)
    score_map = {}
    for item in best_params_per_fold:
        s = json.dumps(item['params'], sort_keys=True)
        score_map[s] = max(score_map.get(s, -np.inf), float(item.get('cv_f1_macro', -np.inf)))
    best_ser = max(tied, key=lambda s: score_map.get(s, -np.inf))
    return json.loads(best_ser)


# ------------------------- Video-Level-Metriken -------------------------
def _video_level_metrics(df_va: pd.DataFrame, y_hat_va: np.ndarray, out_dir: Path, logger: logging.Logger) -> Dict[
    str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if 'video_id' not in df_va.columns:
        return {}
    tmp = df_va[['subject', 'video_id', 'label']].copy();
    tmp['pred'] = y_hat_va
    grouped = []
    for (_, vid), g in tmp.groupby(['subject', 'video_id']):
        y_true = _mode_label(g['label']);
        y_pred = _mode_label(g['pred']);
        grouped.append((y_true, y_pred))
    if not grouped:
        return {}
    y_true_g, y_pred_g = zip(*grouped)
    rep = classification_report(y_true_g, y_pred_g, digits=3, output_dict=False)
    acc = accuracy_score(y_true_g, y_pred_g);
    bacc = balanced_accuracy_score(y_true_g, y_pred_g)
    f1m = f1_score(y_true_g, y_pred_g, average='macro')
    with open(out_dir / 'report_video_level.txt', 'w', encoding='utf-8') as f:
        f.write(rep)
    logger.info("[Eval-Video] metrics: acc=%.4f | bAcc=%.4f | f1_macro=%.4f", acc, bacc, f1m)
    return {"video_level_acc": acc, "video_level_bacc": bacc, "video_level_f1m": f1m}


# ------------------------- CV-Factory -------------------------
def make_cv(cfg: TrainConfig, groups: np.ndarray, use_stratified: bool, n_splits: int):
    if use_stratified and HAS_SGF:
        return StratifiedGroupKFold(n_splits=n_splits)
    return GroupKFold(n_splits=n_splits)


# ------------------------- Nested CV (Eval + Tuning) -------------------------
def nested_cv_evaluate(df: pd.DataFrame, cfg: TrainConfig, logger: logging.Logger, out_dir: Path) -> Dict:
    feat_cols = [*PHYS_FEATURES]
    if {'pred_arousal', 'pred_valence'}.issubset(df.columns):
        feat_cols += ['pred_arousal', 'pred_valence']

    X = df[feat_cols].to_numpy(dtype=float)
    y = df['label'].to_numpy()
    groups = df['subject'].to_numpy()

    outer = make_cv(cfg, groups, cfg.use_stratified_groupkfold, cfg.eval_cv_folds)
    y_true_all, y_pred_all = [], []
    best_params_per_fold = []
    fold_metrics_rows = []

    start_all = time.perf_counter()
    logger.info("[Eval] Nested CV: %d outer folds, inner tuning %d iters x %d folds (scoring=f1_macro, stratified=%s)",
                cfg.eval_cv_folds, cfg.n_iter_tune, cfg.tune_cv_folds, str(cfg.use_stratified_groupkfold and HAS_SGF))

    for fold_idx, (tr_idx, va_idx) in enumerate(
            outer.split(X, y if (cfg.use_stratified_groupkfold and HAS_SGF) else None, groups=groups), start=1
    ):
        fold_t0 = time.perf_counter()
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        g_tr, g_va = groups[tr_idx], groups[va_idx]

        pipe = build_pipeline(cfg)
        inner_cv = make_cv(cfg, g_tr, cfg.use_stratified_groupkfold, cfg.tune_cv_folds)

        best_model, best_params, best_cv_score = random_search_with_logs(
            pipe=pipe,
            param_space=param_distributions(cfg),
            n_iter=cfg.n_iter_tune,
            scoring='f1_macro',
            cv=inner_cv.split(X_tr, y_tr if (cfg.use_stratified_groupkfold and HAS_SGF) else None, groups=g_tr),
            X=X_tr, y=y_tr, groups=g_tr,
            logger=logger,
            seed=cfg.random_state,
            label="[Tuning]"
        )
        logger.info("[Eval] Fold %d/%d tuning done. Best f1_macro(cv)=%.4f",
                    fold_idx, cfg.eval_cv_folds, best_cv_score)

        best_params_per_fold.append({'params': best_params, 'cv_f1_macro': float(best_cv_score)})

        # Vorhersage auf VALIDATION (Originalverteilung, kein Resampling)
        y_hat = best_model.predict(X_va)

        # Metriken
        acc = accuracy_score(y_va, y_hat)
        bacc = balanced_accuracy_score(y_va, y_hat)
        f1m = f1_score(y_va, y_hat, average='macro')
        rep = classification_report(y_va, y_hat, digits=3, output_dict=False)
        labels_sorted = sorted(np.unique(np.concatenate([y_va, y_hat])))
        cm = confusion_matrix(y_va, y_hat, labels=labels_sorted)
        pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(out_dir / f'cm_fold{fold_idx}.csv',
                                                                            index=True)

        with open(out_dir / f'report_fold{fold_idx}.txt', 'w', encoding='utf-8') as f:
            f.write(rep)

        y_true_all.append(y_va);
        y_pred_all.append(y_hat)
        df_va = df.iloc[va_idx].copy()
        if cfg.eval_video_level:
            fold_dir = out_dir / Path(f'fold{fold_idx}_');
            _video_level_metrics(df_va, y_hat, fold_dir, logger)

        fold_metrics_rows.append({'fold': fold_idx, 'accuracy': acc, 'balanced_accuracy': bacc, 'f1_macro': f1m})

        elapsed_total = time.perf_counter() - start_all
        avg_per_fold = elapsed_total / fold_idx
        remaining = max(cfg.eval_cv_folds - fold_idx, 0) * avg_per_fold
        logger.info("[Eval] Fold %d/%d done in %.1fs | elapsed=%.1fs | ETA=%.1fs",
                    fold_idx, cfg.eval_cv_folds, time.perf_counter() - fold_t0, elapsed_total, remaining)

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

    agg = {
        'accuracy_mean': float(fold_df['accuracy'].mean()),
        'accuracy_std': float(fold_df['accuracy'].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        'balanced_accuracy_mean': float(fold_df['balanced_accuracy'].mean()),
        'balanced_accuracy_std': float(fold_df['balanced_accuracy'].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        'f1_macro_mean': float(fold_df['f1_macro'].mean()),
        'f1_macro_std': float(fold_df['f1_macro'].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        'pooled_accuracy': float(acc_all),
        'pooled_balanced_accuracy': float(bacc_all),
        'pooled_f1_macro': float(f1m_all),
    }
    with open(out_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)

    labels_sorted = sorted(np.unique(np.concatenate([y_true_all, y_pred_all])))
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=labels_sorted)
    pd.DataFrame(cm_all, index=labels_sorted, columns=labels_sorted).to_csv(out_dir / 'confusion_matrix_pooled.csv',
                                                                            index=True)

    if cfg.eval_video_level and 'video_id' in df.columns:
        video_metrics = _video_level_metrics(df, y_pred_all, out_dir / Path('video_level_total'), logger)
        if video_metrics:
            with open(out_dir / 'metrics_video_level.json', 'w', encoding='utf-8') as f: json.dump(video_metrics, f,
                                                                                                   indent=2)

    with open(out_dir / 'best_params_per_fold.json', 'w', encoding='utf-8') as f:
        json.dump(best_params_per_fold, f, indent=2)

    return {'feat_cols': feat_cols, 'best_params_per_fold': best_params_per_fold, 'metrics_summary': agg}


# ------------------------- Final Refit -------------------------
def refit_on_all_data(df: pd.DataFrame, cfg: TrainConfig, best_params_per_fold: List[Dict], logger: logging.Logger,
                      out_dir: Path):
    if len(best_params_per_fold) == 0:
        raise RuntimeError("No best params from folds to refit.")
    chosen_params = choose_consensus_params(best_params_per_fold)
    logger.info("Chosen params (consensus): %s", chosen_params)

    feat_cols = [*PHYS_FEATURES]
    if {'pred_arousal', 'pred_valence'}.issubset(df.columns):
        feat_cols += ['pred_arousal', 'pred_valence']

    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df['label'].to_numpy()
    groups = df['subject'].to_numpy()

    if cfg.refit_retune_n_iter and cfg.refit_retune_n_iter > 0:
        pipe = build_pipeline(cfg);
        pipe.set_params(**chosen_params)
        cv = make_cv(cfg, groups, cfg.use_stratified_groupkfold, cfg.refit_retune_folds)
        logger.info("[Refit-Retune] Start mini random search: n_iter=%d, folds=%d", cfg.refit_retune_n_iter,
                    cfg.refit_retune_folds)
        best_model, best_params, best_cv_score = random_search_with_logs(
            pipe=pipe,
            param_space=param_distributions(cfg),
            n_iter=cfg.refit_retune_n_iter,
            scoring='f1_macro',
            cv=cv.split(X_all, y_all if (cfg.use_stratified_groupkfold and HAS_SGF) else None, groups=groups),
            X=X_all, y=y_all, groups=groups,
            logger=logger, seed=cfg.random_state, label="[Refit-Retune]"
        )
        chosen_params = best_params
        logger.info("[Refit-Retune] Best f1_macro(cv)=%.4f | params=%s", best_cv_score, chosen_params)

    pipe_final = build_pipeline(cfg);
    pipe_final.set_params(**chosen_params);
    pipe_final.fit(X_all, y_all)

    rf_step = pipe_final.named_steps['rf']
    importances = pd.DataFrame({'feature': feat_cols, 'importance': rf_step.feature_importances_}) \
        .sort_values('importance', ascending=False)
    importances.to_csv(out_dir / 'feature_importance_refit.csv', index=False)

    if cfg.perm_importance:
        logger.info("[PermutationImportance] Computing (n_repeats=%d) on 20%% random subset for speed.",
                    cfg.perm_importance_n_repeats)
        rng = np.random.RandomState(cfg.random_state)
        subset_size = max(2000, int(0.2 * len(X_all))) if len(X_all) > 0 else 0
        if subset_size > 0:
            idx = rng.choice(len(X_all), size=min(subset_size, len(X_all)), replace=False)
            try:
                result = permutation_importance(
                    pipe_final, X_all[idx], y_all[idx],
                    n_repeats=cfg.perm_importance_n_repeats, random_state=cfg.random_state, n_jobs=-1,
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
    with open(out_dir / 'final_model_params.json', 'w', encoding='utf-8') as f:
        json.dump(chosen_params, f, indent=2)
    logger.info("Final model refit on all data stored to %s", out_dir)


# ------------------------- CLI -------------------------
def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(
        description="RF Training aus Valence–Arousal Ranges (4 Klassen, neutral→bored) – robust (Aggregation, Video-Eval optional)")
    ap.add_argument('--csv', required=True, help='Pfad zur Feature-CSV (mit arousal, valence, subject, phys-Features)')
    ap.add_argument('--out_dir', default='outputs_10w1s/rf_ranges_fixed', help='Ausgabeordner')
    ap.add_argument('--seed', type=int, default=42, help='Random Seed')
    ap.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    ap.add_argument('--exclude_pauses', action='store_true',
                    help='Pausen-Videos (10/11/12) entfernen, falls video_id vorhanden')
    ap.add_argument('--add_oof_av', action='store_true', help='OOF-Prediction von A/V als Zusatzfeatures nutzen')
    ap.add_argument('--smote_variant', default='svm', choices=['svm', 'borderline'], help='SMOTE-Variante')
    # Ranges
    ap.add_argument('--v_low', type=float, default=4.0)
    ap.add_argument('--v_high', type=float, default=6.0)
    ap.add_argument('--a_low', type=float, default=4.0)
    ap.add_argument('--a_high', type=float, default=6.0)
    # Tuning/Eval
    ap.add_argument('--n_iter_tune', type=int, default=50)
    ap.add_argument('--tune_cv_folds', type=int, default=3)
    ap.add_argument('--eval_cv_folds', type=int, default=5)
    # Downsampling bored
    ap.add_argument('--downsample_bored', action='store_true', default=True,
                    help="Reduziert 'bored' vor dem Training (Ziel via --bored_target). Standard: an.")
    ap.add_argument('--bored_target', default='median', choices=['min', 'median'],
                    help="Zielgröße für 'bored' (nach Merge). Default: median.")
    # Makro-Aggregation
    ap.add_argument('--use_macro_agg', action='store_true',
                    help="Aggregiert Fenster zu langen Chunks (z.B. 120s/60s) vor dem Training.")
    ap.add_argument('--macro_seconds', type=int, default=120, help='Länge der Makrofenster in Sekunden.')
    ap.add_argument('--macro_hop', type=int, default=60, help='Schrittweite der Makrofenster in Sekunden.')
    # Video-Level
    ap.add_argument('--eval_video_level', action='store_true',
                    help="Zusätzliche Metriken auf Video-Level (Majority Vote pro (subject,video_id)).")
    # Flags
    ap.add_argument('--no_quantile', action='store_true', help='Deaktiviert den QuantileTransformer.')
    ap.add_argument('--no_class_weight', action='store_true', help='Deaktiviert class_weight im RF.')
    ap.add_argument('--use_stratified_groupkfold', action='store_true',
                    help='Falls verfügbar, nutze StratifiedGroupKFold.')
    # Refit-Verbesserung
    ap.add_argument('--refit_retune_n_iter', type=int, default=0, help='Mini-Retuning auf ALLEN Daten (0 = aus).')
    ap.add_argument('--refit_retune_folds', type=int, default=3, help='Folds für Mini-Retuning.')
    # Permutation Importance
    ap.add_argument('--perm_importance', action='store_true', help='Permutation Importances nach Refit berechnen.')
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
        v_low=float(args.v_low), v_high=float(args.v_high),
        a_low=float(args.a_low), a_high=float(args.a_high),
        n_iter_tune=int(args.n_iter_tune),
        tune_cv_folds=int(args.tune_cv_folds),
        eval_cv_folds=int(args.eval_cv_folds),
        downsample_bored=bool(args.downsample_bored),
        bored_target=str(args.bored_target),
        use_macro_agg=bool(args.use_macro_agg),
        macro_seconds=int(args.macro_seconds),
        macro_hop=int(args.macro_hop),
        eval_video_level=bool(args.eval_video_level),
        no_quantile=bool(args.no_quantile),
        no_class_weight=bool(args.no_class_weight),
        use_stratified_groupkfold=bool(args.use_stratified_groupkfold),
        refit_retune_n_iter=int(args.refit_retune_n_iter),
        refit_retune_folds=int(args.refit_retune_folds),
        perm_importance=bool(args.perm_importance),
        perm_importance_n_repeats=int(args.perm_importance_n_repeats),
    )


# ------------------------- Main -------------------------
if __name__ == "__main__":
    cfg = parse_args()
    logger = setup_logger(cfg.out_dir, cfg.log_level)
    logger.info("CONFIG: %s", vars(cfg))
    log_env_versions(logger)
    np.random.seed(cfg.random_state);
    random.seed(cfg.random_state)

    out_dir = cfg.out_dir;
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_label(cfg, logger)

    # (1) optional: Makro-Aggregation
    if cfg.use_macro_agg:
        df = aggregate_to_macro_windows(df, seconds=cfg.macro_seconds, hop=cfg.macro_hop, logger=logger)
        logger.info("After MacroAgg: Subjects=%d | Samples=%d | Classes=%s",
                    df['subject'].nunique(), len(df), df['label'].value_counts().sort_index().to_dict())

    # (2) Downsampling 'bored' auf median (Default: an)
    if cfg.downsample_bored:
        df = downsample_class_df(df, class_name="bored", logger=logger, seed=cfg.random_state,
                                 target=cfg.bored_target)
        logger.info("Class counts AFTER downsampling(bored): %s", df['label'].value_counts().to_dict())

    # (3) Optional: OOF A/V-Features
    if cfg.add_oof_av:
        df = add_oof_av_features(df, logger, cfg.random_state)

    # (4) Nested CV mit Tuning (SMOTE nur im Train)
    results = nested_cv_evaluate(df, cfg, logger, out_dir)

    # (5) Finaler Refit
    refit_on_all_data(df, cfg, results['best_params_per_fold'], logger, out_dir)

    # Artefakte
    feat_cols = [*PHYS_FEATURES]
    if {'pred_arousal', 'pred_valence'}.issubset(df.columns): feat_cols += ['pred_arousal', 'pred_valence']
    with open(out_dir / 'used_features.json', 'w', encoding='utf-8') as f:
        json.dump(feat_cols, f, indent=2)
    with open(out_dir / 'va_ranges.json', 'w', encoding='utf-8') as f:
        json.dump({"v_low": cfg.v_low, "v_high": cfg.v_high, "a_low": cfg.a_low, "a_high": cfg.a_high}, f, indent=2)

    logger.info("Done. See %s for metrics and artifacts.", out_dir.resolve())
