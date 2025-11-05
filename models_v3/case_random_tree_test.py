# -*- coding: utf-8 -*-

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib

# ---------------------------------------------------------
# AdaptiveSMOTE muss VOR joblib.load() definiert sein,
# exakt gleich benannt wie im Training.
# ---------------------------------------------------------
from imblearn.base import BaseSampler
from sklearn.base import clone
from sklearn.utils._param_validation import Interval
from numbers import Integral


class AdaptiveSMOTE(BaseSampler):
    """
    Gleiche Klasse wie im Training:
    - wrappt SVMSMOTE / BorderlineSMOTE usw.
    - passt k_neighbors dynamisch an kleinste Klasse an
    - no-op fallback falls Resampling nicht möglich ist
    """
    _parameter_constraints = {
        "base_smote": [object],
        "min_k": [Interval(Integral, 1, None, closed="left")],
        "sampling_strategy": [object],
    }
    _sampling_type = "over-sampling"

    def __init__(self, base_smote, min_k=3, sampling_strategy="auto"):
        self.base_smote = base_smote
        self.min_k = min_k
        self.sampling_strategy = sampling_strategy
        self._effective_smote_ = None
        self._disabled_ = False

    def _fit_resample(self, X, y):
        y_arr = np.asarray(y)

        # Weniger als 2 Klassen -> kein Oversampling
        classes, counts = np.unique(y_arr, return_counts=True)
        if classes.size < 2:
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y

        # Kleinste Klasse hat <=1 Sample -> kein Oversampling
        min_count = int(counts.min())
        if min_count <= 1:
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y

        max_k = max(1, min_count - 1)
        k = max(1, min(self.min_k, max_k))

        smote = clone(self.base_smote)
        set_params = {
            "k_neighbors": k,
            "sampling_strategy": self.sampling_strategy,
        }
        if hasattr(smote, "m_neighbors"):
            set_params["m_neighbors"] = min(getattr(smote, "m_neighbors", 10), k, max_k)
        if hasattr(smote, "n_neighbors"):
            set_params["n_neighbors"] = min(getattr(smote, "n_neighbors", 5), k, max_k)

        smote.set_params(**set_params)

        try:
            X_res, y_res = smote.fit_resample(X, y)
            self._effective_smote_ = smote
            self._disabled_ = False
            return X_res, y_res
        except Exception as e:
            logging.getLogger("VideoLabelEvaluator").warning(
                "AdaptiveSMOTE fallback to no-op (k=%d, classes=%s, counts=%s): %r",
                k, classes.tolist(), counts.tolist(), e
            )
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y


# ---------------------------------------------------------
# Konstanten: Features & Label-Mapping wie im Training
# ---------------------------------------------------------

PHYS_FEATURES = [
    'bvp_bpm', 'bvp_ibi', 'bvp_sdnn', 'bvp_sdsd', 'bvp_rmssd', 'bvp_pnn20', 'bvp_pnn50',
    'bvp_mad', 'bvp_sd1', 'bvp_sd2', 'bvp_s', 'bvp_sd1sd2', 'bvp_breathingrate',
    'gsr_mean', 'gsr_slope', 'skt_mean', 'skt_slope'
]

VIDEO_TO_LABEL = {
    "1": "amused",
    "2": "amused",
    "3": "bored",
    "4": "bored",
    "5": "relaxed",
    "6": "relaxed",
    "7": "scary",
    "8": "scary",
    # Pausen fliegen raus
    "10": None,
    "11": None,
    "12": None,
}


def _majority_label(labels: pd.Series) -> str:
    """
    Mehrheit der Labels in einer Gruppe,
    bei Tie alphabetisch deterministisch,
    sonst fallback 'bored' falls leer.
    """
    counts = labels.value_counts()
    if counts.empty:
        return "bored"
    max_n = counts.max()
    tied = sorted(counts[counts == max_n].index.tolist())
    return tied[0]


# ---------------------------------------------------------
# Logger Helper
# ---------------------------------------------------------

def make_logger() -> logging.Logger:
    logger = logging.getLogger("VideoLabelEvaluator")
    if logger.handlers:
        return logger  # schon konfiguriert

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------
# Evaluator-Klasse
# ---------------------------------------------------------

class VideoLabelEvaluator:
    def __init__(
        self,
        model_path: Path,
        df_full: pd.DataFrame,
        subjects_to_test: List[int],
        out_dir: Path,
        logger: logging.Logger = None
    ):
        """
        model_path:
            Pfad zur gespeicherten Pipeline (final_model_pipeline.joblib)

        df_full:
            kompletter Datensatz (alle Subjects),
            muss mindestens enthalten:
              - 'subject'
              - 'video_id'
              - alle PHYS_FEATURES (+ evtl. pred_arousal/pred_valence)

        subjects_to_test:
            Liste mit Subject-IDs, die evaluiert werden sollen

        out_dir:
            Ausgabe-Ordner für Reports
        """
        self.logger = logger or make_logger()

        self.model_path = Path(model_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Loading model from %s", self.model_path)
        loaded_obj = joblib.load(self.model_path)

        # A) komplette Pipeline?
        if hasattr(loaded_obj, "predict") and hasattr(loaded_obj, "predict_proba"):
            pipeline = loaded_obj
        # B) dict, Pipeline darin?
        elif isinstance(loaded_obj, dict):
            cand = None
            for k, v in loaded_obj.items():
                if hasattr(v, "predict") and hasattr(v, "predict_proba"):
                    cand = v
                    break
            if cand is None:
                raise ValueError(
                    "Geladenes joblib-Objekt ist ein dict, aber enthält keine Pipeline mit predict/predict_proba."
                )
            pipeline = cand
        else:
            raise ValueError(
                "Unbekanntes Modellformat. Weder Pipeline noch dict mit Pipeline geladen."
            )

        self.pipeline = pipeline

        # ----------------- Test-Subset bauen -----------------
        df_test = df_full[df_full['subject'].astype(int).isin(subjects_to_test)].copy()
        if len(df_test) == 0:
            self.logger.warning("No rows found for subjects %s", subjects_to_test)

        if 'video_id' not in df_test.columns:
            raise ValueError("Test-Data braucht eine 'video_id' Spalte für Labelmapping.")

        df_test['video_id'] = df_test['video_id'].astype(str)

        if 'label' not in df_test.columns:
            df_test['label'] = df_test['video_id'].map(VIDEO_TO_LABEL)

        before_len = len(df_test)
        df_test = df_test[df_test['label'].notna()].copy()
        self.logger.info(
            "After mapping video_id->label & dropping unmapped: %d -> %d rows",
            before_len, len(df_test)
        )

        self.df_test = df_test

        # ----------------- Features bestimmen -----------------
        self.feat_cols = list(PHYS_FEATURES)
        if {'pred_arousal', 'pred_valence'}.issubset(self.df_test.columns):
            self.feat_cols += ['pred_arousal', 'pred_valence']
            self.logger.info("Using pred_arousal/pred_valence in eval.")
        else:
            self.logger.info("No pred_arousal/pred_valence in eval data (that's fine).")

        missing = [c for c in self.feat_cols if c not in self.df_test.columns]
        if missing:
            raise ValueError(f"Missing required feature columns in test set: {missing}")

        self.X_test = self.df_test[self.feat_cols].to_numpy(dtype=float)
        self.y_test = self.df_test['label'].to_numpy()

        # ----------------- Klassenreihenfolge -----------------
        rf_step = None
        if hasattr(self.pipeline, "named_steps") and "rf" in self.pipeline.named_steps:
            rf_step = self.pipeline.named_steps["rf"]
        else:
            if hasattr(self.pipeline, "classes_"):
                rf_step = self.pipeline

        if rf_step is None:
            raise ValueError("Konnte den RandomForestClassifier im Modell nicht finden ('rf').")

        self.model_classes_ = rf_step.classes_
        self.logger.info("Model classes_: %s", self.model_classes_.tolist())

        # Debug-Info rausschreiben
        with open(self.out_dir / "used_features_eval.json", "w", encoding="utf-8") as f:
            json.dump(self.feat_cols, f, indent=2)

    # ----------------- Fenster-Level Eval -----------------
    def evaluate_window_level(self) -> Dict[str, Any]:
        self.logger.info("[WindowEval] Predicting window-level labels ...")
        y_pred = self.pipeline.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        bacc = balanced_accuracy_score(self.y_test, y_pred)
        f1m = f1_score(self.y_test, y_pred, average='macro')

        rep_txt = classification_report(self.y_test, y_pred, digits=3, output_dict=False)

        labels_sorted = sorted(np.unique(np.concatenate([self.y_test, y_pred])))
        cm = confusion_matrix(self.y_test, y_pred, labels=labels_sorted)

        # Confusion Matrix speichern
        pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
            self.out_dir / "cm_window_level.csv", index=True
        )
        with open(self.out_dir / "report_window_level.txt", "w", encoding="utf-8") as f:
            f.write(rep_txt)

        out = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "f1_macro": float(f1m),
            "labels_sorted": [str(x) for x in labels_sorted],
            "confusion_matrix": cm.astype(int).tolist(),
        }
        with open(self.out_dir / "metrics_window_level.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        self.logger.info(
            "[WindowEval] acc=%.4f | bAcc=%.4f | f1_macro=%.4f",
            acc, bacc, f1m
        )
        return out

    # ----------------- Video-Level Eval -----------------
    def evaluate_video_level_probs(self) -> Dict[str, Any]:
        """
        Für jedes (subject, video_id):
        - nimm alle Fenster,
        - sum(mehrere Fenster pro Klasse von predict_proba),
        - normalisiere diese Summe,
        - argmax => Video-Prediction,
        - ground truth = Mehrheitslabel der Fenster.
        """
        if 'video_id' not in self.df_test.columns:
            raise ValueError("Need 'video_id' column for video-level evaluation.")

        self.logger.info("[VideoEval] Predicting probabilities ...")
        proba_all = self.pipeline.predict_proba(self.X_test)  # shape: [N_windows, n_classes]
        classes_model = self.model_classes_
        prob_cols = [f"prob_{c}" for c in classes_model]

        df_prob = self.df_test.reset_index(drop=True).copy()
        df_prob[prob_cols] = proba_all

        per_video_rows = []
        for (subj, vid), g in df_prob.groupby(["subject", "video_id"]):
            true_lab = _majority_label(g["label"])

            # Wahrscheinlichkeitssummen über alle Fenster
            sums = g[prob_cols].sum(axis=0).to_numpy(dtype=float)
            total = float(sums.sum())
            if total > 0.0:
                norm = sums / total
            else:
                norm = np.full_like(sums, fill_value=1.0 / len(sums), dtype=float)

            pred_idx = int(np.argmax(norm))
            pred_lab = str(classes_model[pred_idx])

            per_video_rows.append({
                "subject": int(subj),
                "video_id": str(vid),
                "true_label": str(true_lab),
                "pred_label": pred_lab,
                "n_windows": int(len(g)),
                "prob_accum_normalized": {
                    str(classes_model[i]): float(norm[i]) for i in range(len(classes_model))
                },
            })

        df_video = pd.DataFrame(per_video_rows)
        df_video.to_csv(self.out_dir / "video_level_predictions.csv", index=False)

        if len(df_video) > 0:
            y_true_vid = df_video["true_label"].to_numpy()
            y_pred_vid = df_video["pred_label"].to_numpy()

            acc_vid = accuracy_score(y_true_vid, y_pred_vid)
            bacc_vid = balanced_accuracy_score(y_true_vid, y_pred_vid)
            f1m_vid = f1_score(y_true_vid, y_pred_vid, average="macro")

            rep_vid_txt = classification_report(y_true_vid, y_pred_vid, digits=3, output_dict=False)

            labels_sorted_vid = sorted(np.unique(np.concatenate([y_true_vid, y_pred_vid])))
            cm_vid = confusion_matrix(y_true_vid, y_pred_vid, labels=labels_sorted_vid)

            pd.DataFrame(cm_vid, index=labels_sorted_vid, columns=labels_sorted_vid).to_csv(
                self.out_dir / "cm_video_level.csv", index=True
            )
            with open(self.out_dir / "report_video_level.txt", "w", encoding="utf-8") as f:
                f.write(rep_vid_txt)

            summary_vid = {
                "accuracy": float(acc_vid),
                "balanced_accuracy": float(bacc_vid),
                "f1_macro": float(f1m_vid),
                "labels_sorted": [str(x) for x in labels_sorted_vid],
                "confusion_matrix": cm_vid.astype(int).tolist(),
                "per_video": per_video_rows,
            }
        else:
            self.logger.warning("[VideoEval] No videos found for these subjects.")
            summary_vid = {
                "accuracy": None,
                "balanced_accuracy": None,
                "f1_macro": None,
                "labels_sorted": [],
                "confusion_matrix": [],
                "per_video": [],
            }

        with open(self.out_dir / "metrics_video_level.json", "w", encoding="utf-8") as f:
            json.dump(summary_vid, f, indent=2)

        self.logger.info(
            "[VideoEval] video_acc=%.4f | video_bAcc=%.4f | video_f1_macro=%.4f",
            summary_vid["accuracy"] if summary_vid["accuracy"] is not None else -1,
            summary_vid["balanced_accuracy"] if summary_vid["balanced_accuracy"] is not None else -1,
            summary_vid["f1_macro"] if summary_vid["f1_macro"] is not None else -1
        )
        return summary_vid


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate trained RF pipeline on held-out subjects (window + video level)."
    )
    ap.add_argument("--csv", required=True,
                    help="Pfad zur Feature-CSV (alle Subjects).")
    ap.add_argument("--model", required=True,
                    help="Pfad zum gespeicherten final_model_pipeline.joblib.")
    ap.add_argument("--out_dir", required=True,
                    help="Wohin die Eval-Outputs geschrieben werden sollen.")
    ap.add_argument("--subjects", nargs="+", required=True,
                    help="Liste von Subject-IDs, z.B. --subjects 28 29")
    return ap.parse_args()


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    csv_path = Path(args.csv)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    subjects = [int(s) for s in args.subjects]

    df_full = pd.read_csv(csv_path)

    evaluator = VideoLabelEvaluator(
        model_path=model_path,
        df_full=df_full,
        subjects_to_test=subjects,
        out_dir=out_dir,
    )

    win_metrics = evaluator.evaluate_window_level()
    vid_metrics = evaluator.evaluate_video_level_probs()

    print("\n=== Window-level metrics ===")
    print(json.dumps(win_metrics, indent=2))

    print("\n=== Video-level metrics ===")
    print(json.dumps(vid_metrics, indent=2))
