# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

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
# AdaptiveSMOTE muss VOR joblib.load() definiert sein
# (exakt gleicher Name wie im Training!)
# ---------------------------------------------------------
from imblearn.base import BaseSampler
from sklearn.base import clone
from sklearn.utils._param_validation import Interval
from numbers import Integral


class AdaptiveSMOTE(BaseSampler):
    """
    SMOTE/BorderlineSMOTE/SVMSMOTE mit adaptivem k und sicheren Fallbacks:
    - Wenn im Train-Fold < 2 Klassen vorhanden: no-op.
    - Wenn kleinste Klasse <= 1: no-op.
    - k_neighbors wird an die kleinste Klasse angepasst.
    - Wenn der interne Sampler trotzdem wirft: no-op.
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

        # 1) zu wenig Klassen -> nichts tun
        classes, counts = np.unique(y_arr, return_counts=True)
        if classes.size < 2:
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y

        min_count = int(counts.min())
        # 2) kleinste Klasse hat nur 1 Sample -> nichts tun
        if min_count <= 1:
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y

        # 3) k so wählen, dass es zur kleinsten Klasse passt
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
            logging.getLogger("rf_case_eval_all").warning(
                "AdaptiveSMOTE: fallback to no-op (k=%d, classes=%s, counts=%s): %r",
                k, classes.tolist(), counts.tolist(), e
            )
            self._disabled_ = True
            self._effective_smote_ = None
            return X, y


# ---------------------------------------------------------
# Konstanten & Helper wie im Training
# ---------------------------------------------------------

# Video -> Emotion Mapping (wie in train_rf_from_case_combined.py)
VIDEO_TO_LABEL = {
    1: "amused",
    2: "amused",
    3: "bored",
    4: "bored",
    5: "relaxed",
    6: "relaxed",
    7: "scary",
    8: "scary",
    # Pausen können im combined trotzdem vorkommen
    10: None,
    11: None,
    12: None,
}

META_COLS = {"subject", "start_s", "end_s", "video"}
LABEL_COLS = {"label_valence", "label_arousal"}


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
    logger = logging.getLogger("rf_case_eval_all")
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
# Evaluator-Klasse (wie vorher, leicht angepasst)
# ---------------------------------------------------------

class VideoLabelEvaluatorCase:
    def __init__(
        self,
        model_path: Path,
        df_full: pd.DataFrame,
        subjects_to_test: List[int],
        out_dir: Path,
        classes_to_keep: Optional[List[str]] = None,
        logger: logging.Logger = None
    ):
        """
        model_path:
            Pfad zur gespeicherten Pipeline (final_model_pipeline.joblib),
            wie sie von train_rf_from_case_combined.py erzeugt wird.

        df_full:
            kompletter Datensatz (alle Subjects),
            muss mindestens enthalten:
              - 'subject'
              - 'video'
              - alle Feature-Spalten

        subjects_to_test:
            Liste mit Subject-IDs, die evaluiert werden sollen

        classes_to_keep:
            Optional: Liste von Klassen, auf die gefiltert werden soll,
            z.B. ["scary", "amused"] passend zu einem Experiment.

        out_dir:
            Ausgabe-Ordner für Reports
        """
        self.logger = logger or make_logger()

        self.model_path = Path(model_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.classes_to_keep = classes_to_keep

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
        if "subject" not in df_full.columns:
            raise ValueError("Spalte 'subject' fehlt im DataFrame.")

        if "video" not in df_full.columns:
            raise ValueError("Spalte 'video' fehlt im DataFrame (wird für das Mapping benötigt).")

        df_test = df_full[df_full["subject"].astype(int).isin(subjects_to_test)].copy()
        if len(df_test) == 0:
            self.logger.warning("No rows found for subjects %s", subjects_to_test)

        # Label-Spalte bauen, falls nicht vorhanden
        if "label" not in df_test.columns:
            self.logger.info("Erzeuge 'label' via VIDEO_TO_LABEL Mapping")
            df_test["label"] = df_test["video"].map(VIDEO_TO_LABEL)

        before_len = len(df_test)
        df_test = df_test[df_test["label"].notna()].copy()
        self.logger.info(
            "After video->label mapping & dropping unmapped: %d -> %d rows",
            before_len, len(df_test)
        )

        # Optional: nur bestimmte Klassen wie im Experiment behalten
        if self.classes_to_keep is not None:
            before = len(df_test)
            df_test = df_test[df_test["label"].isin(self.classes_to_keep)].copy()
            self.logger.info(
                "Filter auf Klassen %s: %d -> %d Zeilen",
                self.classes_to_keep, before, len(df_test)
            )

        self.df_test = df_test

        # ----------------- Features bestimmen wie im Training -----------------
        drop_cols = META_COLS | LABEL_COLS | {"label"}
        feat_cols = [c for c in self.df_test.columns if c not in drop_cols]
        if not feat_cols:
            raise ValueError("Keine Feature-Spalten gefunden – prüfe Meta/LABEL_COLS.")
        self.feat_cols = feat_cols
        self.logger.info("Using %d feature columns for eval.", len(self.feat_cols))

        self.X_test = self.df_test[self.feat_cols].to_numpy(dtype=float)
        self.y_test = self.df_test["label"].to_numpy()

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
        Für jedes (subject, video):
        - nimm alle Fenster,
        - sum(mehrere Fenster pro Klasse von predict_proba),
        - normalisiere diese Summe,
        - argmax => Video-Prediction,
        - ground truth = Mehrheitslabel der Fenster.
        """
        if 'video' not in self.df_test.columns:
            raise ValueError("Need 'video' column for video-level evaluation.")

        self.logger.info("[VideoEval] Predicting probabilities ...")
        proba_all = self.pipeline.predict_proba(self.X_test)  # shape: [N_windows, n_classes]
        classes_model = self.model_classes_
        prob_cols = [f"prob_{c}" for c in classes_model]

        df_prob = self.df_test.reset_index(drop=True).copy()
        df_prob[prob_cols] = proba_all

        per_video_rows = []
        for (subj, vid), g in df_prob.groupby(["subject", "video"]):
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
                "video": int(vid),
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
# Helper: Table Loader (csv / gz / parquet)
# ---------------------------------------------------------

def load_any_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    # .parquet lesen
    if suffix == ".parquet":
        return pd.read_parquet(path)

    # .csv oder .gz -> csv
    if suffix in [".csv", ".gz"]:
        return pd.read_csv(path)

    # Fallback: versuch csv
    return pd.read_csv(path, encoding="utf-8", errors="replace")


# ---------------------------------------------------------
# main: alle Modelle durchiterieren
# ---------------------------------------------------------

if __name__ == "__main__":
    logger = make_logger()

    # >>> HIER ggf. anpassen <<<

    # Pfad zu deinem combined-File wie im Training
    base_csv = Path("features_case_60w10s_test/combined.csv.gz")

    # Root, wo die trainierten Modelle liegen (wie in train_rf_from_case_combined.py)
    model_root = Path("outputs_60w10s")

    # Root, wo die Eval-Ergebnisse hinsollen
    eval_root = Path("eval_60w10s")

    # Welche Subjects sollen evaluiert werden?
    SUBJECTS_TO_TEST = [30]  # <- hier deine Test-Subjects eintragen

    # Alle Experimente / Modelle wie im Training
    experiments = [
        ("scary_vs_amused",  ["scary", "amused"]),
        ("bored_vs_relaxed", ["bored", "relaxed"]),
        ("scary_vs_bored",   ["scary", "bored"]),
        ("amused_vs_bored",  ["amused", "bored"]),
    ]

    # Daten einmal laden
    logger.info("Lade Feature-Tabelle aus %s ...", base_csv)
    df_full = load_any_table(base_csv)
    logger.info("Gelesen: %d Zeilen, %d Spalten", len(df_full), df_full.shape[1])

    # Optional: wenn in combined noch keine Pausen gefiltert sind, schmeiß sie raus
    if "video" in df_full.columns:
        before_pause = len(df_full)
        df_full = df_full[~df_full["video"].isin([10, 11, 12])].copy()
        logger.info("Pausen (10/11/12) entfernt: %d -> %d Zeilen", before_pause, len(df_full))

    results_summary = []

    for exp_name, classes in experiments:
        logger.info("\n=== Starte Evaluation für Experiment: %s (%s) ===", exp_name, classes)

        model_path = model_root / exp_name / "final_model_pipeline.joblib"
        out_dir = eval_root / exp_name

        if not model_path.exists():
            logger.error("Modell für Experiment %s nicht gefunden unter %s – überspringe.",
                         exp_name, model_path)
            continue

        evaluator = VideoLabelEvaluatorCase(
            model_path=model_path,
            df_full=df_full,
            subjects_to_test=SUBJECTS_TO_TEST,
            out_dir=out_dir,
            classes_to_keep=classes,
            logger=logger,
        )

        win_metrics = evaluator.evaluate_window_level()
        vid_metrics = evaluator.evaluate_video_level_probs()

        # Kleine Übersicht in einer Liste sammeln
        results_summary.append({
            "experiment": exp_name,
            "classes": classes,
            "window_acc": win_metrics["accuracy"],
            "window_bacc": win_metrics["balanced_accuracy"],
            "window_f1_macro": win_metrics["f1_macro"],
            "video_acc": vid_metrics["accuracy"],
            "video_bacc": vid_metrics["balanced_accuracy"],
            "video_f1_macro": vid_metrics["f1_macro"],
        })

        logger.info("Experiment %s fertig.", exp_name)

    # Gesamtübersicht als CSV
    if results_summary:
        eval_root.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(eval_root / "summary_all_experiments.csv", index=False)
        logger.info("Gesamtübersicht gespeichert nach %s",
                    eval_root / "summary_all_experiments.csv")

    logger.info("Alle Evaluationen abgeschlossen.")
