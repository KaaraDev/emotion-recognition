# -*- coding: utf-8 -*-
# CaseRandomForestTest_Classifier.py

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Eigene Module
from models.CaseDataPreprocessor import CaseDataPreprocessor
from models.CaseRandomForestClassifier import CaseRandomForestClassifier, RFConfig

# ------------------------------------------------------------
# 0) Settings
# ------------------------------------------------------------
MODEL_PATH = "C:/Users/metin/OneDrive/Desktop/Informatik/10.Semester/thesis/emotion-recognition/models_v3/outputs/rf_model.joblib"
REMOVE_VIDEOS = {10, 11, 12}       # Blue/Start/End global raus
TEST_SUBJECTS = {30}               # echter Holdout
FEATURES_DIR = "features_case"     # falls kombiniert vorliegt, nutzen wir das
COMBINED_BASENAME = "combined"
PREFER_PARQUET = True

# Falls Preprocessor-Fallback nötig:
PREP_BASE_PATH = ".."              # Pfad zu deinem CASE-Dataset
FS = 20
WINDOW_SIZE = 10
STEP_SIZE = 5                      # <--- am besten wie im Training setzen


# ------------------------------------------------------------
# 1) Modell laden
# ------------------------------------------------------------
clf = CaseRandomForestClassifier.load(MODEL_PATH)
assert clf.model is not None, "Kein Modell im Bundle gefunden."
assert clf.le_ is not None and len(clf.le_.classes_) > 0, "LabelEncoder im Modell-Bundle fehlt."
assert hasattr(clf, "emo_map") and isinstance(clf.emo_map, dict), "emo_map fehlt im Modell-Bundle."

print(f"[INFO] Geladenes Modell: Klassen={list(clf.le_.classes_)}")


# ------------------------------------------------------------
# 2) Testdaten laden (bevorzugt aus combined.*), sonst Preprocessor
# ------------------------------------------------------------
def load_test_from_combined() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt combined-Features und filtert auf TEST_SUBJECTS & REMOVE_VIDEOS."""
    dir_path = os.path.abspath(FEATURES_DIR)
    pq = os.path.join(dir_path, f"{COMBINED_BASENAME}.parquet")
    gz = os.path.join(dir_path, f"{COMBINED_BASENAME}.csv.gz")

    if PREFER_PARQUET and os.path.exists(pq):
        df = pd.read_parquet(pq)
        print(f"[LOAD] Combined Parquet: {pq}  shape={df.shape}")
    elif os.path.exists(gz):
        df = pd.read_csv(gz)
        print(f"[LOAD] Combined CSV.GZ:  {gz}  shape={df.shape}")
    else:
        return None, None

    required_meta = ["subject", "start_s", "end_s", "video"]
    for c in required_meta:
        if c not in df.columns:
            raise ValueError(f"Spalte '{c}' fehlt in combined-Datei.")

    label_cols = ["label_valence", "label_arousal"]
    for c in label_cols:
        if c not in df.columns:
            raise ValueError(f"Spalte '{c}' fehlt in combined-Datei.")

    meta = df[required_meta].copy()
    # Features = alles außer Meta + Labelspalten
    drop_cols = set(required_meta + label_cols)
    X = df.drop(columns=[c for c in df.columns if c in drop_cols]).copy()

    # Filter: Videos global raus
    mask_vid = ~meta["video"].isin(REMOVE_VIDEOS)
    X, meta = X.loc[mask_vid].reset_index(drop=True), meta.loc[mask_vid].reset_index(drop=True)

    # Filter: Subjekt 30
    mask_subj = meta["subject"].isin(TEST_SUBJECTS)
    X_te, meta_te = X.loc[mask_subj].reset_index(drop=True), meta.loc[mask_subj].reset_index(drop=True)

    print(f"[FILTER] Test-Set: Intervalle={len(X_te)} | Subjekte={sorted(meta_te['subject'].unique())} "
          f"| Videos={sorted(meta_te['video'].unique())}")
    return X_te, meta_te


def load_test_with_preprocessor() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erzeugt Testfeatures via Preprocessor (Parameter müssen zum Training passen!)."""
    prep = CaseDataPreprocessor(
        base_path=PREP_BASE_PATH,
        fs=FS,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        subjects=list(TEST_SUBJECTS),
        label_shift_s=0.0,
        use_video_as_feature=False,
    )
    X_te, yv, ya, meta_te = prep.prepare_all()

    # Videos global raus
    mask_vid = ~meta_te["video"].isin(REMOVE_VIDEOS)
    X_te, meta_te = X_te.loc[mask_vid].reset_index(drop=True), meta_te.loc[mask_vid].reset_index(drop=True)

    print(f"[PREP] Test-Set via Preprocessor: Intervalle={len(X_te)} | Subjekte={sorted(meta_te['subject'].unique())} "
          f"| Videos={sorted(meta_te['video'].unique())}")
    return X_te, meta_te


X_te, meta_te = load_test_from_combined()
if X_te is None:
    print("[INFO] Keine combined-Datei gefunden – nutze Preprocessor-Fallback.")
    X_te, meta_te = load_test_with_preprocessor()

assert len(X_te) > 0, "Leeres Test-Set nach Filterung."


# ------------------------------------------------------------
# 3) Ground-Truth-Labels (aus Video→Emotion) & Feature-Ausrichtung
# ------------------------------------------------------------
# Wahre Labels aus den Video-IDs mappen (gleiches Mapping wie im Training im Modell gespeichert)
y_true_str = meta_te["video"].map(clf.emo_map)
if y_true_str.isna().any():
    fehlend = sorted(set(meta_te.loc[y_true_str.isna(), "video"]))
    raise ValueError(f"Für folgende Video-IDs fehlt ein Label in emo_map: {fehlend}")

# In die gleiche Klassenordnung encoden wie das geladene Modell
y_true_int = clf.le_.transform(y_true_str.values.astype(str))

# Feature-Spalten passend zum trainierten Modell ausrichten
if clf.X_ is not None and isinstance(clf.X_, pd.DataFrame) and len(clf.X_.columns) > 0:
    feat_names = list(clf.X_.columns)
    X_te = X_te.reindex(columns=feat_names, fill_value=0.0)
else:
    print("[WARN] Feature-Namen im Modellbundle fehlen – nutze aktuelle Spalten von X_te.")


# ------------------------------------------------------------
# 4) Vorhersage
# ------------------------------------------------------------
y_pred_int = clf.predict(X_te)
y_proba = clf.predict_proba(X_te)

# ------------------------------------------------------------
# 5) Intervall-Level Metriken
# ------------------------------------------------------------
acc = accuracy_score(y_true_int, y_pred_int)
bacc = balanced_accuracy_score(y_true_int, y_pred_int)
f1m = f1_score(y_true_int, y_pred_int, average="macro")

print("\n[INTERVAL] Scores:")
print(f"  accuracy={acc:.4f}  balanced_acc={bacc:.4f}  f1_macro={f1m:.4f}")

print("\n[INTERVAL] classification_report:")
print(classification_report(y_true_int, y_pred_int, target_names=list(clf.le_.classes_)))

print("[INTERVAL] confusion_matrix:")
print(confusion_matrix(y_true_int, y_pred_int))

# ------------------------------------------------------------
# 6) Video-Level Aggregation (Summe der Klassenwahrscheinlichkeiten → argmax)
# ------------------------------------------------------------
tmp = pd.DataFrame(y_proba)
tmp["subject"] = meta_te["subject"].values
tmp["video"] = meta_te["video"].values
# Summe der Wahrscheinlichkeiten pro (subject, video)
proba_sum = tmp.groupby(["subject", "video"]).sum(numeric_only=True)

# Vorhersage auf Video-Level
y_pred_video = proba_sum.values.argmax(axis=1)

# True-Label auf Video-Level = Modus der Intervall-Labels je (subject,video)
df_true = pd.DataFrame({
    "subject": meta_te["subject"].values,
    "video": meta_te["video"].values,
    "y_true_int": y_true_int
})
y_true_video = (
    df_true.groupby(["subject", "video"])["y_true_int"]
           .agg(lambda x: np.bincount(x).argmax())
           .values
)

acc_v = accuracy_score(y_true_video, y_pred_video)
f1m_v = f1_score(y_true_video, y_pred_video, average="macro")

print("\n[VIDEO] Scores:")
print(f"  accuracy={acc_v:.4f}  f1_macro={f1m_v:.4f}")

# Optional: Tabellarische Ansicht je (subject, video)
video_view = (
    df_true.assign(y_pred_int=y_pred_int)
           .groupby(["subject","video"])
           .agg(true_label=("y_true_int", lambda x: clf.le_.classes_[np.bincount(x).argmax()]),
                pred_label=("y_pred_int", lambda x: clf.le_.classes_[np.bincount(x).argmax()]),
                n_windows=("y_true_int", "size"))
           .reset_index()
           .sort_values(["subject","video"])
)
print("\n[VIDEO] Übersicht pro (subject, video):")
print(video_view.to_string(index=False))

# ------------------------------------------------------------
# 7) (Optional) CSV mit einem Ausschnitt speichern
# ------------------------------------------------------------
SAVE_CSV = True
if SAVE_CSV:
    n = min(10000, len(X_te))
    df_out = pd.DataFrame({
        "subject": meta_te["subject"].values[:n],
        "video": meta_te["video"].values[:n],
        "true_label": y_true_str.values[:n],
        "pred_label": [clf.le_.classes_[i] for i in y_pred_int[:n]],
    })
    out_path = "rf_classifier_predictions_sample.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[INFO] Beispiel-Predictions gespeichert unter: {out_path}")
