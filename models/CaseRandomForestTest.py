# -*- coding: utf-8 -*-
# CaseRandomForestTest.py

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# Eigene Module
from models.CaseDataPreprocessor import CaseDataPreprocessor
from models.CaseRandomForest import CaseRandomForest

# ------------------------------------------------------------
# 0) RFConfig-SHIM (nur nötig, wenn altes Modell RFConfig als Objekt gepickelt hat)
#    → harmless: wenn die gespeicherte Datei bereits ein dict benutzt, wird der Shim ignoriert.
# ------------------------------------------------------------
try:
    from models.CaseRandomForest import RFConfig  # falls im Modul vorhanden
except Exception:
    from dataclasses import dataclass
    from typing import Optional


    @dataclass
    class RFConfig:
        n_estimators: int = 300
        max_depth: Optional[int] = None
        min_samples_leaf: int = 3
        n_jobs: int = -1
        random_state: int = 42

# ------------------------------------------------------------
# 1) Modell laden
# ------------------------------------------------------------
MODEL_PATH = "rf_va_model.joblib"
rfw = CaseRandomForest.load(MODEL_PATH)  # setzt u.a. Feature-Namen, falls im Bundle

# ------------------------------------------------------------
# 2) Testdaten vorbereiten – identische Preprocessing-Parameter wie beim Training!
#    → Passe subjects / fs / window_size / step_size / label_shift_s ggf. an.
# ------------------------------------------------------------
prep_test = CaseDataPreprocessor(
    base_path="..",
    fs=20,
    window_size=5,
    step_size=2,
    subjects=list(range(29, 31)),  # Hold-out: 25–30
    label_shift_s=0.0,
    use_video_as_feature=False,
)
X_te, yv, ya, meta_te = prep_test.prepare_all()
y_te = np.column_stack([yv.values, ya.values]).astype(np.float32)

# ------------------------------------------------------------
# 3) Feature-Spalten an das trainierte Modell anpassen (Reihenfolge/Fehlende)
# ------------------------------------------------------------
if rfw.X_ is not None:
    feat_names = list(rfw.X_.columns)
    X_te = X_te.reindex(columns=feat_names, fill_value=0.0)
else:
    # Fallback: Falls Feature-Namen im Bundle fehlen.
    # (Kommt vor, wenn sehr altes Saveformat genutzt wurde.)
    print("[WARN] Keine Feature-Namen im geladenen Modell gefunden – nutze aktuelle X_te-Spaltenreihenfolge.")

# ------------------------------------------------------------
# 4) Vorhersage
# ------------------------------------------------------------
y_hat = rfw.predict(X_te)


# ------------------------------------------------------------
# 5) Metriken gesamt
# ------------------------------------------------------------
def pearson_multi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    out = []
    for j in range(y_true.shape[1]):
        t, p = y_true[:, j], y_pred[:, j]
        out.append(0.0 if np.std(t) == 0 or np.std(p) == 0 else np.corrcoef(t, p)[0, 1])
    return float(np.mean(out))


print("TEST R2      :", r2_score(y_te, y_hat, multioutput="uniform_average"))
print("TEST MAE     :", mean_absolute_error(y_te, y_hat, multioutput="uniform_average"))
print("TEST Pearson :", pearson_multi(y_te, y_hat))


# ------------------------------------------------------------
# 6) Metriken pro Dimension (Valence / Arousal)
# ------------------------------------------------------------
def pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    return 0.0 if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])


for j, name in enumerate(["Valence", "Arousal"]):
    r2 = r2_score(y_te[:, j], y_hat[:, j])
    mae = mean_absolute_error(y_te[:, j], y_hat[:, j])
    r = pearson_1d(y_te[:, j], y_hat[:, j])
    print(f"{name:8s}  R2={r2:.4f}  MAE={mae:.4f}  r={r:.4f}")

# ------------------------------------------------------------
# 7) Optional: pro Subjekt zusammenfassen
# ------------------------------------------------------------
summ_rows = []
subs = meta_te["subject"].values
for s in np.unique(subs):
    idx = subs == s
    r2 = r2_score(y_te[idx], y_hat[idx], multioutput="uniform_average")
    mae = mean_absolute_error(y_te[idx], y_hat[idx], multioutput="uniform_average")
    summ_rows.append({"subject": int(s), "R2": r2, "MAE": mae})
df_subj = pd.DataFrame(summ_rows).sort_values("R2", ascending=False)
print("\n[Per-Subject Summary] (Top 10 by R2)")
print(df_subj.head(10).to_string(index=False))

# ------------------------------------------------------------
# 8) Optional: Ergebnisse speichern (kleiner Sample, um Speicher zu schonen)
# ------------------------------------------------------------
SAVE_CSV = True
if SAVE_CSV:
    n = min(10000, len(X_te))  # nur ein Ausschnitt
    df_out = pd.DataFrame({
        "subject": meta_te["subject"].values[:n],
        "video": meta_te["video"].values[:n] if "video" in meta_te.columns else -1,
        "val_true": y_te[:n, 0],
        "aro_true": y_te[:n, 1],
        "val_pred": y_hat[:n, 0],
        "aro_pred": y_hat[:n, 1],
    })
    out_path = "rf_predictions_sample.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[INFO] Beispiel-Predictions gespeichert unter: {out_path}")
