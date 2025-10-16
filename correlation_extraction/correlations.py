# -*- coding: utf-8 -*-
"""
Korrelation zwischen physiologischen Daten und Emotionen (Valence & Arousal)
CASE Dataset — interpolated Daten
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# KONFIGURATION
# ------------------------------------------------------------
SUBJECT = 18  # z.B. Teilnehmer 1
PATH = Path("../case_dataset-master/data/interpolated")

path_phys = PATH / "physiological" / f"sub_{SUBJECT}.csv"
path_ann = PATH / "annotations" / f"sub_{SUBJECT}.csv"

# ------------------------------------------------------------
# DATEN LADEN & SYNCHRONISIEREN
# ------------------------------------------------------------
phys = pd.read_csv(path_phys)
ann = pd.read_csv(path_ann)

# Zeitachsen in Sekunden
phys["time_s"] = phys["daqtime"] / 1000
ann["time_s"] = ann["jstime"] / 1000

# Annotationen auf Physiologie-Zeitachse interpolieren
for col in ["valence", "arousal"]:
    phys[col] = np.interp(phys["time_s"], ann["time_s"], ann[col])

# ------------------------------------------------------------
# KORRELATION PRO SIGNAL
# ------------------------------------------------------------
signals = ["gsr", "bvp", "ecg", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap"]

corrs = []
for sig in signals:
    if sig not in phys.columns:
        continue
    r_val, _ = pearsonr(phys[sig], phys["valence"])
    r_aro, _ = pearsonr(phys[sig], phys["arousal"])
    corrs.append({"signal": sig, "r_valence": r_val, "r_arousal": r_aro})

df_corr = pd.DataFrame(corrs).set_index("signal")
print(df_corr.round(3))

# ------------------------------------------------------------
# OPTIONAL: VISUALISIERUNG
# ------------------------------------------------------------
ax = df_corr.plot(kind="bar", figsize=(8,4))
plt.title(f"Korrelationen – Subjekt {SUBJECT}")
plt.ylabel("Pearson r")
plt.ylim(-1, 1)
plt.axhline(0, color="black", lw=0.8)
plt.tight_layout()
plt.show()
