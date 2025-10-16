# -*- coding: utf-8 -*-
"""
Valence & Arousal Timeline (Subject X) — feste Video-Reihenfolge
Spalten: jstime, valence, arousal, video
Hinweis: Videos 1..8 werden in fixer Reihenfolge (1→8) aneinandergehängt.
Die Zeitachse ist dadurch nicht mehr stetig.
"""

import re
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys
from matplotlib import colors as mcolors

# ---------------------------------------
# Pfad zur CSV (exact 1 Subject)
# ---------------------------------------
csv_path = "../case_dataset-master/data/interpolated/annotations/sub_1.csv"

# ---------------------------------------
# Daten laden
# ---------------------------------------
df = pd.read_csv(csv_path)

# ---------------------------------------
# Feste Reihenfolge der Videos (nur Emotionen)
# ---------------------------------------
fixed_order = [1, 2, 3, 4, 5, 6, 7, 8]  # Amusement1, Amusement2, ..., Scary2

# Nur diese Videos verwenden
df = df[df["video"].isin(fixed_order)].copy()

# ---------------------------------------
# Video-ID → Name (Mapping)
# ---------------------------------------
video_map = {
    1: "Amusement 1",
    2: "Amusement 2",
    3: "Boredom 1",
    4: "Boredom 2",
    5: "Relaxation 1",
    6: "Relaxation 2",
    7: "Scary 1",
    8: "Scary 2",
    # 10/11/12 werden hier absichtlich ignoriert
}

# ---------------------------------------
# Hilfsfunktionen für Kategorien & Farbabstufungen
# ---------------------------------------
EMO_CATS = {"Amusement", "Boredom", "Relaxation", "Scary"}


def get_category(label: str) -> str:
    head = label.split()[0]
    return head if head in EMO_CATS else label


def get_variant_index(label: str) -> int:
    m = re.search(r"\b(\d+)\b", label)
    return int(m.group(1)) if m else 1


def adjust_lightness(color, factor=1.0):
    """Farbhelligkeit anpassen: factor <1 = dunkler, >1 = heller"""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


# Grundfarben pro (Emotions-)Kategorie
base_colors = {
    "Amusement": "orange",
    "Boredom": "violet",
    "Relaxation": "green",
    "Scary": "red",
}

# Helligkeit je Variante (1 = dunkler, 2 = heller)
variant_lightness = {
    1: 0.8,
    2: 1.2,
}

# ---------------------------------------
# Zeit neu aufbauen: Blöcke nach fixer Reihenfolge aneinanderhängen
# ---------------------------------------
segments = []
spans = []  # für axvspan: (start_min, end_min, label, color)

t_offset_min = 0.0  # kumulative Minuten auf neuer Achse

for vid in fixed_order:
    seg = df[df["video"] == vid]
    if seg.empty:
        continue

    label = video_map.get(vid, f"Video {vid}")
    category = get_category(label)
    var_idx = get_variant_index(label)
    base = base_colors.get(category, "tab:purple")
    shade = adjust_lightness(base, variant_lightness.get(var_idx, 1.0))

    # relative Segmentzeit in Minuten (beginnend bei 0 pro Block)
    t0 = seg["jstime"].iloc[0]
    rel_time_min = (seg["jstime"] - t0) / 1000.0 / 60.0

    # in die neue Achse einsortieren (an t_offset anhängen)
    new_time_min = rel_time_min + t_offset_min

    # Segment speichern
    segments.append(pd.DataFrame({
        "time_min_fixed": new_time_min.values,
        "valence": seg["valence"].values,
        "arousal": seg["arousal"].values,
        "video": vid,
        "label": label,
    }))

    # Spannbreite für Hintergrund
    start = float(new_time_min.iloc[0])
    end = float(new_time_min.iloc[-1])
    spans.append((start, end, label, shade))

    # Offset für nächstes Video erhöhen
    t_offset_min = end  # nahtlos anhängen (egal ob „Sprung“ dazwischen)

# Neu zusammengesetztes DataFrame
if len(segments) == 0:
    raise RuntimeError("Keine Segmente für die feste Reihenfolge gefunden.")
df_fixed = pd.concat(segments, ignore_index=True)

# ---------------------------------------
# Plot anlegen
# ---------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

# Valence & Arousal Linien (auf neuer Zeitachse)
line_valence, = ax.plot(df_fixed["time_min_fixed"], df_fixed["valence"], label="Valence", alpha=0.9)
line_arousal, = ax.plot(df_fixed["time_min_fixed"], df_fixed["arousal"], label="Arousal", alpha=0.9)

# Farbige Hintergründe pro (re-sequenziertem) Video-Block
for start, end, label, shade in spans:
    ax.axvspan(start, end, color=shade, alpha=0.18)

# Achsen, Titel, Limits
ax.set_title("Valence & Arousal Timeline (feste Video-Reihenfolge, Subject 1)")
ax.set_xlabel("Time (minutes, resequenced)")
ax.set_ylabel("Rating [0.5 – 9.5]")
ax.set_ylim(0.5, 9.5)

# ---------------------------------------
# Legenden
# ---------------------------------------
# (1) Video-Legende in fixer Reihenfolge, aber nur wenn vorhanden
video_patches = []
seen_labels = set()
present_vids = [v for v in fixed_order if (df_fixed["video"] == v).any()]
for vid in present_vids:
    label = video_map.get(vid, f"Video {vid}")
    if label in seen_labels:
        continue
    seen_labels.add(label)

    category = get_category(label)
    var_idx = get_variant_index(label)
    base = base_colors.get(category, "tab:purple")
    shade = adjust_lightness(base, variant_lightness.get(var_idx, 1.0))
    video_patches.append(mpatches.Patch(color=shade, alpha=0.35, label=label))

# (2) Linien-Legende (Valence/Arousal)
line_handles, line_labels = ax.get_legend_handles_labels()
hl_ordered = OrderedDict(zip(line_labels, line_handles))  # Duplikate entfernen
line_handles = list(hl_ordered.values())
line_labels = list(hl_ordered.keys())

# Kombinieren (Videos zuerst, dann Linien)
handles = video_patches + line_handles
labels = [p.get_label() for p in video_patches] + line_labels
ax.legend(handles=handles, labels=labels, loc="upper right")

plt.tight_layout()
plt.show()
