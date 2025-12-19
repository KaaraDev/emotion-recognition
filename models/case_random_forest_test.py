# plot_rf_scary_vs_boring_from_cv.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------

@dataclass
class PlotCfg:
    cv_pred_path: Path      # Pfad zu cv_predictions.csv
    out_dir: Path           # Ordner für Plots

    exp_pos: str = "scary"  # positive Klasse
    exp_neg: str = "bored"  # negative Klasse


# ----------------------------------------------------------
# Laden der CV-Predictions
# ----------------------------------------------------------

def load_cv_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"cv_predictions.csv nicht gefunden unter {path}")
    df = pd.read_csv(path)
    required_cols = {"fold", "subject", "video", "start_s", "end_s",
                     "true_label", "pred_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in {path}: {missing}")
    return df


# ----------------------------------------------------------
# Plotten für mehrere Subjects
# ----------------------------------------------------------

def plot_subjects(cfg: PlotCfg, subjects: List[int]) -> None:
    print(f"Lade OOF-Predictions aus {cfg.cv_pred_path} ...")
    df = load_cv_predictions(cfg.cv_pred_path)

    # Nur die relevanten Klassen (Exp) behalten, falls andere drin sind
    mask_classes = df["true_label"].isin([cfg.exp_pos, cfg.exp_neg])
    df = df[mask_classes].copy()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subjects:
        print(f"\n=== Subject {subject_id} ===")
        df_subj = df[df["subject"] == subject_id].copy()
        if df_subj.empty:
            print("Keine Fenster für dieses Subject, überspringe.")
            continue

        # chronologisch sortieren
        df_subj = df_subj.sort_values("end_s").reset_index(drop=True)

        # Arrays bauen
        vids = df_subj["video"].to_numpy(dtype=int)
        t_start = df_subj["start_s"].to_numpy(float)
        t_end = df_subj["end_s"].to_numpy(float)
        y_true_str = df_subj["true_label"].to_numpy(str)
        y_pred_str = df_subj["pred_label"].to_numpy(str)

        # 0/1 Encoding (neg=0, pos=1)
        ys = (y_true_str == cfg.exp_pos).astype(int)
        preds = (y_pred_str == cfg.exp_pos).astype(int)

        # --- Video-Infos (Start, Ende, Dauer) ---
        video_info: Dict[int, Tuple[float, float, float]] = {}
        for v in np.unique(vids):
            d_vid = df_subj[df_subj["video"] == v]
            v_start = float(d_vid["start_s"].min())
            v_end = float(d_vid["end_s"].max())
            v_dur = v_end - v_start
            if v_dur <= 0:
                # Sicherheit: falls numerische Probleme
                v_dur = max(1e-6, v_dur)
            video_info[v] = (v_start, v_end, v_dur)

        # Videos in der Reihenfolge ihres Beginns
        videos_in_order = sorted(video_info.keys(),
                                 key=lambda v: video_info[v][0])

        # Kumulative Offsets über alle Videos
        offset_s: Dict[int, float] = {}
        video_segments_cum: List[Tuple[int, float, float]] = []  # (vid, start_s, end_s)
        cum = 0.0
        for v in videos_in_order:
            v_start, v_end, v_dur = video_info[v]
            offset_s[v] = cum
            video_segments_cum.append((v, cum, cum + v_dur))
            cum += v_dur

        # Zeitachse für jedes Fenster: Ende des Fensters relativ zur Videolänge
        x_time_s = np.zeros_like(t_end, dtype=float)
        for i in range(len(t_end)):
            v = vids[i]
            v_start, _, _ = video_info[v]
            rel = t_end[i] - v_start
            x_time_s[i] = offset_s[v] + rel

        # --- Aggregierte "Probabilities" (auf Basis der Vorhersage-Klasse) ---
        agg_p_neg = np.zeros(len(ys), dtype=float)
        agg_p_pos = np.zeros(len(ys), dtype=float)

        count_total: Dict[int, int] = {}
        count_neg: Dict[int, int] = {}
        count_pos: Dict[int, int] = {}

        for i, v in enumerate(vids):
            if v not in count_total:
                count_total[v] = 0
                count_neg[v] = 0
                count_pos[v] = 0

            count_total[v] += 1
            if preds[i] == 1:
                count_pos[v] += 1
            else:
                count_neg[v] += 1

            agg_p_neg[i] = count_neg[v] / count_total[v]
            agg_p_pos[i] = count_pos[v] / count_total[v]

        # --- Plot bauen ---
        fig_path = cfg.out_dir / f"subject{subject_id}_rf_{cfg.exp_pos}_vs_{cfg.exp_neg}.png"
        plot_single_subject(
            subject_id=subject_id,
            exp_pos=cfg.exp_pos,
            exp_neg=cfg.exp_neg,
            vids=vids,
            x_time_s=x_time_s,
            ys=ys,
            preds=preds,
            video_segments_cum=video_segments_cum,
            agg_p_neg=agg_p_neg,
            agg_p_pos=agg_p_pos,
            fig_path=fig_path,
        )
        print(f"Gespeichert: {fig_path}")


# ----------------------------------------------------------
# Einzelnen Subject-Plot zeichnen (wie beim CNN-Plot)
# ----------------------------------------------------------

def plot_single_subject(subject_id: int,
                        exp_pos: str,
                        exp_neg: str,
                        vids: np.ndarray,
                        x_time_s: np.ndarray,
                        ys: np.ndarray,
                        preds: np.ndarray,
                        video_segments_cum: List[Tuple[int, float, float]],
                        agg_p_neg: np.ndarray,
                        agg_p_pos: np.ndarray,
                        fig_path: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [2, 2]},
    )

    # TOP PANEL: True vs Pred
    ax1.scatter(x_time_s, ys, marker="o", label="True", alpha=0.7)
    ax1.scatter(x_time_s, preds, marker="x", label="Pred", alpha=0.7)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels([exp_neg, exp_pos])
    ax1.set_ylabel("Class")
    ax1.set_title(f"Subject {subject_id} – {exp_pos} vs. {exp_neg} (RF, OOF)")

    for vid, s, e in video_segments_cum:
        ax1.axvline(e, linestyle="--", alpha=0.3)
        mid = (s + e) / 2
        ax1.text(mid, 1.15, f"Video {vid}", ha="center", va="bottom")

    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    # BOTTOM PANEL: aggregierte "Probabilities"
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Aggregated probability")
    ax2.set_xlabel("Time (s)")

    vids_arr = np.array(vids)
    first = True
    for vid, start_s, end_s in video_segments_cum:
        m = (vids_arr == vid)
        if not np.any(m):
            continue

        xs = np.concatenate([[start_s], x_time_s[m]])
        pb = np.concatenate([[0.5], agg_p_neg[m]])
        ps = np.concatenate([[0.5], agg_p_pos[m]])

        ax2.plot(xs, pb, marker=".", color="tab:blue",
                 label="P(boring)" if first else None)
        ax2.plot(xs, ps, marker=".", color="tab:orange",
                 label="P(scary)" if first else None)
        first = False

    for _, _, e in video_segments_cum:
        ax2.axvline(e, linestyle="--", alpha=0.3)

    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    cfg = PlotCfg(
        cv_pred_path=Path("outputs_20w10s_plain/scary_vs_bored/cv_predictions.csv"),
        out_dir=Path("figs_rf_scary_vs_bored_20w10s"),
        exp_pos="scary",
        exp_neg="bored",
    )

    # Beispiel-Subjects (anpassen wie du willst)
    subjects = [11,18,19,20,22,24]
    plot_subjects(cfg, subjects)


if __name__ == "__main__":
    main()
