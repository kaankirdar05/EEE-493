"""Aggregate scene results into detection curves."""
from __future__ import annotations

import numpy as np

from .matching import SceneResult


def pd_vs_snr(results: list[SceneResult], bins: np.ndarray) -> dict:
    """Probability of detection per SNR bin (bin edges ``bins``, dB)."""
    snr = np.array([g.snr_db for r in results for g in r.gt if g.snr_db is not None], dtype=float)
    det = np.array([g.detected for r in results for g in r.gt if g.snr_db is not None], dtype=bool)
    idx = np.digitize(snr, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    pd = np.full(len(centers), np.nan)
    count = np.zeros(len(centers), dtype=int)
    for k in range(len(centers)):
        sel = idx == k
        count[k] = int(sel.sum())
        if count[k]:
            pd[k] = float(det[sel].mean())
    return {"snr_center_db": centers, "pd": pd, "n_signals": count}


def summarize(results: list[SceneResult]) -> dict:
    """Overall numbers: recall, false alarms per scene, localisation errors of matched boxes."""
    gts = [g for r in results for g in r.gt]
    det = [g for g in gts if g.detected]

    def med(values):
        arr = np.array(values, dtype=float)
        return float(np.median(arr)) if arr.size else float("nan")

    return {
        "n_scenes": len(results),
        "n_signals": len(gts),
        "recall": (len(det) / len(gts)) if gts else float("nan"),
        "false_alarms_per_scene": float(np.mean([r.n_false_alarms for r in results])) if results else float("nan"),
        "fragments_per_scene": float(np.mean([r.n_fragments for r in results])) if results else float("nan"),
        "median_iou": med([g.iou for g in det]),
        "median_fc_err_over_bw": med([g.fc_err for g in det]),
        "median_bw_err_rel": med([g.bw_err for g in det]),
        "median_t_start_err_rel": med([g.t_start_err for g in det]),
        "median_t_stop_err_rel": med([g.t_stop_err for g in det]),
    }
