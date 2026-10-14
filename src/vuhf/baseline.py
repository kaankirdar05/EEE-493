"""Detector configurations and scene scoring shared by the baseline scripts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import (
    ca_cfar_2d,
    calibrate_multiplier,
    cfar_statistic,
    energy_detector,
    energy_statistic,
    mask_to_boxes,
)
from .detectors.boxes import boxes_to_mask
from .eval.matching import evaluate_scene
from .geometry import Box, SpecGeometry


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    kind: str                                   # "energy" or "cfar"
    avg: tuple[int, int] = (1, 3)               # cells averaged before thresholding (frequency, time)
    floor: str = "global"                       # energy: "global" or "per_bin" noise floor
    q: float = 0.5                              # energy: quantile used for the noise floor (0.5 = median)
    margin_db: float | None = None              # energy: fixed margin above the floor instead of a Pfa
    train: tuple[int, int] = (24, 0)            # cfar: training half-size (frequency, time)
    guard: tuple[int, int] = (8, 0)             # cfar: guard half-size (frequency, time)


@dataclass(frozen=True)
class Post:
    """Mask -> box post-processing."""

    link: tuple[int, int] = (4, 16)   # chosen on a validation set, see scripts/tune_postprocessing.py
    min_cells: int = 40


# Averaging window (3 frequency x 9 time cells) was chosen on a separate validation set
# (scripts/tune_postprocessing.py): it raises the share of a signal's cells that are
# flagged from ~0.7 to ~0.9 without changing the false-alarm budget.
DEFAULT_SPECS = [
    DetectorSpec("energy_median", "energy", avg=(3, 9)),
    DetectorSpec("energy_q25", "energy", avg=(3, 9), q=0.25),
    DetectorSpec("energy_per_bin", "energy", avg=(3, 9), floor="per_bin"),
    DetectorSpec("energy_median+6dB", "energy", avg=(1, 3), margin_db=6.0),  # rule of thumb from the project plan
    DetectorSpec("cfar_freq_g8_t24", "cfar", avg=(3, 9), train=(24, 0), guard=(8, 0)),
    DetectorSpec("cfar_freq_g24_t48", "cfar", avg=(3, 9), train=(48, 0), guard=(24, 0)),
    DetectorSpec("cfar_2d_g2x1_t8x4", "cfar", avg=(1, 1), train=(8, 4), guard=(2, 1)),
]


def calibrate(spec: DetectorSpec, geom: SpecGeometry, pfa: float, seed: int = 0) -> float | None:
    """Threshold multiplier (energy) or CFAR scale factor holding ``pfa`` on pure noise.
    ``None`` for the fixed-margin energy detector."""
    if spec.kind == "energy":
        if spec.margin_db is not None:
            return None
        return calibrate_multiplier(lambda P: energy_statistic(P, spec.avg, spec.floor, spec.q), geom, pfa, seed=seed)
    return calibrate_multiplier(lambda P: cfar_statistic(P, spec.train, spec.guard, spec.avg), geom, pfa, seed=seed)


def detect_mask(spec: DetectorSpec, P: np.ndarray, threshold: float | None) -> np.ndarray:
    if spec.kind == "energy":
        return energy_detector(P, avg=spec.avg, floor=spec.floor, q=spec.q, margin_db=spec.margin_db, multiplier=threshold)[0]
    return ca_cfar_2d(P, train=spec.train, guard=spec.guard, avg=spec.avg, alpha=threshold)[0]


def score_mask(boxes_gt: list[Box], mask: np.ndarray, geom: SpecGeometry, post: Post,
               iou_thrs=(0.5, 0.3), cover_thr: float = 0.3) -> dict:
    """Score one detector mask against the ground truth of one scene.

    Returns per-ground-truth rows and per-scene counts.  A ground-truth signal is

    * ``det@t``   detected if a predicted box matches it with IoU >= t
                  (depends on the box-forming post-processing),
    * ``covered`` covered if at least ``cover_thr`` of its cells are flagged in the
                  raw mask (measures the detector's sensitivity only).
    """
    preds = mask_to_boxes(mask, geom, link=post.link, min_cells=post.min_cells)
    rows = [dict(label=g.label, snr_db=g.snr_db, bandwidth=g.bandwidth, duration=g.duration) for g in boxes_gt]
    for j, g in enumerate(boxes_gt):
        k0, k1, j0, j1 = geom.box_to_cells(g)
        rows[j]["covered"] = bool(mask[k0 : k1 + 1, j0 : j1 + 1].mean() >= cover_thr)
    n_fa, n_frag = {}, {}
    for t in iou_thrs:
        res = evaluate_scene(preds, boxes_gt, t)
        n_fa[t], n_frag[t] = res.n_false_alarms, res.n_fragments
        for j, g in enumerate(res.gt):
            rows[j][f"det@{t}"] = g.detected
            if t == iou_thrs[0]:
                rows[j].update(iou=g.iou, fc_err=g.fc_err, bw_err=g.bw_err, t_start_err=g.t_start_err, t_stop_err=g.t_stop_err)
    far = ~boxes_to_mask(boxes_gt, geom, margin=(4, 4))
    return dict(rows=rows, n_fa=n_fa, n_frag=n_frag, n_pred=len(preds), preds=preds,
                cell_fa=float(mask[far].mean()) if far.any() else float("nan"),
                cell_pd=float(np.mean([r["covered"] for r in rows])) if rows else float("nan"))
