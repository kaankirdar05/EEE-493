"""Match predicted boxes to ground-truth boxes and score a scene."""
from __future__ import annotations

from dataclasses import dataclass

from ..geometry import Box, iou


@dataclass
class GtResult:
    """What happened to one ground-truth box."""

    snr_db: float | None
    detected: bool
    iou: float          # IoU with its matched prediction (0 if missed)
    fc_err: float       # |centre-frequency error| / true bandwidth (nan if missed)
    bw_err: float       # |bandwidth error| / true bandwidth (nan if missed)
    t_start_err: float  # |start error| / true duration (nan if missed)
    t_stop_err: float   # |stop error| / true duration (nan if missed)
    bandwidth: float    # true bandwidth, cycles/sample
    duration: float     # true duration, samples


@dataclass
class SceneResult:
    gt: list[GtResult]
    n_pred: int
    n_false_alarms: int  # unmatched predictions lying mostly outside every ground-truth box
    n_fragments: int = 0  # unmatched predictions lying mostly inside a ground-truth box (duplicates, pieces)


def match_boxes(preds: list[Box], gts: list[Box], iou_thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU. Returns (pred_idx, gt_idx, iou)."""
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            v = iou(p, g)
            if v >= iou_thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, v))
    return out


def _inside_fraction(p: Box, gts: list[Box]) -> float:
    """Largest share of ``p``'s area that falls inside a single ground-truth box."""
    best = 0.0
    for g in gts:
        dt = min(p.t1, g.t1) - max(p.t0, g.t0)
        df = min(p.f1, g.f1) - max(p.f0, g.f0)
        if dt > 0 and df > 0 and p.area > 0:
            best = max(best, dt * df / p.area)
    return best


def evaluate_scene(preds: list[Box], gts: list[Box], iou_thr: float = 0.5, fragment_frac: float = 0.5) -> SceneResult:
    """Score one scene.  An unmatched prediction is a *fragment* if at least
    ``fragment_frac`` of its area lies inside a ground-truth box (it is a piece of a
    real signal that did not reach the IoU threshold), otherwise a *false alarm*."""
    nan = float("nan")
    matches = {j: (i, v) for i, j, v in match_boxes(preds, gts, iou_thr)}
    matched_preds = {i for i, _ in matches.values()}
    unmatched = [p for i, p in enumerate(preds) if i not in matched_preds]
    n_frag = sum(1 for p in unmatched if _inside_fraction(p, gts) >= fragment_frac)
    results: list[GtResult] = []
    for j, g in enumerate(gts):
        if j in matches:
            i, v = matches[j]
            p = preds[i]
            results.append(
                GtResult(
                    snr_db=g.snr_db,
                    detected=True,
                    iou=v,
                    fc_err=abs(p.center_freq - g.center_freq) / g.bandwidth,
                    bw_err=abs(p.bandwidth - g.bandwidth) / g.bandwidth,
                    t_start_err=abs(p.t0 - g.t0) / g.duration,
                    t_stop_err=abs(p.t1 - g.t1) / g.duration,
                    bandwidth=g.bandwidth,
                    duration=g.duration,
                )
            )
        else:
            results.append(GtResult(g.snr_db, False, 0.0, nan, nan, nan, nan, g.bandwidth, g.duration))
    return SceneResult(gt=results, n_pred=len(preds), n_false_alarms=len(unmatched) - n_frag, n_fragments=n_frag)
