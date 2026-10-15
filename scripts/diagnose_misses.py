"""Why is a ground-truth signal not matched?  Run on the cached scenes of run_detection_baseline.py.

For the energy detector and the wide-guard CFAR, every signal that is not matched at IoU 0.3 is put in one bucket:

* not covered        fewer than 30 % of its cells are flagged (detector sensitivity: too weak or too narrow)
* overlaps a neighbour  covered, but another signal overlaps it in time and lies within 0.01 cycles/sample
                     in frequency, so connected components merge them into one box
* isolated, bad box  covered, nothing nearby, but no predicted box reaches IoU 0.3 (box too small or split)

It also compares the detectors on the same signals (paired, energy against each other detector), and with
``--yolo-weights`` adds the trained YOLO model.

    python scripts/diagnose_misses.py --n-scenes 300 --seed 1000 --yolo-weights data_cache/yolo_runs/signal/weights/best.pt --yolo-conf 0.05
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask
from vuhf.data.sources import cached
from vuhf.detectors import mask_to_boxes
from vuhf.detectors.boxes import boxes_to_mask
from vuhf.eval.matching import match_boxes
from vuhf.geometry import SpecGeometry
from vuhf.spectrogram import power_spectrogram

DETECTORS = ("energy_median", "cfar_freq_g24_t48")


def touches_other(boxes, j, freq_gap=0.01):
    g = boxes[j]
    return any(
        k != j and min(g.t1, h.t1) - max(g.t0, h.t0) > 0 and min(g.f1, h.f1) - max(g.f0, h.f0) > -freq_gap
        for k, h in enumerate(boxes)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scenes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--snr-db", type=float, nargs=2, default=[0.0, 20.0])
    ap.add_argument("--n-samples", type=int, default=262144)
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    ap.add_argument("--yolo-weights", type=Path, default=None)
    ap.add_argument("--yolo-conf", type=float, default=0.05)
    args = ap.parse_args()

    geom = SpecGeometry(args.n_samples, 512, 256)
    post = Post()
    specs = {s.name: s for s in DEFAULT_SPECS if s.name in DETECTORS}
    thr = {n: calibrate(s, geom, args.pfa) for n, s in specs.items()}
    names = list(specs)
    yolo = None
    if args.yolo_weights is not None:
        from vuhf.detectors.yolo import YoloDetector

        yolo = YoloDetector(args.yolo_weights, geom, min_conf=args.yolo_conf)
        names.append("yolo")

    for level in args.levels:
        key = f"torchsig_imp{level}_snr{args.snr_db[0]:g}-{args.snr_db[1]:g}_seed{args.seed}"
        buckets = {n: Counter() for n in names}
        paired = []
        for sc in cached(None, key, args.n_scenes, args.cache_dir):
            P = power_spectrogram(sc.iq, geom)
            gts = sc.boxes
            touch = [touches_other(gts, j) for j in range(len(gts))]
            det = {}
            for n in names:
                if n == "yolo":
                    preds = yolo.predict(P, args.yolo_conf)
                    mask = boxes_to_mask(preds, geom)
                else:
                    mask = detect_mask(specs[n], P, thr[n])
                    preds = mask_to_boxes(mask, geom, link=post.link, min_cells=post.min_cells)
                matched = {j for _, j, _ in match_boxes(preds, gts, 0.3)}
                det[n] = [j in matched for j in range(len(gts))]
                for j, g in enumerate(gts):
                    k0, k1, j0, j1 = geom.box_to_cells(g)
                    covered = mask[k0 : k1 + 1, j0 : j1 + 1].mean() >= 0.3
                    if j in matched:
                        buckets[n]["matched"] += 1
                    elif not covered:
                        buckets[n]["not covered"] += 1
                    elif touch[j]:
                        buckets[n]["overlaps a neighbour"] += 1
                    else:
                        buckets[n]["isolated, bad box"] += 1
            paired += [tuple(det[n][j] for n in names) + (touch[j],) for j in range(len(gts))]

        print(f"\n== impairment level {level}: {len(paired)} signals")
        for n, c in buckets.items():
            tot = sum(c.values())
            print(f"  {n:<20}", {k: f"{v / tot:.1%}" for k, v in c.items()})
        cols = np.array(paired, bool)
        t = cols[:, -1]
        ids = {n: cols[:, k] for k, n in enumerate(names)}
        e = ids["energy_median"]
        print(f"  {t.mean():.0%} of signals overlap a neighbour. recall@0.3 isolated / overlapping:")
        for n in names:
            print(f"    {n:<20}{ids[n][~t].mean():.2f} / {ids[n][t].mean():.2f}")
        for n in names[1:]:
            b, cc = int((e & ~ids[n]).sum()), int((~e & ids[n]).sum())
            z = (b - cc) / math.sqrt(max(b + cc, 1))
            print(f"  paired IoU>=0.3, energy vs {n}: energy only {b}, {n} only {cc}, McNemar z = {z:.2f} ({'significant' if abs(z) >= 1.96 else 'not significant'} at 95 %)")


if __name__ == "__main__":
    main()
