"""Why is a ground-truth signal not matched?  Run on the cached scenes of run_detection_baseline.py.

For the energy detector and the wide-guard CFAR, every signal that is not matched at IoU 0.3 is put in one bucket:

* not covered        fewer than 30 % of its cells are flagged (detector sensitivity: too weak or too narrow)
* overlaps a neighbour  covered, but another signal overlaps it in time and lies within 0.01 cycles/sample
                     in frequency, so connected components merge them into one box
* isolated, bad box  covered, nothing nearby, but no predicted box reaches IoU 0.3 (box too small or split)

It also compares the two detectors on the same signals (paired).

    python scripts/diagnose_misses.py --n-scenes 300 --seed 1000
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
    args = ap.parse_args()

    geom = SpecGeometry(args.n_samples, 512, 256)
    post = Post()
    specs = {s.name: s for s in DEFAULT_SPECS if s.name in DETECTORS}
    thr = {n: calibrate(s, geom, args.pfa) for n, s in specs.items()}

    for level in args.levels:
        key = f"torchsig_imp{level}_snr{args.snr_db[0]:g}-{args.snr_db[1]:g}_seed{args.seed}"
        buckets = {n: Counter() for n in specs}
        paired = []
        for sc in cached(None, key, args.n_scenes, args.cache_dir):
            P = power_spectrogram(sc.iq, geom)
            gts = sc.boxes
            touch = [touches_other(gts, j) for j in range(len(gts))]
            det = {}
            for n, s in specs.items():
                mask = detect_mask(s, P, thr[n])
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
            paired += [(det[DETECTORS[0]][j], det[DETECTORS[1]][j], touch[j]) for j in range(len(gts))]

        print(f"\n== impairment level {level}: {len(paired)} signals")
        for n, c in buckets.items():
            tot = sum(c.values())
            print(f"  {n:<20}", {k: f"{v / tot:.1%}" for k, v in c.items()})
        e, c, t = (np.array(x, bool) for x in zip(*paired))
        b, cc = int((e & ~c).sum()), int((~e & c).sum())
        z = (b - cc) / math.sqrt(max(b + cc, 1))
        print(f"  paired IoU>=0.3: energy only {b}, CFAR only {cc}, McNemar z = {z:.2f} (|z| < 1.96: no significant difference)")
        print(f"  {t.mean():.0%} of signals overlap a neighbour; recall isolated: energy {e[~t].mean():.2f}, CFAR {c[~t].mean():.2f}; "
              f"overlapping: energy {e[t].mean():.2f}, CFAR {c[t].mean():.2f}")


if __name__ == "__main__":
    main()
