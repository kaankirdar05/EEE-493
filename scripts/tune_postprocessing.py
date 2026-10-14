"""Pick averaging window and box post-processing on a *validation* set of scenes.

Run this on scenes generated with a different seed from the ones used for the final
evaluation, then pass the chosen settings to ``run_detection_baseline.py``.

Selection rule: highest recall at IoU >= 0.3 among the settings with at most
``--max-fa`` false alarms per scene (fragments are not counted as false alarms).

    python scripts/tune_postprocessing.py --n-scenes 24 --seed 7 --impairment-level 1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from vuhf.baseline import DEFAULT_SPECS, DetectorSpec, Post, calibrate, detect_mask, score_mask
from vuhf.data.sources import cached, synthetic_source, torchsig_source
from vuhf.geometry import SpecGeometry
from vuhf.spectrogram import power_spectrogram


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "torchsig"], default="torchsig")
    ap.add_argument("--n-scenes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--impairment-level", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--snr-db", type=float, nargs=2, default=[0.0, 20.0])
    ap.add_argument("--max-fa", type=float, default=2.0)
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("results/tuning.json"))
    args = ap.parse_args()

    n_samples = 262144
    geom = SpecGeometry(n_samples, 512, 256)
    key = f"{args.source}_imp{args.impairment_level}_snr{args.snr_db[0]:g}-{args.snr_db[1]:g}_seed{args.seed}"
    if args.source == "torchsig":
        factory = lambda: torchsig_source(args.n_scenes, args.seed, impairment_level=args.impairment_level,
                                          snr_db=tuple(args.snr_db), n_samples=n_samples)
    else:
        factory = lambda: synthetic_source(args.n_scenes, args.seed, n_samples=n_samples, snr_db_range=tuple(args.snr_db))
    scenes = list(cached(factory, key, args.n_scenes, args.cache_dir))
    print(f"{len(scenes)} validation scenes ({key})")

    specs = [s for s in DEFAULT_SPECS if s.kind == "energy" and s.floor == "global" and s.margin_db is None] + \
            [s for s in DEFAULT_SPECS if s.name in ("cfar_freq_g24_t48",)]
    avgs = [(1, 3), (3, 9)]
    posts = [Post(link, mc) for link, mc in itertools.product([(2, 6), (4, 16), (8, 32)], [12, 40])]
    thr_cache = {}
    results = []
    Ps = [power_spectrogram(s.iq, geom) for s in scenes]
    for base in specs:
        for avg in avgs:
            spec = DetectorSpec(**{**base.__dict__, "avg": avg, "name": f"{base.name}_avg{avg[0]}x{avg[1]}"})
            thr = thr_cache.setdefault(spec.name, calibrate(spec, geom, args.pfa))
            masks = [detect_mask(spec, P, thr) for P in Ps]
            for post in posts:
                sc = [score_mask(s.boxes, m, geom, post, iou_thrs=(0.3,)) for s, m in zip(scenes, masks)]
                rows = [r for x in sc for r in x["rows"]]
                results.append(dict(detector=spec.name, link=post.link, min_cells=post.min_cells,
                                    recall30=float(np.mean([r["det@0.3"] for r in rows])),
                                    covered=float(np.mean([r["covered"] for r in rows])),
                                    fa_per_scene=float(np.mean([x["n_fa"][0.3] for x in sc])),
                                    frag_per_scene=float(np.mean([x["n_frag"][0.3] for x in sc]))))
    ok = [r for r in results if r["fa_per_scene"] <= args.max_fa]
    print(f"\n{'detector':<30}{'link':>9}{'min':>5}{'recall30':>10}{'covered':>9}{'FA/sc':>8}{'frag/sc':>9}")
    for r in sorted(results, key=lambda r: (r["detector"], -r["recall30"])):
        mark = "*" if r in ok else " "
        print(f"{mark}{r['detector']:<29}{str(r['link']):>9}{r['min_cells']:>5}{r['recall30']:>10.3f}{r['covered']:>9.3f}"
              f"{r['fa_per_scene']:>8.2f}{r['frag_per_scene']:>9.2f}")
    print(f"\n* = at most {args.max_fa:g} false alarms per scene")
    best = {}
    for r in ok:
        d = r["detector"]
        if d not in best or r["recall30"] > best[d]["recall30"]:
            best[d] = r
    print("\nbest per detector under the false-alarm budget:")
    for d, r in best.items():
        print(f"  {d:<30} link={r['link']} min_cells={r['min_cells']} recall30={r['recall30']:.3f} FA/scene={r['fa_per_scene']:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(validation=key, max_fa=args.max_fa, all=results, best=best), indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
