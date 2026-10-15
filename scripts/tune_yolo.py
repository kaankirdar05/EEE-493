"""Choose the YOLO confidence threshold on the validation scenes (never on the evaluation scenes).

Runs the trained model once on the cached validation scenes (make_yolo_dataset.py --split val --cache-iq),
then sweeps the confidence threshold.  Selection rule, like tune_postprocessing.py: highest recall at IoU >= 0.3
among the thresholds with at most ``--max-fa`` false-alarm boxes per scene.  The default budget of 0.25 is about
what the energy detector produces at its design Pfa, so the comparison is at a similar false-alarm level.

    python scripts/tune_yolo.py --weights data_cache/yolo_runs/signal/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vuhf.baseline import score_boxes
from vuhf.data.sources import cached
from vuhf.detectors.boxes import boxes_to_mask
from vuhf.detectors.yolo import YoloDetector
from vuhf.geometry import SpecGeometry
from vuhf.spectrogram import power_spectrogram

CONFS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--seed-base", type=int, default=11)
    ap.add_argument("--n-per-level", type=int, default=60)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--snr-db", type=float, nargs=2, default=[0.0, 20.0])
    ap.add_argument("--max-fa", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("results/yolo_tuning.json"))
    args = ap.parse_args()

    geom = SpecGeometry(262144, 512, 256)
    det = YoloDetector(args.weights, geom, min_conf=min(CONFS), nms_iou=args.nms_iou)
    scenes, raw = [], []
    for lv in args.levels:
        seed = args.seed_base + 1000 * lv
        key = f"torchsig_imp{lv}_snr{args.snr_db[0]:g}-{args.snr_db[1]:g}_seed{seed}"
        for sc in cached(None, key, args.n_per_level, args.cache_dir):
            scenes.append(sc)
            raw.append(det.predict_raw([power_spectrogram(sc.iq, geom)])[0])
    print(f"{len(scenes)} validation scenes")

    rows = []
    for conf in CONFS:
        sc_res = []
        for sc, r in zip(scenes, raw):
            preds = [b for b, c in r if c >= conf]
            sc_res.append(score_boxes(sc.boxes, preds, boxes_to_mask(preds, geom), geom, iou_thrs=(0.5, 0.3)))
        g = [x for s in sc_res for x in s["rows"]]
        rows.append(dict(conf=conf, recall30=float(np.mean([x["det@0.3"] for x in g])), recall50=float(np.mean([x["det@0.5"] for x in g])),
                         fa_per_scene=float(np.mean([s["n_fa"][0.3] for s in sc_res])),
                         frag_per_scene=float(np.mean([s["n_frag"][0.3] for s in sc_res])),
                         pred_per_scene=float(np.mean([s["n_pred"] for s in sc_res]))))
    ok = [r for r in rows if r["fa_per_scene"] <= args.max_fa]
    best = max(ok, key=lambda r: r["recall30"]) if ok else min(rows, key=lambda r: r["fa_per_scene"])
    print(f"{'conf':>6}{'rec@.3':>8}{'rec@.5':>8}{'FA/sc':>7}{'frag/sc':>9}{'pred/sc':>9}")
    for r in rows:
        print(f"{'*' if r is best else ' '}{r['conf']:>5.2f}{r['recall30']:>8.3f}{r['recall50']:>8.3f}{r['fa_per_scene']:>7.2f}"
              f"{r['frag_per_scene']:>9.2f}{r['pred_per_scene']:>9.2f}")
    print(f"\nchosen confidence {best['conf']} (max FA/scene {args.max_fa})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(weights=str(args.weights), validation_seeds=[args.seed_base + 1000 * lv for lv in args.levels],
                                        max_fa=args.max_fa, nms_iou=args.nms_iou, chosen_conf=best["conf"], sweep=rows), indent=2))


if __name__ == "__main__":
    main()
