"""Choose the box linking of the classical detectors and the YOLO confidence on ONE annotated real recording, to be applied
to a different recording (never to the windows that are reported).

Same selection rule as tune_postprocessing.py / tune_yolo.py: highest recall at IoU >= 0.3 among the settings with at most
``--max-fa`` false alarms per window (classical; one setting shared by energy and CA-CFAR, ranked by the mean of their recalls,
each of them within the budget) and ``--yolo-max-fa`` (YOLO).  Windows are spread evenly over the recording.

    python scripts/tune_real.py data_real/iqengine/bluetooth/bluetooth.sigmf-meta --out results/real/tuning_on_bluetooth.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np

from vuhf.baseline import DEFAULT_SPECS, calibrate, detect_mask, score_boxes
from vuhf.data.real import iter_windows, open_recording, window_boxes
from vuhf.detectors import mask_to_boxes
from vuhf.detectors.boxes import boxes_to_mask
from vuhf.detectors.yolo import YoloDetector
from vuhf.geometry import SpecGeometry
from vuhf.spectrogram import power_spectrogram

ROOT = Path(__file__).resolve().parents[1]
CLASSICAL = ("energy_median", "cfar_freq_g24_t48")
LINKS = [(1, 2), (2, 4), (2, 6), (4, 16), (8, 32)]
MIN_CELLS = [12, 40]
CONFS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--window", type=int, default=262144)
    ap.add_argument("--n-windows", type=int, default=40)
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--max-fa", type=float, default=2.0)
    ap.add_argument("--yolo-max-fa", type=float, default=0.25)
    ap.add_argument("--yolo-weights", type=Path, default=ROOT / "results/yolo/best.pt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    rec = open_recording(args.input)
    n_all = (rec.n_samples - args.window) // args.window + 1
    keep = set(np.linspace(0, n_all - 1, min(args.n_windows, n_all)).round().astype(int).tolist())
    geom = SpecGeometry(args.window, 512, 256)
    specs = {s.name: s for s in DEFAULT_SPECS}
    thr = {k: calibrate(specs[k], geom, args.pfa) for k in CLASSICAL}
    yolo = YoloDetector(args.yolo_weights, geom, min_conf=min(CONFS))

    wins = []                                   # per window: ground truth, raw masks, raw YOLO boxes with confidences
    for i, (start, iq) in enumerate(iter_windows(rec, args.window)):
        if i not in keep:
            continue
        P = power_spectrogram(np.nan_to_num(iq), geom)
        wins.append(dict(gts=window_boxes(rec, start, args.window), masks={k: detect_mask(specs[k], P, thr[k]) for k in CLASSICAL},
                         yolo=yolo.predict_raw([P])[0]))
    n_gt = sum(len(w["gts"]) for w in wins)
    print(f"{args.input.name}: {len(wins)} windows, {n_gt} annotated signals")

    def score(preds_per_window):
        sc = [score_boxes(w["gts"], p, boxes_to_mask(p, geom), geom, iou_thrs=(0.3,)) for w, p in zip(wins, preds_per_window)]
        rows = [r for s in sc for r in s["rows"]]
        return float(np.mean([r["det@0.3"] for r in rows])), float(np.mean([s["n_fa"][0.3] for s in sc]))

    grid = []
    for link, mc in itertools.product(LINKS, MIN_CELLS):
        res = {k: score([mask_to_boxes(w["masks"][k], geom, link=link, min_cells=mc) for w in wins]) for k in CLASSICAL}
        grid.append(dict(link=list(link), min_cells=mc, **{f"{k}_recall30": res[k][0] for k in CLASSICAL},
                         **{f"{k}_fa_per_window": res[k][1] for k in CLASSICAL}))
    ok = [g for g in grid if all(g[f"{k}_fa_per_window"] <= args.max_fa for k in CLASSICAL)]
    best_post = max(ok, key=lambda g: np.mean([g[f"{k}_recall30"] for k in CLASSICAL])) if ok else None

    ygrid = []
    for conf in CONFS:
        r, fa = score([[b for b, c in w["yolo"] if c >= conf] for w in wins])
        ygrid.append(dict(conf=conf, recall30=r, fa_per_window=fa))
    yok = [g for g in ygrid if g["fa_per_window"] <= args.yolo_max_fa]
    best_conf = max(yok, key=lambda g: g["recall30"]) if yok else None

    print(f"\n{'link':>9}{'min':>5}" + "".join(f"{k[:12]:>22}" for k in CLASSICAL) + "   (recall30 / FA per window)")
    for g in grid:
        print(f"{str(tuple(g['link'])):>9}{g['min_cells']:>5}" + "".join(f"{g[k + '_recall30']:>14.3f} / {g[k + '_fa_per_window']:<5.2f}" for k in CLASSICAL)
              + ("   *" if g in ok else ""))
    print("\nYOLO  conf  recall30  FA/window")
    for g in ygrid:
        print(f"      {g['conf']:<6}{g['recall30']:<10.3f}{g['fa_per_window']:<8.2f}{'*' if g in yok else ''}")
    print(f"\nchosen post-processing: {best_post and (tuple(best_post['link']), best_post['min_cells'])}; YOLO conf: {best_conf and best_conf['conf']}"
          f"   (* = within the false-alarm budget; None = no setting within budget)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(input=args.input.name, n_windows=len(wins), n_annotated=n_gt, max_fa=args.max_fa, yolo_max_fa=args.yolo_max_fa,
                                         post_grid=grid, chosen_post=best_post, yolo_grid=ygrid, chosen_yolo=best_conf), indent=2))


if __name__ == "__main__":
    main()
