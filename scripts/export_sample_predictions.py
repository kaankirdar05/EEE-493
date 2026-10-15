"""Run the energy detector, CA-CFAR and YOLO on the sample scenes and save what each of them predicts.

Reads the scenes of ``data_sample/iq`` (see export_sample.py) and writes to ``data_sample/predictions/``:

    <detector>/imp<L>_scene<NNNN>.txt   predicted boxes in YOLO format on the sample spectrogram image
                                        (``0 x_centre y_centre width height confidence``, fractions of the image;
                                        confidence is 1 for the energy detector and CFAR), one folder per detector
    predictions.csv                     every predicted box in physical units (ms, MHz), with the true signal it matches
    scene_summary.csv                   per scene and detector: true signals, found at IoU 0.3 / 0.5, extra boxes
    figures/comparison_imp<L>.png       the scenes of one impairment level, one column per detector

    python scripts/export_sample_predictions.py --yolo-weights results/yolo/best.pt --yolo-conf 0.05
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from plot_comparison import DETECTORS, FS, plot_scenes
from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask
from vuhf.data.sources import cached
from vuhf.detectors import mask_to_boxes
from vuhf.detectors.yolo import YoloDetector, box_to_pixels, image_width
from vuhf.eval.matching import match_boxes
from vuhf.geometry import SpecGeometry
from vuhf.spectrogram import power_spectrogram

CLASSICAL = ("energy_median", "cfar_freq_g24_t48")


def yolo_lines(preds, geom):
    n, w = geom.n_fft, image_width(geom)
    out = []
    for b, conf in preds:
        x0, y0, x1, y1 = box_to_pixels(b, geom)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        out.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / n:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / n:.6f} {conf:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--yolo-conf", type=float, default=0.05)
    ap.add_argument("--sample-dir", type=Path, default=Path("data_sample"))
    ap.add_argument("--n-per-level", type=int, default=4)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    geom = SpecGeometry(262144, 512, 256)
    specs = {s.name: s for s in DEFAULT_SPECS}
    thr = {n: calibrate(specs[n], geom, 1e-3) for n in CLASSICAL}
    post = Post()
    yolo = YoloDetector(args.yolo_weights, geom, min_conf=args.yolo_conf)
    out = args.sample_dir / "predictions"
    for sub in [*(n for n, _, _ in DETECTORS), "figures"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    pred_rows, summary_rows = [], []
    for lv in args.levels:
        key = f"torchsig_imp{lv}_snr0-20_seed{args.seed}"
        gallery = []
        for i, sc in enumerate(cached(None, key, args.n_per_level, args.sample_dir / "iq")):
            P = power_spectrogram(sc.iq, geom)
            name = f"imp{lv}_scene{i:04d}"
            scored = {n: [(b, 1.0) for b in mask_to_boxes(detect_mask(specs[n], P, thr[n]), geom, link=post.link, min_cells=post.min_cells)]
                      for n in CLASSICAL}
            scored["yolo"] = [(b, c) for b, c in yolo.predict_raw([P])[0] if c >= args.yolo_conf]
            gts = sc.boxes
            for det, _, _ in DETECTORS:
                preds = scored[det]
                (out / det / f"{name}.txt").write_text("\n".join(yolo_lines(preds, geom)) + ("\n" if preds else ""))
                m3 = {i_: (j, v) for i_, j, v in match_boxes([b for b, _ in preds], gts, 0.3)}
                found3 = len({j for j, _ in m3.values()})
                found5 = len({j for _, j, _ in match_boxes([b for b, _ in preds], gts, 0.5)})
                for k, (b, conf) in enumerate(preds):
                    j, v = m3.get(k, (None, 0.0))
                    pred_rows.append(dict(impairment_level=lv, scene=i, detector=det, confidence=round(conf, 3),
                                          start_ms=round(b.t0 / FS * 1e3, 3), stop_ms=round(b.t1 / FS * 1e3, 3),
                                          freq_low_mhz=round(b.f0 * FS / 1e6, 3), freq_high_mhz=round(b.f1 * FS / 1e6, 3),
                                          matches_true_signal=int(j is not None), matched_signal_type=gts[j].label if j is not None else "",
                                          iou=round(v, 3) if j is not None else ""))
                summary_rows.append(dict(impairment_level=lv, scene=i, detector=det, true_signals=len(gts), found_iou30=found3,
                                         found_iou50=found5, predicted_boxes=len(preds), extra_boxes=len(preds) - len(m3)))
            gallery.append(dict(index=i, level=lv, P=P.astype(np.float32), gts=gts, preds={d: [b for b, _ in v] for d, v in scored.items()}))
        plot_scenes(gallery, geom, out / "figures" / f"comparison_imp{lv}.png",
                    f"Sample scenes, impairment level {lv}: true signals (white dashed) and the three detectors")
        print(f"level {lv}: {len(gallery)} scenes", flush=True)

    for fname, rows in (("predictions.csv", pred_rows), ("scene_summary.csv", summary_rows)):
        with open(out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for det, label, _ in DETECTORS:
        r = [x for x in summary_rows if x["detector"] == det]
        print(f"{label:<16} found {sum(x['found_iou30'] for x in r)} of {sum(x['true_signals'] for x in r)} signals (IoU 0.3), "
              f"{sum(x['extra_boxes'] for x in r)} extra boxes")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
