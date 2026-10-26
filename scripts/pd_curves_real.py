"""Probability of detection against SNR on the annotated real recordings (dect6, bluetooth), in the style of the synthetic pd_curves.png.

    python scripts/pd_curves_real.py data_real/iqengine/dect6/dect6.sigmf-meta data_real/iqengine/bluetooth/bluetooth.sigmf-meta \\
        --joint results/real_yolo/weights/joint.pt --out results/real_pd

The recordings carry no SNR that is comparable across the two files, so the SNR of every annotated signal is estimated the same way
for both: mean power inside its annotated time-frequency box over the noise level of the window (median / ln 2), minus the noise itself,
in dB.  The box is the nominal band, so this underestimates the SNR in the occupied part of the band a little.

Two figures per recording: ``all`` = every full window, the detectors that were not trained on the recording (energy and CA-CFAR with the
reference linking (4, 16) and with (1, 2) / 40 cells, YOLO trained on synthetic scenes); ``test`` = only the test blocks of
``block_split``, where the YOLO retrained on the recording's train blocks (--joint) is added.  Top row: a predicted box matches the signal
with IoU >= 0.3; bottom row: at least 30 % of the signal's cells lie inside a predicted box (found, whatever the box shape).
The annotated signals span only 14 to 37 dB (dect6: three discrete power levels), so these curves cannot show the low-SNR behaviour of the
synthetic pd_curves.png (0 to 20 dB).
The shaded band is the 95 % Wilson interval of each bin; bins with fewer than 3 signals are not drawn.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask, score_boxes  # noqa: E402
from vuhf.data.real import block_split, open_recording, window_boxes  # noqa: E402
from vuhf.detectors import mask_to_boxes  # noqa: E402
from vuhf.detectors.boxes import boxes_to_mask  # noqa: E402
from vuhf.detectors.yolo import YoloDetector  # noqa: E402
from vuhf.geometry import SpecGeometry  # noqa: E402
from vuhf.spectrogram import power_spectrogram  # noqa: E402

W = 262144
ROOT = Path(__file__).resolve().parents[1]
CLASSICAL = ("energy_median", "cfar_freq_g24_t48")
STYLE = {   # name -> (label, colour, linestyle)
    "energy": ("Energy, link (4, 16)", "#2a78d6", "-"),
    "energy_l12": ("Energy, link (1, 2)", "#2a78d6", "--"),
    "cfar": ("CA-CFAR, link (4, 16)", "#eb6834", "-"),
    "cfar_l12": ("CA-CFAR, link (1, 2)", "#eb6834", "--"),
    "yolo_syn": ("YOLO, synthetic-trained", "#1baf7a", "-"),
    "yolo_joint": ("YOLO, retrained on the recording", "#9b59b6", "-"),
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def snr_estimate(P, geom, box) -> float:
    k0, k1, j0, j1 = geom.box_to_cells(box)
    floor = np.median(P) / np.log(2)
    return float(10 * np.log10(max(P[k0 : k1 + 1, j0 : j1 + 1].mean() / floor - 1.0, 1e-3)))


def plot(rows, names, out, title, min_n=3, nbins=8):
    snr = np.array([r["snr_db"] for r in rows["energy"]])
    edges = np.linspace(np.floor(snr.min()), np.ceil(snr.max()), nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(snr, edges[:-1]) - 1
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.3), sharex=True, sharey=True)
    for ax, (flag, label) in zip(axes, [("det30", "Pd, IoU >= 0.3"), ("covered", "Pd, covered >= 30 %")]):
        for n in names:
            lab, col, ls = STYLE[n]
            det = np.array([r[flag] for r in rows[n]], dtype=bool)
            xs, ys, lo, hi = [], [], [], []
            for k in range(len(centres)):
                sel = idx == k
                if sel.sum() >= min_n:
                    p = det[sel].mean()
                    a, b = wilson(int(det[sel].sum()), int(sel.sum()))
                    xs.append(centres[k]); ys.append(p); lo.append(a); hi.append(b)
            ax.plot(xs, ys, marker="o", ms=4, color=col, ls=ls, label=lab)
            if ls == "-":
                ax.fill_between(xs, lo, hi, color=col, alpha=0.08, lw=0)
        ax.set(ylabel=label, ylim=(-0.02, 1.04))
        ax.set_facecolor("white")
        ax.grid(color="#dcdbd7", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#b5b4af")
        ax.tick_params(colors="#52514e", labelsize=9)
    n_bin = [int(np.sum(idx == k)) for k in range(len(centres))]
    axes[1].set_xlim(edges[0] - 0.4, edges[-1] + 0.4)
    axes[1].set_xlabel("signal SNR (dB), estimated from the data")
    for k, c in enumerate(centres):
        if n_bin[k]:
            axes[1].text(c, -0.015, f"n={n_bin[k]}", ha="center", va="bottom", fontsize=6.5, color="0.35")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--joint", type=Path, required=True, help="YOLO weights retrained on the recordings' train blocks")
    ap.add_argument("--yolo-weights", type=Path, default=ROOT / "results/yolo/best.pt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args.out.mkdir(parents=True, exist_ok=True)
    geom = SpecGeometry(W, 512, 256)
    specs = {s.name: s for s in DEFAULT_SPECS}
    thr = {k: calibrate(specs[k], geom, 1e-3) for k in CLASSICAL}
    yolo = {"yolo_syn": YoloDetector(args.yolo_weights, geom, min_conf=0.05), "yolo_joint": YoloDetector(args.joint, geom, min_conf=0.05)}
    summary = {}

    for path in args.recordings:
        rec = open_recording(path)
        stem = path.name.split(".")[0]
        n_all = (rec.n_samples - W) // W + 1
        test = set(block_split(n_all)["test"])
        rows = {n: [] for n in STYLE}          # one record per annotated signal: snr, det30, det50, covered, in_test
        for i in range(n_all):
            s = i * W
            P = power_spectrogram(np.nan_to_num(rec.read(s, W)), geom)
            gts = [dataclasses.replace(g, snr_db=snr_estimate(P, geom, g)) for g in window_boxes(rec, s, W)]
            if not gts:
                continue
            preds = {}
            for k, short in zip(CLASSICAL, ("energy", "cfar")):
                m = detect_mask(specs[k], P, thr[k])
                preds[short] = mask_to_boxes(m, geom, link=Post().link, min_cells=Post().min_cells)
                preds[short + "_l12"] = mask_to_boxes(m, geom, link=(1, 2), min_cells=40)
            for n, d in yolo.items():
                preds[n] = [b for b, c in d.predict_raw([P])[0] if c >= 0.05]
            for n, pb in preds.items():
                sc = score_boxes(gts, pb, boxes_to_mask(pb, geom), geom, iou_thrs=(0.5, 0.3))
                for r in sc["rows"]:
                    rows[n].append(dict(window=i, snr_db=r["snr_db"], det30=r["det@0.3"], det50=r["det@0.5"], covered=r["covered"], in_test=i in test))
        with open(args.out / f"per_signal_{stem}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["detector", "window", "in_test", "snr_db", "det_iou30", "det_iou50", "covered"])
            for n, rs in rows.items():
                for r in rs:
                    w.writerow([n, r["window"], r["in_test"], f"{r['snr_db']:.2f}", int(r["det30"]), int(r["det50"]), int(r["covered"])])
        # figure 1: all windows, detectors not trained on the recording
        names_all = ["energy", "energy_l12", "cfar", "cfar_l12", "yolo_syn"]
        plot({n: rows[n] for n in names_all}, names_all, args.out / f"pd_{stem}_all.png",
             f"{stem}: Pd against SNR, all {n_all} windows ({len(rows['energy'])} signals)")
        # figure 2: test blocks only, with the retrained YOLO
        names_test = list(STYLE)
        rt = {n: [r for r in rows[n] if r["in_test"]] for n in names_test}
        plot(rt, names_test, args.out / f"pd_{stem}_test.png",
             f"{stem}: Pd against SNR, test blocks only ({len(rt['energy'])} signals; the retrained YOLO never saw these windows)", nbins=5)
        snr = np.array([r["snr_db"] for r in rows["energy"]])
        summary[stem] = dict(n_signals=len(snr), snr_db_percentiles={str(q): float(np.percentile(snr, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
                             n_signals_test=len(rt["energy"]))
        print(stem, summary[stem], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
