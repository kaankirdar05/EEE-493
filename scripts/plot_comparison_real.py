"""Comparison scenes on real recordings, in the style of plot_comparison.py: the same test windows, one column per detector.

    python scripts/plot_comparison_real.py data_real/iqengine/dect6/dect6.sigmf-meta data_real/iqengine/bluetooth/bluetooth.sigmf-meta \\
        --joint results/real_yolo/weights/joint.pt --out results/real_scenes

Windows are drawn at random (fixed ``--seed``, not hand-picked) from the TEST blocks of ``block_split`` that contain at least one annotation, so the
YOLO retrained on the recording's train blocks never saw them.  Columns: energy detector and CA-CFAR with the reference box linking (4, 16),
YOLO trained on synthetic scenes, YOLO retrained on the recording (both at confidence 0.05).  White dashed = annotated signal, solid colour = detection that
matches a signal (IoU >= 0.3), dotted colour = detection that matches nothing (false alarm or fragment).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask  # noqa: E402
from vuhf.data.real import block_split, open_recording, window_boxes  # noqa: E402
from vuhf.detectors import mask_to_boxes  # noqa: E402
from vuhf.detectors.yolo import YoloDetector  # noqa: E402
from vuhf.eval.matching import match_boxes  # noqa: E402
from vuhf.geometry import SpecGeometry  # noqa: E402
from vuhf.spectrogram import power_spectrogram  # noqa: E402

W = 262144
ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [("energy", "Energy detector", "#2a78d6"), ("cfar", "CA-CFAR", "#eb6834"),
           ("yolo_syn", "YOLO, synthetic-trained", "#1baf7a"), ("yolo_joint", "YOLO, retrained on the recording", "#9b59b6")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd7"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--joint", type=Path, required=True)
    ap.add_argument("--yolo-weights", type=Path, default=ROOT / "results/yolo/best.pt")
    ap.add_argument("--n-show", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args.out.mkdir(parents=True, exist_ok=True)
    geom = SpecGeometry(W, 512, 256)
    specs = {s.name: s for s in DEFAULT_SPECS}
    thr = {k: calibrate(specs[k], geom, 1e-3) for k in ("energy_median", "cfar_freq_g24_t48")}
    yolo = {"yolo_syn": YoloDetector(args.yolo_weights, geom, min_conf=0.05), "yolo_joint": YoloDetector(args.joint, geom, min_conf=0.05)}
    halo = [pe.withStroke(linewidth=3.2, foreground="black")]

    for path in args.recordings:
        rec = open_recording(path)
        stem = path.name.split(".")[0]
        n_all = (rec.n_samples - W) // W + 1
        pool = [i for i in block_split(n_all)["test"] if window_boxes(rec, i * W, W)]
        rng = np.random.default_rng(args.seed)
        chosen = sorted(rng.choice(pool, size=min(args.n_show, len(pool)), replace=False).tolist())
        fs, fc = rec.fs, rec.fc
        fig, axes = plt.subplots(len(chosen), 4, figsize=(15.5, 2.55 * len(chosen) + 1.0), squeeze=False)
        t_ms = W / fs * 1e3
        for r, i in enumerate(chosen):
            s = i * W
            P = power_spectrogram(np.nan_to_num(rec.read(s, W)), geom)
            db = 10 * np.log10(np.maximum(P, 1e-12) / (np.median(P) / np.log(2)))
            gts = window_boxes(rec, s, W)
            preds = {}
            for k, short in (("energy_median", "energy"), ("cfar_freq_g24_t48", "cfar")):
                preds[short] = mask_to_boxes(detect_mask(specs[k], P, thr[k]), geom, link=Post().link, min_cells=Post().min_cells)
            for n, d in yolo.items():
                preds[n] = [b for b, c in d.predict_raw([P])[0] if c >= 0.05]
            for c, (name, label, color) in enumerate(COLUMNS):
                ax = axes[r, c]
                ax.imshow(np.clip(db, -3, 20), origin="lower", aspect="auto", cmap="gray", vmin=-3, vmax=24,
                          extent=(0, t_ms, (fc - fs / 2) / 1e6, (fc + fs / 2) / 1e6))
                for g in gts:
                    ax.add_patch(plt.Rectangle((g.t0 / fs * 1e3, (fc + g.f0 * fs) / 1e6), g.duration / fs * 1e3, g.bandwidth * fs / 1e6, fill=False,
                                               ec="white", ls="--", lw=1.3, path_effects=halo))
                pr = preds[name]
                m = match_boxes(pr, gts, 0.3)
                ok, found = {a for a, _, _ in m}, len({b for _, b, _ in m})
                for k, b in enumerate(pr):
                    ax.add_patch(plt.Rectangle((b.t0 / fs * 1e3, (fc + b.f0 * fs) / 1e6), b.duration / fs * 1e3, b.bandwidth * fs / 1e6, fill=False,
                                               ec=color, ls="-" if k in ok else ":", lw=2.0, path_effects=halo))
                ax.text(0.01, 0.03, f"found {found} of {len(gts)}   extra boxes {len(pr) - len(ok)}", transform=ax.transAxes, fontsize=9.5, color="white",
                        path_effects=[pe.withStroke(linewidth=3, foreground="black")], va="bottom")
                if r == 0:
                    ax.set_title(label, fontsize=11.5, color=INK, fontweight="bold", pad=8)
                if c == 0:
                    ax.set_ylabel(f"window {i}\nfrequency (MHz)", fontsize=9.5, color=INK2)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=8.5, colors=INK2)
                if r == len(chosen) - 1:
                    ax.set_xlabel("time (ms)", fontsize=9.5, color=INK2)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        handles = [plt.Line2D([], [], color="white", ls="--", lw=1.6, path_effects=halo), plt.Line2D([], [], color="#888", lw=2),
                   plt.Line2D([], [], color="#888", lw=2, ls=":")]
        fig.legend(handles, ["annotated signal", "detection that matches an annotated signal (IoU >= 0.3)", "detection that matches nothing (false alarm or fragment)"],
                   loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"{stem}: the same test windows, four detectors (windows drawn at random from the test blocks)", y=1.03, fontsize=12.5, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        for c, (name, label, color) in enumerate(COLUMNS):
            pos = axes[0, c].get_position()
            fig.add_artist(plt.Line2D([pos.x0, pos.x1], [pos.y1 + 0.034] * 2, color=color, lw=4, transform=fig.transFigure))
        out = args.out / f"comparison_scenes_{stem}.png"
        fig.savefig(out, dpi=105, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(out, "windows", chosen, flush=True)


if __name__ == "__main__":
    main()
