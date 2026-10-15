"""Export a small, readable sample of the test scenes into ``data_sample/`` (small enough for git).

Takes the first ``--n-per-level`` scenes (indices 0, 1, ...; not hand-picked) of every impairment level from the
cache that run_detection_baseline.py fills, and writes

    iq/                    the scenes as .npz (iq, boxes, labels), same names as the cache, so
                           ``--cache-dir data_sample/iq`` makes the evaluation scripts run on them
    spectrogram_images/    the 512 x 512 grayscale image YOLO sees (.png) and its box labels (.txt, YOLO format)
    annotated/             the spectrogram with the true boxes, signal types and SNRs drawn on it (for people)
    labels.csv             one row per true signal: file, level, class, SNR, start/stop in ms, low/high frequency in MHz
    overview.png           all sample scenes on one page

    python scripts/export_sample.py --n-per-level 4
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from vuhf.data.sources import cached  # noqa: E402
from vuhf.detectors.yolo import spectrogram_image, yolo_label_lines  # noqa: E402
from vuhf.geometry import SpecGeometry  # noqa: E402
from vuhf.spectrogram import power_spectrogram  # noqa: E402

FS = 10e6          # sample rate used when the scenes were generated
N_SAMPLES = 262144


def draw(ax, scene, P, title, small=False):
    db = 10 * np.log10(P / (np.median(P) / np.log(2)))
    t_ms = N_SAMPLES / FS * 1e3
    ax.imshow(np.clip(db, -3, 20), origin="lower", aspect="auto", cmap="gray", vmin=-3, vmax=24,
              extent=(0, t_ms, -FS / 2e6, FS / 2e6))
    halo = [pe.withStroke(linewidth=2.6, foreground="black")]
    for b in scene.boxes:
        x0, y0 = b.t0 / FS * 1e3, b.f0 * FS / 1e6
        ax.add_patch(plt.Rectangle((x0, y0), b.duration / FS * 1e3, b.bandwidth * FS / 1e6, fill=False, ec="#ffd23f", lw=1.4, path_effects=halo))
        ax.text(x0 + 0.3, y0 + b.bandwidth * FS / 1e6 + 0.1, f"{b.label}  {b.snr_db:.0f} dB" if not small else f"{b.label}",
                color="#ffd23f", fontsize=5.5 if small else 8, va="bottom", path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    ax.set_title(title, fontsize=7 if small else 10, loc="left")
    ax.tick_params(labelsize=6 if small else 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-level", type=int, default=4)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--n-scenes", type=int, default=300, help="size of the cached evaluation set the sample is taken from")
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=Path("data_sample"))
    args = ap.parse_args()

    geom = SpecGeometry(N_SAMPLES, 512, 256)
    for sub in ("iq", "spectrogram_images", "annotated"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    rows, thumbs = [], []
    for lv in args.levels:
        key = f"torchsig_imp{lv}_snr0-20_seed{args.seed}"
        for i, sc in enumerate(cached(None, key, args.n_scenes, args.cache_dir)):
            if i >= args.n_per_level:
                break
            shutil.copy(args.cache_dir / f"{key}_{i:04d}.npz", args.out / "iq" / f"{key}_{i:04d}.npz")
            P = power_spectrogram(sc.iq, geom)
            name = f"imp{lv}_scene{i:04d}"
            Image.fromarray(spectrogram_image(P)).save(args.out / "spectrogram_images" / f"{name}.png", optimize=True)
            (args.out / "spectrogram_images" / f"{name}.txt").write_text("\n".join(yolo_label_lines(sc.boxes, geom)) + "\n")
            title = f"{name}: impairment level {lv}, {len(sc.boxes)} signals"
            fig, ax = plt.subplots(figsize=(8, 4.2))
            draw(ax, sc, P, title)
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("frequency (MHz)")
            fig.tight_layout()
            fig.savefig(args.out / "annotated" / f"{name}.png", dpi=90)
            plt.close(fig)
            thumbs.append((sc, P, title))
            for b in sc.boxes:
                rows.append(dict(file=f"iq/{key}_{i:04d}.npz", impairment_level=lv, scene=i, signal_type=b.label, snr_db=round(b.snr_db, 1),
                                 start_ms=round(b.t0 / FS * 1e3, 3), stop_ms=round(b.t1 / FS * 1e3, 3),
                                 freq_low_mhz=round(b.f0 * FS / 1e6, 3), freq_high_mhz=round(b.f1 * FS / 1e6, 3),
                                 bandwidth_mhz=round(b.bandwidth * FS / 1e6, 3)))
    with open(args.out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    nc = args.n_per_level
    fig, axes = plt.subplots(len(args.levels), nc, figsize=(3.6 * nc, 2.6 * len(args.levels)), squeeze=False, sharex=True, sharey=True)
    for ax, (sc, P, title) in zip(axes.ravel(), thumbs):
        draw(ax, sc, P, title.split(": ")[0], small=True)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (MHz)", fontsize=7)
    fig.suptitle("Sample test scenes: first scenes of each impairment level (yellow = true signals)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "overview.png", dpi=100)
    plt.close(fig)
    print(f"{len(thumbs)} scenes, {len(rows)} signals written to {args.out}")


if __name__ == "__main__":
    main()
