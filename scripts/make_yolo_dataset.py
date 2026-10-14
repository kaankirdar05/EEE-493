"""Generate a YOLO dataset (spectrogram images + box labels) from TorchSig scenes.

    python scripts/make_yolo_dataset.py --split train --n-per-level 1000 --seed-base 100
    python scripts/make_yolo_dataset.py --split val   --n-per-level 60   --seed-base 11 --cache-iq

Layout written under ``--out`` (default data_cache/yolo):  images/<split>/*.png, labels/<split>/*.txt, vuhf.yaml.
Every (impairment level, chunk) uses its own TorchSig seed, so no two scenes are shared.  Training scenes
use seeds >= 100, validation 11, the tuning set of the classical detectors 7 and the final evaluation 1000.
``--cache-iq`` also stores the I/Q scenes in ``data_cache`` (the format of run_detection_baseline.py), which the
validation set needs so that the confidence threshold can be chosen with the package's own scorer.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

N_SAMPLES = 262144


def _write(scene, name, out, split, geom):
    from vuhf.detectors.yolo import spectrogram_image, yolo_label_lines
    from vuhf.spectrogram import power_spectrogram

    img = spectrogram_image(power_spectrogram(scene.iq, geom))
    cv2.imwrite(str(out / "images" / split / f"{name}.png"), img)
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(yolo_label_lines(scene.boxes, geom)))


def work(task):
    import torch

    torch.set_num_threads(1)
    from vuhf.data.sources import cached, torchsig_source
    from vuhf.geometry import SpecGeometry

    split, level, seed, n, snr, out, cache_iq, cache_dir = task
    geom = SpecGeometry(N_SAMPLES, 512, 256)
    factory = lambda: torchsig_source(n, seed, impairment_level=level, snr_db=snr, n_samples=N_SAMPLES)
    if cache_iq:
        key = f"torchsig_imp{level}_snr{snr[0]:g}-{snr[1]:g}_seed{seed}"
        scenes = cached(factory, key, n, cache_dir)
    else:
        scenes = factory()
    for i, sc in enumerate(scenes):
        _write(sc, f"{split}_L{level}_s{seed}_{i:04d}", out, split, geom)
    return split, level, seed, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--n-per-level", type=int, required=True)
    ap.add_argument("--chunks-per-level", type=int, default=4, help="separate seeds (and processes) per impairment level")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--snr-db", type=float, nargs=2, default=[0.0, 20.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-iq", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data_cache/yolo"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    args = ap.parse_args()

    (args.out / "images" / args.split).mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / args.split).mkdir(parents=True, exist_ok=True)
    per = args.n_per_level // args.chunks_per_level
    tasks = [(args.split, lv, args.seed_base + 1000 * lv + c, per, tuple(args.snr_db), args.out.resolve(), args.cache_iq,
              args.cache_dir) for lv in args.levels for c in range(args.chunks_per_level)]
    if args.cache_iq and len(args.levels) > 1 and args.chunks_per_level > 1:
        raise SystemExit("--cache-iq needs one seed per level: pass --chunks-per-level 1")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for split, lv, seed, n in pool.imap_unordered(work, tasks):
            print(f"done {split} level {lv} seed {seed}: {n} scenes", flush=True)
    (args.out / "vuhf.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: signal\n")
    n_img = len(list((args.out / "images" / args.split).glob("*.png")))
    print(f"{args.split}: {n_img} images in {args.out}")


if __name__ == "__main__":
    main()
