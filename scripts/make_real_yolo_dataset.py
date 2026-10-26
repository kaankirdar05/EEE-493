"""Spectrogram images + box labels for training YOLO on an annotated real recording (SigMF with annotations).

    python scripts/make_real_yolo_dataset.py data_real/iqengine/dect6/dect6.sigmf-meta --out C:/yr/dect6

The recording is cut into 262,144-sample windows (the size used everywhere else), the windows are split in time by
``block_split`` (interleaved train / validation / test blocks with a gap between them) and only the train and validation blocks are written:
the test block is never turned into images here.  Train windows overlap within a train block (stride = window / ``--stride-div``) to get more images
from a short recording; validation windows do not.  A window without any annotation is kept as an image with an empty label file
(it teaches the network what an empty window looks like).  Labels follow the scoring rule of ``window_boxes`` (an annotation counts in a
window if at least half of its duration lies inside it).  Layout: images/{train,val}/*.png and labels/{train,val}/*.txt under ``--out``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vuhf.data.real import block_split, open_recording, runs, window_boxes
from vuhf.detectors.yolo import spectrogram_image, yolo_label_lines
from vuhf.geometry import SpecGeometry
from vuhf.spectrogram import power_spectrogram

W = 262144


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride-div", type=int, default=4, help="train windows start every window / this many samples")
    args = ap.parse_args()

    rec = open_recording(args.recording)
    n_all = (rec.n_samples - W) // W + 1
    sp = block_split(n_all)
    print(f"{args.recording.name}: {n_all} windows; train blocks {runs(sp['train'])}, val {runs(sp['val'])}, test {runs(sp['test'])} (test not written)")
    geom = SpecGeometry(W, 512, 256)
    starts = {"train": [s for a, b in runs(sp["train"]) for s in range(a * W, (b + 1) * W - W + 1, W // args.stride_div)],
              "val": [i * W for i in sp["val"]]}
    for split, ss in starts.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_box = n_empty = 0
        for s in ss:
            P = power_spectrogram(np.nan_to_num(rec.read(s, W)), geom)
            lines = yolo_label_lines(window_boxes(rec, s, W), geom)
            name = f"{args.recording.stem}_{split}_{s}"
            cv2.imwrite(str(args.out / "images" / split / f"{name}.png"), spectrogram_image(P))
            (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            n_box += len(lines)
            n_empty += not lines
        print(f"  {split}: {len(ss)} images, {n_box} boxes, {n_empty} without any box")


if __name__ == "__main__":
    main()
