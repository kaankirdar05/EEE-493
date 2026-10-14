"""Train a YOLO signal detector (one class) on the spectrogram images from make_yolo_dataset.py.

    python scripts/train_yolo.py --hours 1.5

Runs on CPU by default.  ``--hours`` gives Ultralytics a wall-clock budget; it then fits the learning-rate
schedule to that time instead of a fixed number of epochs.  The colour augmentations are off (the image is
brightness = power over the noise floor, so changing brightness would change the SNR) while flips in time and
frequency stay on (a mirrored spectrogram has mirrored boxes and is still a valid scene).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data_cache/yolo/vuhf.yaml"))
    ap.add_argument("--model", default="yolov8n.pt", help="starting weights (COCO-pretrained by default)")
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget; overrides --epochs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", type=Path, default=Path("data_cache/yolo_runs"))
    ap.add_argument("--name", default="signal")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()), imgsz=args.imgsz, epochs=args.epochs, time=args.hours, batch=args.batch,
        device=args.device, workers=args.workers, project=str(args.project.resolve()), name=args.name, exist_ok=True,
        seed=args.seed, plots=False, cache="ram", amp=False,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5, flipud=0.5, mosaic=1.0, scale=0.3, translate=0.1, degrees=0.0,
        close_mosaic=3,
    )


if __name__ == "__main__":
    main()
