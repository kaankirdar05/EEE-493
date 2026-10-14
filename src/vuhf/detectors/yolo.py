"""YOLO detector on spectrogram images (single class "signal").

The spectrogram ``P`` (n_fft x n_frames, noise mean 1) becomes a grayscale image:

* rows are frequency bins (row 0 = lowest frequency), columns are time, two neighbouring frames
  averaged into one column (halves the noise variance and makes the image n_fft x n_fft);
* pixel value = power over the noise floor in dB, clipped to ``IMG_DB`` and scaled to 0..255.
  The noise floor is the scene median / ln 2, the same estimate the energy detector uses.

Boxes go back and forth between this pixel grid and the package's physical units (cycles/sample,
samples) with the same cell edges as ``SpecGeometry``, so a YOLO box is scored exactly like a box
from the energy detector or CFAR.

Ultralytics is imported lazily (``pip install ultralytics``).  Note that Ultralytics YOLO is
AGPL-3.0 licensed; for a module İLTAREN wants to embed, a permissive detector would be needed.
"""
from __future__ import annotations

import numpy as np

from ..geometry import Box, SpecGeometry

IMG_DB = (-5.0, 25.0)


def image_width(geom: SpecGeometry) -> int:
    return (geom.n_frames + 1) // 2


def spectrogram_image(P: np.ndarray) -> np.ndarray:
    """uint8 image, shape (n_fft, ceil(n_frames / 2))."""
    if P.shape[1] % 2:
        P = np.concatenate([P, P[:, -1:]], axis=1)
    Pm = 0.5 * (P[:, 0::2] + P[:, 1::2])
    noise = np.median(P) / np.log(2)
    db = 10.0 * np.log10(np.maximum(Pm / noise, 1e-6))
    lo, hi = IMG_DB
    return np.clip((db - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def box_to_pixels(box: Box, geom: SpecGeometry) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) in image pixels, clipped to the image; x = time, y = frequency."""
    n, w = geom.n_fft, image_width(geom)
    y0 = (box.f0 + 0.5) * n + 0.5
    y1 = (box.f1 + 0.5) * n + 0.5
    x0 = (box.t0 - geom._t_offset) / geom.hop / 2.0
    x1 = (box.t1 - geom._t_offset) / geom.hop / 2.0
    return max(x0, 0.0), max(y0, 0.0), min(x1, float(w)), min(y1, float(n))


def pixels_to_box(x0: float, y0: float, x1: float, y1: float, geom: SpecGeometry, **kw) -> Box:
    n = geom.n_fft
    return Box(t0=x0 * 2.0 * geom.hop + geom._t_offset, t1=x1 * 2.0 * geom.hop + geom._t_offset,
               f0=(y0 - 0.5) / n - 0.5, f1=(y1 - 0.5) / n - 0.5, **kw)


def yolo_label_lines(boxes: list[Box], geom: SpecGeometry) -> list[str]:
    """One ``0 xc yc w h`` line (normalised) per ground-truth box; boxes thinner than a pixel are skipped."""
    n, w = geom.n_fft, image_width(geom)
    lines = []
    for b in boxes:
        x0, y0, x1, y1 = box_to_pixels(b, geom)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            continue
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / n:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / n:.6f}")
    return lines


class YoloDetector:
    """Wraps a trained Ultralytics model.  ``predict_raw`` keeps every box above ``min_conf`` with its
    confidence, so the confidence threshold can be chosen afterwards on validation scenes."""

    def __init__(self, weights, geom: SpecGeometry, *, min_conf: float = 0.01, nms_iou: float = 0.7,
                 max_det: int = 50, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.geom, self.min_conf, self.nms_iou, self.max_det, self.device = geom, min_conf, nms_iou, max_det, device
        self.imgsz = max(geom.n_fft, image_width(geom))

    def predict_raw(self, Ps: list[np.ndarray]) -> list[list[tuple[Box, float]]]:
        imgs = [np.repeat(spectrogram_image(P)[:, :, None], 3, axis=2) for P in Ps]
        results = self.model.predict(imgs, imgsz=self.imgsz, conf=self.min_conf, iou=self.nms_iou,
                                     max_det=self.max_det, device=self.device, verbose=False)
        out = []
        for r in results:
            xyxy, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            out.append([(pixels_to_box(*row, self.geom), float(c)) for row, c in zip(xyxy, conf)])
        return out

    def predict(self, P: np.ndarray, conf: float) -> list[Box]:
        return [b for b, c in self.predict_raw([P])[0] if c >= conf]
