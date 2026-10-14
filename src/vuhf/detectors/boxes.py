"""Turn a detection mask into time-frequency boxes (one box per transmission)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry import Box, SpecGeometry


def mask_to_boxes(
    mask: np.ndarray,
    geom: SpecGeometry,
    *,
    link: tuple[int, int] = (2, 6),
    min_cells: int = 12,
    min_bins: int = 2,
    min_frames: int = 2,
) -> list[Box]:
    """Group detected cells into boxes.

    1. *Linking.*  Detected cells closer than ``link`` cells (frequency, time) are
       treated as the same transmission.  Weak signals produce sparse masks, so
       without linking they break into many small fragments.  Linking only decides
       *which cells belong together*; the box is then drawn tightly around the
       original cells, so linking does not fatten the box.
    2. *Connected components* (8-neighbourhood) of the linked mask, one per transmission.
    3. Components with fewer than ``min_cells`` detected cells, or thinner than
       ``min_bins`` x ``min_frames``, are discarded as false alarms.

    Touching transmissions merge into one box; separating them is the job of the AI
    detector or a later segmentation step.
    """
    if mask.shape != geom.shape:
        raise ValueError(f"mask shape {mask.shape} does not match geometry {geom.shape}")
    if not mask.any():
        return []
    linked = ndimage.maximum_filter(mask, size=(2 * link[0] + 1, 2 * link[1] + 1), mode="constant", cval=False)
    labels, n = ndimage.label(linked, structure=np.ones((3, 3), dtype=bool))
    own = np.where(mask, labels, 0)  # labels on the original detected cells only
    counts = np.bincount(own.ravel(), minlength=n + 1)
    boxes: list[Box] = []
    for lab, sl in enumerate(ndimage.find_objects(own), start=1):
        if sl is None or counts[lab] < min_cells:
            continue
        ks, js = sl
        k0, k1, j0, j1 = ks.start, ks.stop - 1, js.start, js.stop - 1
        if (k1 - k0 + 1) < min_bins or (j1 - j0 + 1) < min_frames:
            continue
        boxes.append(geom.cells_to_box(k0, k1, j0, j1))
    return boxes


def boxes_to_mask(boxes: list[Box], geom: SpecGeometry, margin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Rasterise boxes onto the spectrogram grid, optionally grown by ``margin`` cells."""
    out = np.zeros(geom.shape, dtype=bool)
    for b in boxes:
        k0, k1, j0, j1 = geom.box_to_cells(b)
        out[max(k0 - margin[0], 0) : k1 + margin[0] + 1, max(j0 - margin[1], 0) : j1 + margin[1] + 1] = True
    return out
