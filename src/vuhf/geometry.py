"""Time-frequency boxes and the mapping between physical units and spectrogram cells.

Conventions used everywhere in this package:

* Frequency is normalised to the sample rate: cycles/sample in [-0.5, 0.5).
  Multiply by the SDR sample rate to get Hz.
* Time is in samples from the start of the scene.
* A spectrogram ``P`` has shape ``(n_fft, n_frames)``: axis 0 is frequency
  (fft-shifted, so index ``n_fft // 2`` is 0 Hz), axis 1 is time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned time-frequency box. ``t1 > t0`` and ``f1 > f0``."""

    t0: float
    t1: float
    f0: float
    f1: float
    snr_db: float | None = None
    label: str | None = None

    @property
    def center_freq(self) -> float:
        return 0.5 * (self.f0 + self.f1)

    @property
    def bandwidth(self) -> float:
        return self.f1 - self.f0

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def area(self) -> float:
        return max(self.bandwidth, 0.0) * max(self.duration, 0.0)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes (0 if they do not overlap)."""
    dt = min(a.t1, b.t1) - max(a.t0, b.t0)
    df = min(a.f1, b.f1) - max(a.f0, b.f0)
    if dt <= 0 or df <= 0:
        return 0.0
    inter = dt * df
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class SpecGeometry:
    """Describes how a scene of ``n_samples`` I/Q samples becomes a spectrogram."""

    n_samples: int
    n_fft: int = 512
    hop: int = 256

    @property
    def n_frames(self) -> int:
        return 1 + (self.n_samples - self.n_fft) // self.hop

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_fft, self.n_frames)

    # Frame j is treated as covering the `hop` samples around its centre, so the
    # frames tile the time axis without overlap.
    @property
    def _t_offset(self) -> float:
        return (self.n_fft - self.hop) / 2.0

    # ---- cells -> physical -------------------------------------------------
    def cells_to_box(self, k0: int, k1: int, j0: int, j1: int, **kw) -> Box:
        """Inclusive cell ranges (frequency bins k0..k1, frames j0..j1) to a Box."""
        f0 = (k0 - 0.5) / self.n_fft - 0.5
        f1 = (k1 + 0.5) / self.n_fft - 0.5
        t0 = j0 * self.hop + self._t_offset
        t1 = (j1 + 1) * self.hop + self._t_offset
        return Box(t0=t0, t1=t1, f0=f0, f1=f1, **kw)

    # ---- physical -> cells -------------------------------------------------
    def box_to_cells(self, box: Box) -> tuple[int, int, int, int]:
        """Inclusive (k0, k1, j0, j1) of the cells a box overlaps, clipped to the grid."""
        n, h = self.n_fft, self.hop
        k0 = math.floor((box.f0 + 0.5) * n + 0.5)
        k1 = math.ceil((box.f1 + 0.5) * n + 0.5) - 1
        j0 = math.floor((box.t0 - self._t_offset) / h)
        j1 = math.ceil((box.t1 - self._t_offset) / h) - 1
        k0, k1 = max(k0, 0), min(k1, n - 1)
        j0, j1 = max(j0, 0), min(j1, self.n_frames - 1)
        return k0, k1, j0, j1
