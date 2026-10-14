"""I/Q samples -> power spectrogram."""
from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from .geometry import SpecGeometry


def power_spectrogram(iq: np.ndarray, geom: SpecGeometry) -> np.ndarray:
    """Two-sided power spectrogram, shape ``(n_fft, n_frames)``.

    Scaled so that complex white noise of variance ``s2`` gives a mean cell
    value of ``s2`` (in other words ``P`` is in units of noise power per bin).
    Individual noise-only cells are then exponentially distributed, which is
    what the energy detector and CFAR threshold formulas assume.
    """
    iq = np.asarray(iq)
    if iq.shape[0] != geom.n_samples:
        raise ValueError(f"expected {geom.n_samples} samples, got {iq.shape[0]}")
    win = get_window("hann", geom.n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(iq, geom.n_fft)[:: geom.hop]
    spec = np.fft.fft(frames[: geom.n_frames] * win, axis=1)
    power = (spec.real**2 + spec.imag**2) / np.sum(win**2)
    return np.fft.fftshift(power, axes=1).T.copy()
