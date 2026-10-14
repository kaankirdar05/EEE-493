"""Set detector thresholds from a noise-only simulation.

The closed-form thresholds assume independent exponential cells.  A Hann-windowed,
overlapped STFT makes neighbouring cells correlated, so the real false-alarm
rate is somewhat higher than designed.  Simulating white noise through the same
spectrogram geometry and taking the ``1 - pfa`` quantile of the detection
statistic gives a threshold that holds the target rate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..geometry import SpecGeometry
from ..spectrogram import power_spectrogram


def noise_spectrogram(geom: SpecGeometry, rng: np.random.Generator, variance: float = 1.0) -> np.ndarray:
    """Spectrogram of complex white Gaussian noise (no signals)."""
    iq = np.sqrt(variance / 2.0) * (rng.standard_normal(geom.n_samples) + 1j * rng.standard_normal(geom.n_samples))
    return power_spectrogram(iq, geom)


def calibrate_multiplier(
    statistic: Callable[[np.ndarray], np.ndarray],
    geom: SpecGeometry,
    pfa: float,
    n_trials: int = 8,
    seed: int = 0,
) -> float:
    """``1 - pfa`` quantile of ``statistic(P)`` over noise-only spectrograms."""
    rng = np.random.default_rng(seed)
    values = np.concatenate([statistic(noise_spectrogram(geom, rng)).ravel() for _ in range(n_trials)])
    return float(np.quantile(values, 1.0 - pfa))
