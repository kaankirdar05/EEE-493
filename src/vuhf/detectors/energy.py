"""Energy detector on a power spectrogram.

Idea (Urkowitz, 1967): a cell contains a signal if its energy is clearly above
what noise alone would produce.  Three ingredients:

1. *Noise floor estimate.*  Signals occupy a minority of the time-frequency
   plane, so the median of the spectrogram is dominated by noise.  For
   exponentially distributed noise cells, ``mean = median / ln 2``.
2. *Averaging.*  Averaging ``K`` neighbouring cells shrinks the noise
   fluctuations by ``sqrt(K)``, which lets weaker signals cross the threshold.
3. *Threshold.*  Chosen from a target false-alarm probability ``pfa``.  For
   ``K`` independent exponential cells the averaged noise is Gamma(K)
   distributed; the closed form below is a good first guess.  Neighbouring
   cells are correlated (window overlap), so use ``calibrate_multiplier`` to
   set the threshold from a noise-only simulation when the exact Pfa matters.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import gamma

def noise_mean_estimate(P: np.ndarray, floor: str = "global", q: float = 0.5) -> np.ndarray | float:
    """Robust noise-power estimate from a low quantile of the spectrogram.

    For exponentially distributed noise cells the ``q`` quantile is
    ``mean * -ln(1 - q)``, so ``mean = quantile / -ln(1 - q)``.  ``q=0.5`` is
    the median; a smaller ``q`` (e.g. 0.25) tolerates signals that occupy more
    than half of the cells, at the cost of a noisier estimate.

    ``floor="global"``  one value for the whole spectrogram.
    ``floor="per_bin"`` one value per frequency bin (quantile over time), which
    follows a non-flat noise floor such as receiver roll-off but breaks when a
    signal stays on in a bin for more than ``1 - q`` of the time.
    """
    factor = -np.log(1.0 - q)
    if floor == "global":
        return float(np.quantile(P, q) / factor)
    if floor == "per_bin":
        return np.quantile(P, q, axis=1, keepdims=True) / factor
    raise ValueError(f"unknown floor mode {floor!r}")


def energy_statistic(P: np.ndarray, avg: tuple[int, int] = (1, 3), floor: str = "global", q: float = 0.5) -> np.ndarray:
    """Averaged cell power divided by the estimated noise power (≈1 for noise)."""
    p_avg = uniform_filter(P, size=avg, mode="nearest") if avg != (1, 1) else P
    return p_avg / noise_mean_estimate(P, floor, q)


def theoretical_multiplier(pfa: float, n_avg: int) -> float:
    """Threshold / noise-mean for ``n_avg`` independent exponential cells."""
    return float(gamma.isf(pfa, a=n_avg, scale=1.0 / n_avg))


def energy_detector(
    P: np.ndarray,
    *,
    avg: tuple[int, int] = (1, 3),
    floor: str = "global",
    q: float = 0.5,
    pfa: float = 1e-3,
    margin_db: float | None = None,
    multiplier: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Return ``(mask, info)``; ``mask`` is True where a signal is declared.

    Threshold selection, in priority order: ``margin_db`` (fixed margin above
    the noise floor, e.g. 6 dB as in the project plan), an explicit
    ``multiplier`` (for example from ``calibrate_multiplier``), otherwise the
    theoretical multiplier for ``pfa``.
    """
    stat = energy_statistic(P, avg, floor, q)
    if margin_db is not None:
        mult = 10.0 ** (margin_db / 10.0)
    elif multiplier is not None:
        mult = float(multiplier)
    else:
        mult = theoretical_multiplier(pfa, avg[0] * avg[1])
    return stat > mult, {"multiplier": mult, "avg": avg, "floor": floor, "q": q}
