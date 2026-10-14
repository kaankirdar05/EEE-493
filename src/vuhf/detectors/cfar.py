"""Cell-averaging CFAR (CA-CFAR) on a power spectrogram.

Idea (Finn and Johnson, 1968; Rohling, 1983): instead of one global threshold,
test every cell (the *cell under test*, CUT) against the average noise level
of its own neighbourhood.  The neighbourhood is a rectangle in frequency and
time with two parts:

* *guard cells* right around the CUT, ignored so that the signal's own energy
  (which leaks into adjacent cells) does not raise the noise estimate;
* *training cells* in the ring outside the guard region, averaged to estimate
  the local noise power ``mu``.

A cell is declared a signal if ``P > alpha * mu``.  With ``N`` training cells and
exponentially distributed noise cells, choosing ``alpha = N (pfa^(-1/N) - 1)``
gives a false-alarm probability of exactly ``pfa`` *whatever the noise level*,
hence "constant false alarm rate".  That is the advantage over a fixed
threshold: if the noise floor rises or varies across the band, the threshold
follows it.

Known weakness: a signal wider than the training window fills its own training
cells, so the threshold rises and the middle of the signal is missed (only its
edges are detected).  Make the window larger than the widest signal, or use
ordered-statistic CFAR (OS-CFAR) which replaces the mean by a percentile.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def n_training_cells(train: tuple[int, int], guard: tuple[int, int]) -> int:
    (tf, tt), (gf, gt) = train, guard
    outer = (2 * (gf + tf) + 1) * (2 * (gt + tt) + 1)
    inner = (2 * gf + 1) * (2 * gt + 1)
    return outer - inner


def theoretical_alpha(pfa: float, n_train: int) -> float:
    """CA-CFAR scale factor for exponential (square-law, noise-only) cells."""
    return float(n_train * (pfa ** (-1.0 / n_train) - 1.0))


def cfar_statistic(
    P: np.ndarray,
    train: tuple[int, int] = (8, 4),
    guard: tuple[int, int] = (2, 1),
    avg: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Cell power divided by the local training-cell average (≈ noise-only ratio)."""
    (tf, tt), (gf, gt) = train, guard
    outer = (2 * (gf + tf) + 1, 2 * (gt + tt) + 1)
    inner = (2 * gf + 1, 2 * gt + 1)
    n_train = n_training_cells(train, guard)
    cut = uniform_filter(P, size=avg, mode="reflect") if avg != (1, 1) else P
    sum_out = uniform_filter(P, size=outer, mode="reflect") * (outer[0] * outer[1])
    sum_in = uniform_filter(P, size=inner, mode="reflect") * (inner[0] * inner[1])
    noise = np.maximum((sum_out - sum_in) / n_train, np.finfo(float).tiny)
    return cut / noise


def ca_cfar_2d(
    P: np.ndarray,
    *,
    train: tuple[int, int] = (8, 4),
    guard: tuple[int, int] = (2, 1),
    pfa: float = 1e-3,
    avg: tuple[int, int] = (1, 1),
    alpha: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Return ``(mask, info)``; ``mask`` is True where a signal is declared.

    ``train`` and ``guard`` are half-sizes in cells as ``(frequency, time)``.
    Setting the time entries to 0 gives the classic one-dimensional CFAR that
    scans along frequency within each time frame.  ``alpha`` overrides the
    theoretical scale factor (use ``calibrate_multiplier`` to obtain one that
    holds the measured false-alarm rate when cells are correlated or ``avg`` is
    used).
    """
    n_train = n_training_cells(train, guard)
    a = theoretical_alpha(pfa, n_train) if alpha is None else float(alpha)
    stat = cfar_statistic(P, train, guard, avg)
    return stat > a, {"alpha": a, "n_train": n_train, "train": train, "guard": guard}
