"""Small self-contained wideband scene generator (no TorchSig needed).

Signals are band-limited complex Gaussian noise bursts placed at random
frequencies and times.  They are *not* realistic modulations; they exist so the
detectors and the evaluation code can be tested quickly and deterministically.
The realistic benchmark is the TorchSig WidebandSig53 source (see ``torchsig_source``).

SNR convention (matches the "in-band SNR" of the report): signal power divided
by the noise power inside the signal's own bandwidth.  The noise is complex
white Gaussian with variance ``noise_var`` per sample, i.e. a power spectral
density of ``noise_var`` per unit normalised frequency, so the in-band noise
power of a signal of bandwidth ``B`` is ``noise_var * B``.
"""
from __future__ import annotations

import numpy as np

from ..geometry import Box


def _bandlimited_noise(n: int, bandwidth: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-power complex Gaussian noise confined to [-bandwidth/2, bandwidth/2]."""
    white = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    spec = np.fft.fft(white)
    f = np.fft.fftfreq(n)
    edge = max(bandwidth * 0.1, 2.0 / n)
    # flat passband with raised-cosine skirts of width `edge`
    h = np.clip((bandwidth / 2 + edge / 2 - np.abs(f)) / edge, 0.0, 1.0)
    h = 0.5 - 0.5 * np.cos(np.pi * h)
    sig = np.fft.ifft(spec * h)
    return sig / np.sqrt(np.mean(np.abs(sig) ** 2))


def make_scene(
    rng: np.random.Generator,
    *,
    n_samples: int = 131072,
    n_signals: int = 3,
    snr_db_range: tuple[float, float] = (0.0, 20.0),
    bw_range: tuple[float, float] = (0.01, 0.12),
    duration_frac_range: tuple[float, float] = (0.2, 0.9),
    noise_var: float = 1.0,
    max_tries: int = 200,
) -> tuple[np.ndarray, list[Box]]:
    """Return ``(iq, boxes)``: noise plus ``n_signals`` non-overlapping bursts."""
    iq = np.sqrt(noise_var / 2.0) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    boxes: list[Box] = []
    n = np.arange(n_samples)
    for _ in range(n_signals):
        for _ in range(max_tries):
            bw = rng.uniform(*bw_range)
            fc = rng.uniform(-0.5 + bw / 2, 0.5 - bw / 2)
            f0, f1 = fc - bw / 2, fc + bw / 2
            # keep a gap between signals so ground truth boxes stay separable
            if any(not (f1 + 0.02 < b.f0 or f0 - 0.02 > b.f1) for b in boxes):
                continue
            dur = int(rng.uniform(*duration_frac_range) * n_samples)
            t0 = int(rng.integers(0, n_samples - dur))
            snr_db = float(rng.uniform(*snr_db_range))
            power = 10 ** (snr_db / 10) * noise_var * bw
            burst = _bandlimited_noise(dur, bw, rng) * np.sqrt(power)
            burst *= np.exp(2j * np.pi * fc * n[:dur])
            iq[t0 : t0 + dur] += burst
            boxes.append(Box(t0=float(t0), t1=float(t0 + dur), f0=f0, f1=f1, snr_db=snr_db, label="noise-like"))
            break
    return iq, boxes
