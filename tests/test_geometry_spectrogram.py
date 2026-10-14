import numpy as np

from vuhf.detectors.calibration import noise_spectrogram
from vuhf.geometry import Box, SpecGeometry, iou
from vuhf.spectrogram import power_spectrogram


def test_iou_basic():
    a = Box(0, 10, 0.0, 0.1)
    assert iou(a, a) == 1.0
    assert iou(a, Box(20, 30, 0.0, 0.1)) == 0.0
    half = Box(5, 15, 0.0, 0.1)
    assert abs(iou(a, half) - (5 / 15)) < 1e-12


def test_box_cells_roundtrip_contains_box():
    geom = SpecGeometry(n_samples=65536, n_fft=256, hop=128)
    box = Box(t0=10000, t1=30000, f0=-0.2, f1=-0.05)
    k0, k1, j0, j1 = geom.box_to_cells(box)
    back = geom.cells_to_box(k0, k1, j0, j1)
    # the cell box covers the original box and overshoots by at most one cell
    assert back.f0 <= box.f0 + 1e-9 and back.f1 >= box.f1 - 1e-9
    assert back.t0 <= box.t0 + 1e-9 and back.t1 >= box.t1 - 1e-9
    assert box.f0 - back.f0 < 1.01 / geom.n_fft and back.f1 - box.f1 < 1.01 / geom.n_fft
    assert box.t0 - back.t0 < geom.hop + 1e-9 and back.t1 - box.t1 < geom.hop + 1e-9


def test_tone_appears_at_the_right_bin():
    # analogue of "a 100 kHz test tone appears exactly at 100 kHz"
    n, n_fft = 65536, 512
    geom = SpecGeometry(n_samples=n, n_fft=n_fft, hop=256)
    f_tone = 0.1  # cycles/sample
    iq = np.exp(2j * np.pi * f_tone * np.arange(n))
    P = power_spectrogram(iq, geom)
    assert P.shape == geom.shape
    peak_bin = int(np.argmax(P.mean(axis=1)))
    assert peak_bin == round((f_tone + 0.5) * n_fft)
    centre = (peak_bin - n_fft // 2) / n_fft
    assert abs(centre - f_tone) <= 0.5 / n_fft  # within half a bin


def test_noise_normalisation():
    geom = SpecGeometry(n_samples=131072, n_fft=512, hop=256)
    P = noise_spectrogram(geom, np.random.default_rng(1), variance=2.5)
    assert abs(P.mean() - 2.5) / 2.5 < 0.02
    # exponential cells: median = mean * ln 2
    assert abs(np.median(P) / np.log(2) - 2.5) / 2.5 < 0.03
