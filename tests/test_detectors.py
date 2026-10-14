import numpy as np
import pytest

from vuhf.data.synthetic import make_scene
from vuhf.detectors import (
    ca_cfar_2d,
    calibrate_multiplier,
    cfar_statistic,
    energy_detector,
    energy_statistic,
    mask_to_boxes,
    noise_spectrogram,
)
from vuhf.detectors.cfar import n_training_cells, theoretical_alpha
from vuhf.eval.matching import evaluate_scene
from vuhf.geometry import SpecGeometry
from vuhf.spectrogram import power_spectrogram

GEOM = SpecGeometry(n_samples=131072, n_fft=512, hop=256)
PFA = 1e-3
TRAIN, GUARD = (8, 4), (2, 1)


def test_cfar_alpha_formula():
    # N = 4 training cells, pfa = 1e-3  ->  alpha = N (pfa^(-1/N) - 1)
    assert abs(theoretical_alpha(1e-3, 4) - 4 * (1e-3 ** (-0.25) - 1)) < 1e-12
    assert n_training_cells((8, 4), (2, 1)) == (2 * 10 + 1) * (2 * 5 + 1) - (2 * 2 + 1) * (2 * 1 + 1)


def test_cfar_threshold_follows_the_noise_floor():
    # constant false alarm rate: scaling the noise by 1000 must not change the detections
    rng = np.random.default_rng(0)
    P = noise_spectrogram(GEOM, rng)
    m1, _ = ca_cfar_2d(P, train=TRAIN, guard=GUARD, pfa=PFA)
    m2, _ = ca_cfar_2d(P * 1000.0, train=TRAIN, guard=GUARD, pfa=PFA)
    assert np.array_equal(m1, m2)


def test_cfar_false_alarm_rate_after_calibration():
    stat = lambda P: cfar_statistic(P, TRAIN, GUARD)
    alpha = calibrate_multiplier(stat, GEOM, PFA, n_trials=12, seed=10)
    # measure on independent noise
    rng = np.random.default_rng(99)
    rates = [ca_cfar_2d(noise_spectrogram(GEOM, rng), train=TRAIN, guard=GUARD, alpha=alpha)[0].mean() for _ in range(12)]
    assert 0.6 * PFA < np.mean(rates) < 1.6 * PFA


def test_energy_false_alarm_rate_after_calibration():
    stat = lambda P: energy_statistic(P, (1, 3), "global")
    mult = calibrate_multiplier(stat, GEOM, PFA, n_trials=12, seed=11)
    rng = np.random.default_rng(98)
    rates = [energy_detector(noise_spectrogram(GEOM, rng), avg=(1, 3), multiplier=mult)[0].mean() for _ in range(12)]
    assert 0.6 * PFA < np.mean(rates) < 1.6 * PFA


def _run_energy(P, seed=1):
    mult = calibrate_multiplier(lambda x: energy_statistic(x, (1, 3), "global"), GEOM, PFA, seed=seed)
    return energy_detector(P, avg=(1, 3), multiplier=mult)[0]


def _run_cfar_1d(P, train=(24, 0), guard=(8, 0), seed=1):
    # frequency-only CFAR: each time frame is scanned along frequency
    alpha = calibrate_multiplier(lambda x: cfar_statistic(x, train, guard, (1, 3)), GEOM, PFA, seed=seed)
    return ca_cfar_2d(P, train=train, guard=guard, avg=(1, 3), alpha=alpha)[0]


@pytest.mark.parametrize("detector", ["energy", "cfar_1d"])
def test_detects_strong_narrow_signals_in_a_synthetic_scene(detector):
    rng = np.random.default_rng(5)
    iq, gt = make_scene(rng, n_samples=GEOM.n_samples, n_signals=3, snr_db_range=(12.0, 20.0), bw_range=(0.01, 0.03))
    P = power_spectrogram(iq, GEOM)
    mask = _run_energy(P) if detector == "energy" else _run_cfar_1d(P)
    res = evaluate_scene(mask_to_boxes(mask, GEOM), gt, iou_thr=0.5)
    assert all(g.detected for g in res.gt), [g for g in res.gt if not g.detected]
    assert res.n_false_alarms <= 1


def test_ca_cfar_self_masking_on_a_wide_signal():
    """Documented CA-CFAR weakness: a signal wider than the guard region fills its own
    training cells, so the threshold rises and the middle of the signal is missed.
    A global-noise-floor energy detector does not have this problem."""
    from vuhf.geometry import Box

    rng = np.random.default_rng(7)
    iq, gt = make_scene(rng, n_samples=GEOM.n_samples, n_signals=1, snr_db_range=(15.0, 15.0), bw_range=(0.2, 0.2))
    P = power_spectrogram(iq, GEOM)
    box = gt[0]
    k0, k1, j0, j1 = GEOM.box_to_cells(Box(box.t0, box.t1, box.f0 + 0.04, box.f1 - 0.04))  # interior only
    interior = (slice(k0, k1 + 1), slice(j0, j1 + 1))
    frac_energy = _run_energy(P)[interior].mean()
    frac_cfar = _run_cfar_1d(P)[interior].mean()
    assert frac_energy > 0.9
    assert frac_cfar < 0.5


def test_linking_merges_fragments_but_keeps_the_box_tight():
    geom = SpecGeometry(n_samples=131072, n_fft=512, hop=256)
    mask = np.zeros(geom.shape, dtype=bool)
    # one transmission whose cells came out as 3 separate pieces (gaps of 3 cells in time)
    mask[200:240, 100:130] = True
    mask[200:240, 134:170] = True
    mask[200:240, 174:200] = True
    assert len(mask_to_boxes(mask, geom, link=(0, 0), min_cells=12)) == 3
    boxes = mask_to_boxes(mask, geom, link=(2, 6), min_cells=12)
    assert len(boxes) == 1
    k0, k1, j0, j1 = geom.box_to_cells(boxes[0])
    assert (k0, k1, j0, j1) == (200, 239, 100, 199)  # tight around the original cells


def test_isolated_false_alarm_cells_are_dropped():
    geom = SpecGeometry(n_samples=131072, n_fft=512, hop=256)
    mask = np.zeros(geom.shape, dtype=bool)
    mask[50, 50] = mask[300, 400] = mask[301, 401] = True
    assert mask_to_boxes(mask, geom, min_cells=12) == []


def test_fragment_vs_false_alarm_accounting():
    from vuhf.geometry import Box

    gt = [Box(1000, 5000, -0.2, -0.1, snr_db=10.0)]
    pieces = [Box(1000, 2000, -0.2, -0.1), Box(3000, 4000, -0.2, -0.1)]   # each IoU 0.25 < 0.5 -> unmatched pieces
    stray = [Box(1000, 2000, 0.3, 0.4)]                                  # nowhere near any signal
    res = evaluate_scene(pieces + stray, gt, iou_thr=0.5)
    assert not res.gt[0].detected
    assert res.n_fragments == 2 and res.n_false_alarms == 1
    res2 = evaluate_scene([Box(1000, 5000, -0.2, -0.1)], gt, iou_thr=0.5)
    assert res2.gt[0].detected and res2.n_false_alarms == 0 and res2.n_fragments == 0
