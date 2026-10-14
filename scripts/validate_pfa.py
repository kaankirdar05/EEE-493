"""Check that the detectors hold their design false-alarm rate on pure noise.

For each detector and each target Pfa this prints three numbers:

* theory      - measured cell false-alarm rate with the closed-form threshold
* calibrated  - measured rate with the threshold calibrated on separate noise
* target      - the Pfa we asked for

The calibrated threshold should land on the target.  The closed form is a bit
optimistic because neighbouring spectrogram cells are correlated.

    python scripts/validate_pfa.py --n-samples 262144
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vuhf.detectors import calibrate_multiplier, ca_cfar_2d, cfar_statistic, energy_detector, energy_statistic, noise_spectrogram
from vuhf.detectors.cfar import n_training_cells, theoretical_alpha
from vuhf.detectors.energy import theoretical_multiplier
from vuhf.geometry import SpecGeometry

CONFIGS = {
    "energy_global_avg1x3": dict(kind="energy", avg=(1, 3), floor="global"),
    "energy_per_bin_avg1x3": dict(kind="energy", avg=(1, 3), floor="per_bin"),
    "cfar_freq_only_t24_g8": dict(kind="cfar", train=(24, 0), guard=(8, 0), avg=(1, 1)),
    "cfar_2d_t8x4_g2x1": dict(kind="cfar", train=(8, 4), guard=(2, 1), avg=(1, 1)),
}


def measure(cfg, geom, rng, n_scenes, *, multiplier=None):
    """Cell false-alarm rate on fresh noise-only spectrograms."""
    hits = total = 0
    for _ in range(n_scenes):
        P = noise_spectrogram(geom, rng)
        if cfg["kind"] == "energy":
            mask, _ = energy_detector(P, avg=cfg["avg"], floor=cfg["floor"], multiplier=multiplier)
        else:
            mask, _ = ca_cfar_2d(P, train=cfg["train"], guard=cfg["guard"], avg=cfg["avg"], alpha=multiplier)
        hits += int(mask.sum())
        total += mask.size
    return hits / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-samples", type=int, default=262144)
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--pfa", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--n-scenes", type=int, default=24, help="noise-only spectrograms per measurement")
    ap.add_argument("--out", type=Path, default=Path("results/pfa_validation.json"))
    args = ap.parse_args()

    geom = SpecGeometry(args.n_samples, args.n_fft, args.hop)
    rows = []
    print(f"spectrogram {geom.shape}, {args.n_scenes} noise-only scenes per measurement\n")
    print(f"{'detector':<24}{'target':>9}{'theory':>11}{'calibrated':>12}")
    for name, cfg in CONFIGS.items():
        for pfa in args.pfa:
            if cfg["kind"] == "energy":
                th = theoretical_multiplier(pfa, cfg["avg"][0] * cfg["avg"][1])
                stat = lambda P, c=cfg: energy_statistic(P, c["avg"], c["floor"])
            else:
                th = theoretical_alpha(pfa, n_training_cells(cfg["train"], cfg["guard"]))
                stat = lambda P, c=cfg: cfar_statistic(P, c["train"], c["guard"], c["avg"])
            cal = calibrate_multiplier(stat, geom, pfa, n_trials=12, seed=123)
            m_theory = measure(cfg, geom, np.random.default_rng(1), args.n_scenes, multiplier=th)
            m_cal = measure(cfg, geom, np.random.default_rng(2), args.n_scenes, multiplier=cal)
            print(f"{name:<24}{pfa:>9.0e}{m_theory:>11.2e}{m_cal:>12.2e}")
            rows.append(dict(detector=name, target_pfa=pfa, threshold_theory=th, threshold_calibrated=cal,
                             measured_pfa_theory=m_theory, measured_pfa_calibrated=m_cal))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(geometry=vars(args) | {"out": str(args.out)}, rows=rows), indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
