"""Scene sources: each yields ``Scene(iq, boxes, meta)`` for the detection baselines.

* ``synthetic_source``  quick noise-like bursts (no extra dependencies, for tests).
* ``torchsig_source``   WidebandSig53-style scenes generated with TorchSig 2.x
                        (needs ``pip install torchsig``; large because of PyTorch).

Boxes use the package convention: frequency in cycles/sample, time in samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..geometry import Box
from .synthetic import make_scene


@dataclass
class Scene:
    iq: np.ndarray
    boxes: list[Box]
    meta: dict = field(default_factory=dict)


def synthetic_source(n_scenes: int, seed: int = 0, **scene_kw) -> Iterator[Scene]:
    rng = np.random.default_rng(seed)
    for i in range(n_scenes):
        iq, boxes = make_scene(rng, **scene_kw)
        yield Scene(iq, boxes, {"source": "synthetic", "index": i})


def torchsig_source(
    n_scenes: int,
    seed: int = 0,
    *,
    impairment_level: int = 0,
    classes: list[str] | str = "all",
    n_samples: int = 262144,
    sample_rate: float = 10e6,
    num_signals: tuple[int, int] = (1, 4),
    snr_db: tuple[float, float] = (0.0, 20.0),
    bandwidth_hz: tuple[float, float] = (250e3, 2.5e6),
    duration_frac: tuple[float, float] = (0.2, 1.0),
) -> Iterator[Scene]:
    """Generate wideband scenes with several signals each, with ground-truth boxes.

    ``impairment_level`` follows TorchSig: 0 = perfect, 1 = cabled (transmitter
    impairments), 2 = wireless (transmitter impairments + fading).  Only the
    transmitter-side impairments are applied; TorchSig's ML augmentations
    (time reversal, I/Q swap, ...) are left out so the ground truth stays simple.

    ``snr_db`` is TorchSig's own definition: the peak of the signal's average
    PSD above the noise floor per FFT bin (see ``update_signal_snr_bandwidth``
    in torchsig/utils/dsp.py).  For a flat-spectrum signal this equals the
    in-band SNR; it is *not* the Es/N0 of the Sig53 paper.
    """
    # imported lazily so the rest of the package works without PyTorch
    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.transforms.impairments import Impairments
    from torchsig.utils.defaults import TorchSigDefaults

    md = TorchSigDefaults().default_dataset_metadata
    fft_size = 512
    md.update(
        num_iq_samples_dataset=n_samples,
        sample_rate=sample_rate,
        num_signals_min=num_signals[0],
        num_signals_max=num_signals[1],
        snr_db_min=snr_db[0],
        snr_db_max=snr_db[1],
        bandwidth_min=bandwidth_hz[0],
        bandwidth_max=bandwidth_hz[1],
        signal_duration_in_samples_min=duration_frac[0] * n_samples,
        signal_duration_in_samples_max=duration_frac[1] * n_samples,
        fft_size=fft_size,
        fft_stride=fft_size,
        frequency_min=-sample_rate / 2,
        frequency_max=sample_rate / 2 - 1,
        signal_center_freq_min=-sample_rate / 2 + bandwidth_hz[0] / 2,
        signal_center_freq_max=sample_rate / 2 - bandwidth_hz[0] / 2,
    )
    component_transforms = [Impairments(impairment_level).signal_transforms] if impairment_level > 0 else []
    ds = TorchSigIterableDataset(
        signal_generators=classes,
        metadata=md,
        component_transforms=component_transforms,
        target_labels=None,  # yield the full Signal object, with component metadata
        seed=seed,
    )
    for i in range(n_scenes):
        sig = next(ds)
        iq = np.asarray(sig.data)
        boxes = []
        for comp in sig.component_signals:
            t0 = float(comp.start_in_samples)
            t1 = float(comp.start_in_samples + comp.duration_in_samples)
            f0 = float(comp.lower_freq) / sample_rate
            f1 = float(comp.upper_freq) / sample_rate
            boxes.append(Box(t0=t0, t1=min(t1, n_samples), f0=max(f0, -0.5), f1=min(f1, 0.5),
                             snr_db=float(comp.snr_db), label=str(comp.class_name)))
        yield Scene(iq, boxes, {"source": "torchsig", "index": i, "impairment_level": impairment_level})


def cached(source_iter_factory, key: str, n_scenes: int, cache_dir) -> Iterator[Scene]:
    """Yield scenes from ``cache_dir`` if all ``n_scenes`` are stored under ``key``;
    otherwise generate them with ``source_iter_factory()`` and store them.

    Generating TorchSig scenes takes seconds each, so tuning and evaluation runs
    share one cached set (the cache directory is git-ignored)."""
    from pathlib import Path

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    files = [cache / f"{key}_{i:04d}.npz" for i in range(n_scenes)]
    if all(f.exists() for f in files):
        for f in files:
            z = np.load(f, allow_pickle=False)
            boxes = [Box(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), str(lab))
                     for r, lab in zip(z["boxes"], z["labels"])]
            yield Scene(z["iq"], boxes, {"source": key, "index": int(f.stem.rsplit("_", 1)[1])})
        return
    for f, scene in zip(files, source_iter_factory()):
        arr = np.array([[b.t0, b.t1, b.f0, b.f1, b.snr_db if b.snr_db is not None else np.nan] for b in scene.boxes]).reshape(-1, 5)
        np.savez(f, iq=scene.iq.astype(np.complex64), boxes=arr, labels=np.array([b.label or "" for b in scene.boxes], dtype=str))
        yield scene
