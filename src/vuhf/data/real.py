"""Readers for real I/Q recordings, so the detectors can be run on files from public datasets.

    rec = open_recording("capture.sigmf-meta")                 # SigMF: rate, centre frequency and annotations come from the metadata
    rec = open_recording("capture.cu8", fs=2.4e6, fc=150e6)    # raw RTL-SDR file: rate and centre frequency must be given
    for start, iq in iter_windows(rec, window=262144): ...      # complex64 windows of the recording

Supported: SigMF (``.sigmf-meta`` / ``.sigmf-data``; cf32_le, cf64_le, ci16_le, ci8, cu8), raw ``.cu8``/``.cs8``/``.cf32``,
SDRangel ``.sdriq`` (16 or 24 bit), 2-channel ``.wav`` (I = left, Q = right), MATLAB ``.mat`` (v5 through scipy, v7.3 through h5py).

Annotations (``Recording.annotations``) are time/frequency boxes in absolute units (samples from the start of the recording, Hz).
``window_boxes`` turns them into the package's ``Box`` convention (samples inside a window, cycles/sample relative to the centre
frequency), so the same scoring code as for the synthetic scenes can be used.

Every detector here estimates the noise level from the data itself, so the absolute scale of the samples does not matter.
"""
from __future__ import annotations

import json
import struct
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ..geometry import Box


@dataclass
class Annotation:
    sample_start: int
    sample_count: int
    f_low_hz: float          # absolute RF frequency
    f_high_hz: float
    label: str = ""


@dataclass
class Recording:
    path: Path
    fs: float                # sample rate, Hz
    fc: float                # centre frequency, Hz (0 if unknown)
    n_samples: int
    fmt: str
    annotations: list[Annotation] = field(default_factory=list)
    snippet_len: int | None = None   # set when the array holds independent snippets (rows); windows then never cross rows
    _read: object = None             # callable (start, count) -> complex64 array

    def read(self, start: int, count: int) -> np.ndarray:
        count = max(0, min(count, self.n_samples - start))
        return self._read(start, count).astype(np.complex64, copy=False)


# ---------------------------------------------------------------- raw interleaved files
_RAW = {  # name -> (numpy dtype of one scalar, offset, scale)
    "cu8": (np.uint8, -127.5, 1 / 127.5),
    "cs8": (np.int8, 0.0, 1 / 128.0),
    "ci8": (np.int8, 0.0, 1 / 128.0),
    "ci16_le": ("<i2", 0.0, 1 / 32768.0),
    "ci16_be": (">i2", 0.0, 1 / 32768.0),
    "cf32_le": ("<f4", 0.0, 1.0),
    "cf32_be": (">f4", 0.0, 1.0),
    "cf64_le": ("<f8", 0.0, 1.0),
    "cf32": ("<f4", 0.0, 1.0),
}


def _raw_reader(path: Path, kind: str, header_bytes: int = 0):
    dt, off, scale = _RAW[kind]
    itemsize = np.dtype(dt).itemsize * 2
    n = (path.stat().st_size - header_bytes) // itemsize

    def read(start, count):
        a = np.fromfile(path, dtype=dt, count=2 * count, offset=header_bytes + start * itemsize)
        a = (a.astype(np.float32) + off) * scale
        return a[0::2] + 1j * a[1::2]

    return n, read


def _array_reader(x: np.ndarray):
    x = np.asarray(x).astype(np.complex64)
    return len(x), (lambda start, count: x[start : start + count])


# ---------------------------------------------------------------- SigMF
def read_sigmf(path: Path, fs: float | None = None, fc: float | None = None) -> Recording:
    meta_path = path if path.suffix == ".sigmf-meta" else path.with_suffix(".sigmf-meta")
    data_path = meta_path.with_suffix(".sigmf-data")
    meta = json.loads(meta_path.read_text())
    g, caps = meta.get("global", {}), meta.get("captures", [])
    kind = g["core:datatype"]
    if kind not in _RAW:
        raise ValueError(f"SigMF datatype {kind!r} is not supported (supported: {sorted(_RAW)})")
    if len(caps) > 1:
        warnings.warn(f"{meta_path.name}: {len(caps)} capture segments; only the first is used for frequency and header offset")
    cap0 = caps[0] if caps else {}
    rate = float(fs or g.get("core:sample_rate", 0))
    if not rate:
        raise ValueError("no sample rate in the SigMF metadata; pass fs=")
    freq = float(fc if fc is not None else cap0.get("core:frequency", 0.0))
    n, read = _raw_reader(data_path, kind, int(cap0.get("core:header_bytes", 0)))
    ann = []
    for a in meta.get("annotations", []):
        if "core:freq_lower_edge" not in a or "core:freq_upper_edge" not in a:
            continue                                   # an annotation without a frequency extent is not a box
        ann.append(Annotation(int(a["core:sample_start"]), int(a.get("core:sample_count", n - a["core:sample_start"])),
                              float(a["core:freq_lower_edge"]), float(a["core:freq_upper_edge"]), str(a.get("core:label", ""))))
    return Recording(data_path, rate, freq, n, "sigmf:" + kind, ann, None, read)


# ---------------------------------------------------------------- SDRangel .sdriq
def read_sdriq(path: Path, fs: float | None = None, fc: float | None = None) -> Recording:
    """32-byte header: rate u32, centre frequency u64, start time u64 (ms), sample size u32 (16 or 24 bit), zeros u32, CRC32 u32
    of the first 28 bytes (SDRangel documentation, plugins/samplesource/fileinput).  Samples follow as interleaved I/Q: int16 for
    16 bit, int32 for 24 bit.  The scale of 24-bit files is a guess (2**23); the detectors do not depend on the scale."""
    with open(path, "rb") as f:
        head = f.read(32)
    rate, freq, t_ms, bits, _, crc = struct.unpack("<IQQIII", head)
    if zlib.crc32(head[:28]) & 0xFFFFFFFF != crc:
        warnings.warn(f"{path.name}: header CRC does not match; sample rate {rate} may be wrong (pass fs= to override)")
    if bits not in (16, 24):
        raise ValueError(f"unexpected sample size {bits} in {path.name}")
    dt, scale = ("<i2", 1 / 32768.0) if bits == 16 else ("<i4", 1 / 2**23)
    itemsize = np.dtype(dt).itemsize * 2
    n = (path.stat().st_size - 32) // itemsize

    def read(start, count):
        a = np.fromfile(path, dtype=dt, count=2 * count, offset=32 + start * itemsize).astype(np.float32) * scale
        return a[0::2] + 1j * a[1::2]

    return Recording(path, float(fs or rate), float(fc if fc is not None else freq), n, f"sdriq:{bits}bit", [], None, read)


# ---------------------------------------------------------------- WAV
def read_wav(path: Path, fs: float | None = None, fc: float | None = None) -> Recording:
    from scipy.io import wavfile

    rate, data = wavfile.read(path, mmap=True)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{path.name}: expected a 2-channel I/Q WAV file, got shape {data.shape}")
    kind = data.dtype
    scale, off = {np.dtype("int16"): (1 / 32768.0, 0.0), np.dtype("int32"): (1 / 2**31, 0.0), np.dtype("float32"): (1.0, 0.0),
                  np.dtype("uint8"): (1 / 128.0, -128.0)}.get(kind, (None, None))
    if scale is None:
        raise ValueError(f"unsupported WAV sample type {kind}")

    def read(start, count):
        a = (np.asarray(data[start : start + count], dtype=np.float32) + off) * scale
        return a[:, 0] + 1j * a[:, 1]

    return Recording(path, float(fs or rate), float(fc or 0.0), len(data), f"wav:{kind}", [], None, read)


# ---------------------------------------------------------------- MATLAB
def read_mat(path: Path, fs: float | None = None, fc: float | None = None, key: str | None = None) -> Recording:
    """Loads the whole array.  A 1-D array is one recording.  A 2-D array of many rows (for example 7,500 examples x N samples)
    is treated as independent snippets: ``snippet_len`` is set and windows never cross rows."""
    arr, meta = None, {}
    try:
        from scipy.io import loadmat

        d = loadmat(path, squeeze_me=True)
        cands = {k: v for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.size > 1}
        meta = {k: v for k, v in d.items() if not k.startswith("__") and np.ndim(v) == 0}
    except NotImplementedError:                                   # MATLAB v7.3 = HDF5
        import h5py

        cands = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.size > 1:
                    cands[name] = obj[()]
            f.visititems(visit)
        for k, v in list(cands.items()):                           # MATLAB complex data: compound dtype with real / imag
            if v.dtype.names and {"real", "imag"} <= set(v.dtype.names):
                cands[k] = v["real"] + 1j * v["imag"]
    if not cands:
        raise ValueError(f"{path.name}: no numeric array found")
    name = key or next((k for k in cands if k.lower() in ("iq_samples", "iq", "x", "samples", "data")), None)
    if name is None:
        name = max(cands, key=lambda k: np.iscomplexobj(cands[k]) * 10**12 + cands[k].size)
    arr = np.asarray(cands[name]).squeeze()
    if not np.iscomplexobj(arr):                                   # real array with a 2-column or 2-row I/Q layout
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr[:, 0] + 1j * arr[:, 1]
        elif arr.ndim == 2 and arr.shape[0] == 2:
            arr = arr[0] + 1j * arr[1]
        else:
            raise ValueError(f"{path.name}: array {name!r} {arr.shape} is real and not an I/Q pair; pass key=")
    rate = fs or next((float(meta[k]) for k in ("fs", "Fs", "sample_rate", "samplerate") if k in meta), None)
    if not rate:
        raise ValueError(f"{path.name}: no sample rate in the file; pass fs=")
    freq = float(fc if fc is not None else next((float(meta[k]) for k in ("fc", "Fc", "center_freq", "centre_freq") if k in meta), 0.0))
    if arr.ndim == 2:                                              # rows = independent snippets
        snippet = arr.shape[1]
        n, read = _array_reader(arr.reshape(-1))
        return Recording(path, float(rate), freq, n, f"mat:{name}{arr.shape}", [], snippet, read)
    n, read = _array_reader(arr)
    return Recording(path, float(rate), freq, n, f"mat:{name}", [], None, read)


# ---------------------------------------------------------------- entry points
def open_recording(path, fs: float | None = None, fc: float | None = None, fmt: str | None = None) -> Recording:
    path = Path(path)
    ext = (fmt or path.suffix.lstrip(".")).lower()
    if ext in ("sigmf-meta", "sigmf-data", "sigmf"):
        return read_sigmf(path, fs, fc)
    if ext == "sdriq":
        return read_sdriq(path, fs, fc)
    if ext == "wav":
        return read_wav(path, fs, fc)
    if ext == "mat":
        return read_mat(path, fs, fc)
    if ext in ("cu8", "cs8", "cf32", "ci16_le", "ci8"):
        if not fs:
            raise ValueError(f"{path.name}: raw files carry no metadata; pass fs= (and fc= if known)")
        n, read = _raw_reader(path, ext)
        return Recording(path, float(fs), float(fc or 0.0), n, ext, [], None, read)
    raise ValueError(f"unknown recording format {ext!r}")


def iter_windows(rec: Recording, window: int = 262144, hop: int | None = None, max_windows: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start_sample, iq)`` windows.  Only full windows are yielded, except that a recording (or snippet) shorter than
    ``window`` is returned whole.  For snippet arrays the window is the snippet length and never crosses a row."""
    if rec.snippet_len:
        window = min(window, rec.snippet_len)
        hop = hop or window
        starts = [r * rec.snippet_len + s for r in range(rec.n_samples // rec.snippet_len)
                  for s in range(0, rec.snippet_len - window + 1, hop)]
    elif rec.n_samples < window:
        starts, window = [0], rec.n_samples
    else:
        starts = list(range(0, rec.n_samples - window + 1, hop or window))
    for k, s in enumerate(starts):
        if max_windows is not None and k >= max_windows:
            return
        yield s, rec.read(s, window)


def window_boxes(rec: Recording, start: int, length: int, min_overlap: float = 0.5) -> list[Box]:
    """Annotations of ``rec`` as Boxes inside the window [start, start + length): time in samples from the window start,
    frequency in cycles/sample relative to the centre frequency.  An annotation is kept if at least ``min_overlap`` of its
    duration lies in the window and it overlaps the sampled band; it is clipped to the window and to +-0.5 cycles/sample."""
    out = []
    for a in rec.annotations:
        t0, t1 = max(a.sample_start, start), min(a.sample_start + a.sample_count, start + length)
        if t1 <= t0 or (t1 - t0) < min_overlap * a.sample_count:
            continue
        f0 = max((a.f_low_hz - rec.fc) / rec.fs, -0.5)
        f1 = min((a.f_high_hz - rec.fc) / rec.fs, 0.5)
        if f1 <= f0:
            continue
        out.append(Box(t0=float(t0 - start), t1=float(t1 - start), f0=float(f0), f1=float(f1), label=a.label or None))
    return out
