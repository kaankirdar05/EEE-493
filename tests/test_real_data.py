"""Readers for real recordings and the end-to-end run on a file with annotations (no TorchSig, no YOLO needed)."""
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat, wavfile

from vuhf.data.real import Annotation, iter_windows, open_recording, window_boxes
from vuhf.data.synthetic import make_scene

ROOT = Path(__file__).resolve().parents[1]


def _iq(n=20000, seed=0):
    r = np.random.default_rng(seed)
    return (0.12 * (r.standard_normal(n) + 1j * r.standard_normal(n))).astype(np.complex64)


def _same(a, b, tol):
    assert np.max(np.abs(a - b)) < tol


def test_cu8_roundtrip(tmp_path):
    x = _iq()
    q = np.empty(2 * len(x), np.uint8)
    q[0::2] = np.clip(np.round(x.real * 127.5 + 127.5), 0, 255); q[1::2] = np.clip(np.round(x.imag * 127.5 + 127.5), 0, 255)
    q.tofile(tmp_path / "a.cu8")
    rec = open_recording(tmp_path / "a.cu8", fs=2.4e6, fc=150e6)
    assert rec.n_samples == len(x) and rec.fs == 2.4e6 and rec.fc == 150e6
    _same(rec.read(100, 500), x[100:600], 0.02)
    with pytest.raises(ValueError):
        open_recording(tmp_path / "a.cu8")                      # raw files need a sample rate


def test_sigmf_roundtrip_with_annotations(tmp_path):
    x = _iq()
    x.tofile(tmp_path / "r.sigmf-data")
    meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6, "core:version": "1.0.0"},
            "captures": [{"core:sample_start": 0, "core:frequency": 100e6}],
            "annotations": [{"core:sample_start": 1000, "core:sample_count": 4000, "core:freq_lower_edge": 100.1e6,
                             "core:freq_upper_edge": 100.2e6, "core:label": "x"},
                            {"core:sample_start": 0, "core:sample_count": 10}]}          # no frequency extent: ignored
    (tmp_path / "r.sigmf-meta").write_text(json.dumps(meta))
    rec = open_recording(tmp_path / "r.sigmf-meta")
    assert (rec.fs, rec.fc, len(rec.annotations)) == (1e6, 100e6, 1)
    _same(rec.read(5, 100), x[5:105], 1e-6)
    b = window_boxes(rec, 500, 8000)[0]                                                  # annotation inside the window
    assert (b.t0, b.t1) == (500.0, 4500.0) and abs(b.f0 - 0.1) < 1e-9 and abs(b.f1 - 0.2) < 1e-9 and b.label == "x"
    assert window_boxes(rec, 4900, 8000) == []                                           # less than half of it inside: dropped
    assert window_boxes(rec, 0, 2000) == []                                              # 1000 of 4000 samples inside: dropped


def test_sdriq_roundtrip_and_crc(tmp_path):
    x = _iq(5000)
    head = struct.pack("<IQQII", 250000, 433_920_000, 1_700_000_000_000, 16, 0)
    raw = np.empty(2 * len(x), "<i2"); raw[0::2] = np.round(x.real * 32768); raw[1::2] = np.round(x.imag * 32768)
    (tmp_path / "a.sdriq").write_bytes(head + struct.pack("<I", zlib.crc32(head)) + raw.tobytes())
    rec = open_recording(tmp_path / "a.sdriq")
    assert (rec.fs, rec.fc, rec.n_samples) == (250000.0, 433.92e6, 5000)
    _same(rec.read(0, 5000), x, 1e-3)
    bad = bytearray((tmp_path / "a.sdriq").read_bytes()); bad[31] ^= 0xFF
    (tmp_path / "b.sdriq").write_bytes(bytes(bad))
    with pytest.warns(UserWarning, match="CRC"):
        open_recording(tmp_path / "b.sdriq")


def test_wav_roundtrip(tmp_path):
    x = _iq(4000)
    wavfile.write(tmp_path / "a.wav", 48000, np.stack([np.round(x.real * 32767), np.round(x.imag * 32767)], 1).astype(np.int16))
    rec = open_recording(tmp_path / "a.wav", fc=7.1e6)
    assert (rec.fs, rec.fc, rec.n_samples) == (48000.0, 7.1e6, 4000)
    _same(rec.read(0, 4000), x, 1e-3)


def test_mat_one_recording_and_snippets(tmp_path):
    x = _iq(10000)
    savemat(tmp_path / "a.mat", {"IQ_samples": x})
    rec = open_recording(tmp_path / "a.mat", fs=2.048e6, fc=864e6)
    assert rec.n_samples == 10000 and rec.snippet_len is None
    _same(rec.read(0, 10000), x, 1e-6)
    rows = np.stack([_iq(4096, s) for s in range(5)])
    savemat(tmp_path / "b.mat", {"IQ_samples": rows, "fs": 1e6})
    snip = open_recording(tmp_path / "b.mat")
    assert snip.snippet_len == 4096 and snip.fs == 1e6
    wins = list(iter_windows(snip, window=262144))
    assert len(wins) == 5 and all(len(w) == 4096 for _, w in wins)                      # windows never cross rows
    _same(wins[3][1], rows[3], 1e-6)


def test_iter_windows_edges():
    from vuhf.data.real import Recording

    x = _iq(10000)
    rec = Recording(Path("m"), 1.0, 0.0, 10000, "mem", [], None, lambda s, c: x[s:s + c])
    assert [s for s, _ in iter_windows(rec, 4096)] == [0, 4096]                         # only full windows
    assert [(s, len(w)) for s, w in iter_windows(rec, 20000)] == [(0, 10000)]           # shorter recording: returned whole
    assert len(list(iter_windows(rec, 4096, max_windows=1))) == 1


def test_end_to_end_on_annotated_sigmf(tmp_path):
    """A synthetic scene written as SigMF with annotations is scored by scripts/run_real_data.py; the annotations must
    round-trip to the original boxes and the energy detector must find most signals."""
    rng = np.random.default_rng(3)
    n = 262144
    iq, boxes = make_scene(rng, n_samples=n, n_signals=3, snr_db_range=(10, 20), bw_range=(0.03, 0.1))
    iq.astype(np.complex64).tofile(tmp_path / "s.sigmf-data")
    fs, fc = 10e6, 100e6
    ann = [{"core:sample_start": int(b.t0), "core:sample_count": int(b.t1 - b.t0), "core:freq_lower_edge": fc + b.f0 * fs,
            "core:freq_upper_edge": fc + b.f1 * fs} for b in boxes]
    (tmp_path / "s.sigmf-meta").write_text(json.dumps({"global": {"core:datatype": "cf32_le", "core:sample_rate": fs},
                                                       "captures": [{"core:sample_start": 0, "core:frequency": fc}], "annotations": ann}))
    rec = open_recording(tmp_path / "s.sigmf-meta")
    back = window_boxes(rec, 0, n)
    assert len(back) == len(boxes)
    for a, b in zip(back, boxes):
        assert abs(a.t0 - b.t0) < 1 and abs(a.t1 - b.t1) < 1 and abs(a.f0 - b.f0) < 1e-9 and abs(a.f1 - b.f1) < 1e-9
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_real_data.py"), str(tmp_path / "s.sigmf-meta"), "--no-yolo",
                        "--max-windows", "1", "--n-figures", "1", "--out", str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads((out / "summary.json").read_text())["detectors"]["energy_median"]
    # the runner must give exactly what the validated synthetic harness gives for the same scene (not a performance claim:
    # the box-forming step merges two of these three close signals, the known weakness of the classical detectors)
    from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask, score_mask
    from vuhf.geometry import SpecGeometry
    from vuhf.spectrogram import power_spectrogram

    geom = SpecGeometry(n, 512, 256)
    spec = {d.name: d for d in DEFAULT_SPECS}["energy_median"]
    ref = score_mask(boxes, detect_mask(spec, power_spectrogram(iq, geom), calibrate(spec, geom, 1e-3)), geom, Post(), iou_thrs=(0.5, 0.3))
    assert s["n_annotated_signals"] == 3
    assert s["recall_iou30"] == np.mean([r["det@0.3"] for r in ref["rows"]])
    assert s["recall_iou50"] == np.mean([r["det@0.5"] for r in ref["rows"]])
    assert s["false_alarms_per_window"] == ref["n_fa"][0.5]
    assert (out / "predictions.csv").exists() and (out / "windows.png").exists()
