# Sample of the test data

12 scenes, 38 signals: the **first 4 scenes of each impairment level** of the test set (seed 1000, scenes 0 to 3, not hand-picked).
The full set (900 test scenes, 3 000 training images) is not stored in git; it is regenerated from fixed seeds (see the main README, "Reproduce").
`overview.png` shows all 12 scenes on one page.

| folder / file | what it is |
|---|---|
| `iq/` | the scenes as `.npz`: `iq` (complex64, 262 144 samples), `boxes` (one row per signal: `t0, t1, f0, f1, snr_db`), `labels` (signal type). Same file names as the cache of `run_detection_baseline.py` |
| `spectrogram_images/` | `imp<L>_scene<NNNN>.png`: the 512 x 512 grayscale image YOLO is trained on and sees (brightness = power over the noise floor, 0 to 255). `.txt` next to it: its labels in YOLO format, one line per signal, `class x_centre y_centre width height`, all as fractions of the image (x = time, y = frequency, class 0 = signal) |
| `annotated/` | the same spectrogram with the true boxes, signal types and SNRs drawn on it, axes in ms and MHz, for people |
| `labels.csv` | one row per true signal: file, impairment level, scene, signal type, SNR (dB), start and stop (ms), low and high frequency (MHz), bandwidth (MHz) |

## Units

* Scenes are 262 144 samples at 10 MS/s, which is 26.2 ms. Frequency in `boxes` is in cycles per sample (-0.5 to 0.5); multiply by 10 to get MHz.
  Time in `boxes` is in samples; divide by 10 000 to get ms.
* Impairment level 0 = perfect, 1 = cabled transmitter, 2 = wireless (transmitter impairments plus fading).
* `snr_db` is TorchSig 2.2.0's definition (peak of the signal's average spectrum above the noise floor, per FFT bin), not Es/N0.
* The frequency extent of a box is TorchSig's own bandwidth estimate (bins at least 3 dB above the noise floor), so very weak signals get narrow boxes.

## Use it

```python
import numpy as np
z = np.load("data_sample/iq/torchsig_imp1_snr0-20_seed1000_0000.npz")
iq, boxes, labels = z["iq"], z["boxes"], z["labels"]      # (262144,) complex64, (n, 5), (n,)
```

The evaluation scripts run on the sample without TorchSig installed (the scenes are read from `iq/`):

```
python scripts/run_detection_baseline.py --source torchsig --n-scenes 4 --seed 1000 --impairment-level 1 \
    --cache-dir data_sample/iq --out /tmp/sample_eval
python scripts/run_detection_baseline.py ... --yolo-weights results/yolo/best.pt --yolo-conf 0.05   # adds YOLO
```

Rebuild the sample from the full cache with `python scripts/export_sample.py --n-per-level 4`.
