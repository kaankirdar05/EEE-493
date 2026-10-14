# EEE 493: AI-assisted detection, identification and characterization of V/UHF signals

TÜBİTAK BİLGEM İLTAREN industry project, Bilkent EEE. The chain is
**detect → identify (modulation) → characterize (carrier, bandwidth, SNR)**, each stage a separate,
swappable module with its own metrics.

## Status

Stage 1 is done: the classical detection baselines (energy detector, CA-CFAR) with an evaluation
harness, run on WidebandSig53-style scenes generated with TorchSig.

```
src/vuhf/
  geometry.py            Box, SpecGeometry (physical units <-> spectrogram cells)
  spectrogram.py         I/Q -> power spectrogram (noise variance 1 -> mean cell 1)
  detectors/energy.py    energy detector
  detectors/cfar.py      cell-averaging CFAR (2-D, 1-D along frequency when the time sizes are 0)
  detectors/calibration.py  set a threshold from a noise-only simulation so the Pfa is what you asked for
  detectors/boxes.py     detection mask -> one box per transmission
  data/sources.py        TorchSig scene source (+ small synthetic source for tests), on-disk cache
  eval/                  box matching (IoU), fragments vs false alarms, detection curves
  baseline.py            detector configurations and scene scoring
scripts/
  validate_pfa.py            does each detector hold its design false-alarm rate on pure noise?
  tune_postprocessing.py     choose averaging window / box linking on a validation set
  run_detection_baseline.py  final evaluation: tables, Pd curves, example figure
  diagnose_misses.py         why a signal is missed (too weak / merged with a neighbour / bad box), paired energy vs CFAR
tests/                       14 unit tests (pytest)
results/                     the numbers and figures quoted below
```

## How the two detectors work

Both start from a power spectrogram `P[f, t]` (STFT magnitude squared, frequency × time). With only
noise, each cell is exponentially distributed, so a signal is a region of cells that are
*unusually large for noise*. The detectors differ in how they decide "unusually large".

**Energy detector.** Estimate the noise level once for the whole scene (a low quantile of the
spectrogram, which signals barely affect because they cover a minority of cells; for exponential
noise `mean = median / ln 2`). Average a small block of neighbouring cells (this shrinks the noise
fluctuation, so weaker signals stand out) and flag cells above `multiplier × noise`. The
multiplier comes from the target false-alarm probability. Simple and sensitive, but it assumes the
noise floor is the same everywhere and over time: if the noise rises, false alarms explode; if it
falls, weak signals are missed (the "noise uncertainty" problem).

**CA-CFAR (cell-averaging constant false alarm rate).** For each *cell under test*, estimate the
noise from the cells around it: skip a ring of *guard cells* (so the signal's own energy does not
leak into the estimate), average the *training cells* beyond the guard, and flag the cell if
`P > α × (training average)`. With `N` training cells and exponential noise, `α = N (Pfa^(−1/N) − 1)`
gives exactly the chosen `Pfa` whatever the noise level, hence "constant false alarm rate". The
threshold follows a changing or non-flat noise floor, which the energy detector cannot do. The
weakness is the mirror image: a signal wider (or longer) than the guard region fills its own
training cells, the threshold rises with it, and the middle of the signal is missed. The
frequency-only variants scan along frequency inside each time frame; the guard has to exceed the
signal's half-width.

**From mask to boxes.** Detected cells within a few cells of each other are linked, each connected
group becomes one box drawn tightly around its cells, and tiny groups are dropped. Linking
bridges the gaps weak signals leave in the mask.

## Reproduce

```
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pip install torchsig        # only needed for TorchSig scenes; pulls PyTorch (several GB with CUDA wheels)
pytest

python scripts/validate_pfa.py
python scripts/tune_postprocessing.py --n-scenes 24 --seed 7 --impairment-level 1     # validation scenes
for L in 0 1 2; do
  python scripts/run_detection_baseline.py --source torchsig --n-scenes 300 --seed 1000 --impairment-level $L
done
python scripts/diagnose_misses.py --n-scenes 300 --seed 1000     # why signals are missed, paired energy vs CFAR
```

Tuning uses seed 7, the final evaluation uses different scenes (seed 1000). Scenes are cached in
`data_cache/` (git-ignored, about 2 MB per scene, so about 2 GB for the three levels).

## Results

### The thresholds work: false-alarm rate on pure noise (`results/pfa_validation.json`)

Target cell false-alarm probability 1e-3, measured on 24 fresh noise-only spectrograms of 512 × 1023
cells (about 12 million cells):

| detector | closed-form threshold | calibrated threshold |
|---|---|---|
| energy, global floor | 1.4e-3 | 9.7e-4 |
| energy, per-bin floor | 1.4e-3 | 9.8e-4 |
| CFAR, frequency only | 1.5e-3 | 9.7e-4 |
| CFAR, 2-D | 1.1e-3 | 9.5e-4 |

The closed forms are slightly optimistic because the Hann window and overlap make neighbouring cells
correlated. The calibrated threshold lands on the target. CFAR also keeps its false-alarm rate when
the noise power is scaled by 1000 (unit test).

### Detection on TorchSig wideband scenes

300 scenes per level (750 to 760 signals each), SNR 0 to 20 dB (TorchSig's definition), 1 to 4 signals per
scene, 262 144 samples at 10 MS/s, spectrogram 512 × 1023, design Pfa 1e-3, validation-chosen
averaging 3 × 9 cells and linking (4, 16) with at least 40 cells. Impairment level 0 is perfect,
1 is a cabled transmitter, 2 adds fading (wireless). With 750 signals a recall has a 95 % interval of
about ±3 points, so differences smaller than that are noise.

* **covered**: at least 30 % of the signal's cells are flagged (the detector's sensitivity).
* **recall@0.3 / 0.5**: a predicted box matches the signal's box with that IoU.
* **FA/scene**: predicted boxes lying mostly outside every signal. **frag/scene**: extra pieces of real signals.

Impairment level 1 (cabled), 750 signals:

| detector | covered | recall@0.3 | recall@0.5 | FA/scene | frag/scene | cell Pfa | ms/scene |
|---|---|---|---|---|---|---|---|
| energy, median floor | 0.83 | 0.75 | 0.68 | 0.15 | 0.33 | 2.3e-4 | 23 |
| energy, 25 % quantile floor | 0.84 | 0.75 | 0.68 | 0.15 | 0.34 | 2.6e-4 | 22 |
| energy, median + 6 dB (plan's rule) | 0.59 | 0.74 | 0.68 | 0.13 | 0.22 | 1.9e-4 | 20 |
| energy, per-bin floor | 0.44 | 0.61 | 0.54 | 0.19 | 1.44 | 8.7e-4 | 24 |
| CA-CFAR, frequency only, guard 24 | 0.45 | 0.77 | 0.65 | 0.20 | 0.60 | 8.4e-4 | 24 |
| CA-CFAR, frequency only, guard 8 | 0.15 | 0.63 | 0.50 | 0.28 | 1.56 | 1.0e-3 | 24 |
| CA-CFAR, 2-D | 0.00 | 0.16 | 0.11 | 0.08 | 0.45 | 9.8e-4 | 22 |

The other two levels give the same picture. Median energy detector / wide-guard CFAR, recall@0.3:
perfect 0.73 / 0.73, cabled 0.75 / 0.77, wireless 0.73 / 0.75. Covered: 0.84 / 0.44, 0.83 / 0.45, 0.80 / 0.41.
Full tables and curves are in `results/torchsig_imp{0,1,2}/` (`summary.json`, `per_signal.csv`, `pd_curves.png`,
`example_scene.png`); the miss analysis below is `scripts/diagnose_misses.py`.

Why a signal is not matched at IoU 0.3 (percent of all signals, three levels):

| detector | matched | not covered (too weak / narrow) | covered, overlaps a neighbour | covered, isolated, bad box |
|---|---|---|---|---|
| energy, median floor | 73 to 75 | 5 to 8 | 17 to 19 | 1.5 to 2 |
| CA-CFAR, freq only, guard 24 | 73 to 77 | 15 to 19 | 5 to 6 | 2 to 3 |

### What the numbers say

1. **The detectors work.** The energy detector flags 92 to 94 % of signals with SNR ≥ 5 dB and about
   half (49 to 55 %) of those below 5 dB, with 0.15 to 0.26 false-alarm boxes per scene and a matched-box median IoU of 0.88 to 0.89.
   On isolated signals (no other signal overlapping in time and nearby in frequency) recall@0.3 is 0.86 to 0.93 for both detectors.
2. **Detection is not the bottleneck, box forming is.** About 39 % of the signals overlap in time with a neighbour
   that is within 0.01 cycles/sample in frequency. For these, recall is only about 0.47 (energy), because connected
   components merge the touching signals into one box (the example figure shows three signals in one red box).
   This costs the energy detector about 18 points of recall and is the largest single loss. Splitting touching
   signals is a segmentation problem (job 4.3) and is where the AI detector (4.4) should help.
3. **Energy and CA-CFAR end up equal on box matching, for opposite reasons.** The paired comparison on the same signals
   is not significant at any level (McNemar |z| ≤ 1.8). The energy detector loses signals by merging them (18 % of signals)
   and the CFAR loses them by not seeing the middle of wide ones (15 to 19 % not covered, against 5 to 8 %).
   CFAR's per-cell sensitivity is much lower (covered 0.41 to 0.45 against 0.80 to 0.84), but its boxes cover only the edges and the parts
   of signals that stick out, which splits touching signals apart and gives it slightly better recall on overlapping ones
   (0.53 to 0.55 against 0.47 to 0.48). Its boxes are looser: median IoU 0.77 to 0.80 against 0.88 to 0.89, and its false alarms are higher
   (0.20 to 0.37 against 0.15 to 0.26 per scene) at the same design Pfa.
4. **CA-CFAR self-masking is real.** The narrow-guard CFAR covers only 13 to 16 % of the signals, the wide-guard
   CFAR 41 to 45 %, against 80 to 84 % for the energy detector, because TorchSig signals are often wide and long-lived. The 2-D CFAR with a
   small window finds almost nothing for the same reason.
5. **A per-bin noise floor hurts** here (covered 0.43 to 0.48 against 0.80 to 0.84), as expected: a signal that stays on in a
   bin for more than half the time biases that bin's median upward. A global floor works better, and a 25 %
   quantile floor is on par with the median.
6. **Narrow signals are the other weak spot.** Signals narrower than 0.02 cycles/sample (0.2 MHz here) are matched only 19 to 43 % of the time by
   the energy detector and 6 to 14 % by the CFAR (the 3 × 9 averaging and the 2-bin minimum width remove them).
7. **CFAR's advantage needs a non-flat noise floor** (receiver roll-off, interference). The noise floor in these
   TorchSig scenes looks flat, which favours the energy detector, so this comparison is not the last word.
   Real SDR captures are the fair test.
8. **The measured cell false-alarm rate is lower than the design 1e-3 for the energy detector** at levels 1 and 2 (2e-4 to 4e-4),
   measured on cells away from signals. The calibration on pure noise holds (see above), so I think part of the scene's noise estimate
   is pulled up by signals or by the noise not being quite exponential in impaired scenes, but I have not diagnosed it. It makes the
   detector slightly conservative, not wrong.

## Data notes

* The scenes come from the **TorchSig 2.2.0 generator** (`WidebandSig53`-style: several signals per scene,
  box labels, impairment levels). They are **not the original static WidebandSig53 release**, and TorchSig 2.x
  draws from its full signal list (OFDM variants, BLE, ADS-B, LoRa, DVB-S2, tones, chirps and others) rather than the 53 classes of the paper.
  The detection numbers here are not comparable with the paper's mAP.
* **SNR definition.** In 2.2.0 `snr_db` is set from the peak of the signal's time-averaged PSD above the
  noise floor per FFT bin (`update_signal_snr_bandwidth` in `torchsig/utils/dsp.py`). For a
  flat-spectrum signal this is the in-band SNR. It is not the Es/N0 of the Sig53 paper, so
  any report line saying "TorchSig uses Es/N0" should name the version.
* Ground-truth bandwidth is TorchSig's own estimate (bins within 3 dB of the noise floor in the max-hold
  spectrum, widened by half a bin), so very weak signals can have a very narrow box.

## Limitations and next steps

* One-bin-wide signals (tones, ADS-B) are missed: the 3 × 9 averaging and the 2-bin minimum width remove them.
  Handle narrowband separately (a 1-bin detector with a stricter Pfa) or by channelizing first.
* Touching signals are merged (see above, about 18 points of recall). Next: segmentation / YOLO-style detector (jobs 4.3, 4.4) and a
  comparison on the same scenes.
* All numbers are on synthetic scenes. The same harness runs on real SDR captures once they carry box labels.
* Next stage: modulation identification (cumulants + SVM, then CNN/ResNet) on the crops these boxes produce.
