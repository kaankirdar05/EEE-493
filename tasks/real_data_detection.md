# Task for a local session: run the three detectors on real recordings

Paste the whole file as the first message of a new Claude Code session on a machine with internet access.
The branch is `claude/real-data-detection` of https://github.com/kaankirdar05/EEE-493 .

---

## Goal

Run our three detection models on the **real datasets listed at the end of `EEE493_SNR_Datasets.pdf`** (Section 4, "Candidate
Real Datasets"), using only the datasets that are **proper for the detection task**, and report honestly how each model does.
The three models, all already implemented in this repo (`src/vuhf`):

| name in the code | what it is |
|---|---|
| `energy_median` | energy detector: 3 x 9 cell averaging, global noise floor from the median, threshold for a cell false-alarm probability of 1e-3 |
| `cfar_freq_g24_t48` | cell-averaging CFAR along frequency (guard 24, training 48 bins each side), same Pfa |
| `yolo` | YOLOv8n, one class "signal", weights `results/yolo/best.pt`, confidence 0.05, trained on 3,000 synthetic TorchSig wideband scenes |

All three work on the same power spectrogram (512-point FFT, hop 256) of a window of I/Q samples; the classical ones threshold
the numbers, YOLO sees a 512 x 512 grayscale image of it. Box-forming for the classical ones: link (4, 16) cells, at least 40 cells.

## What already exists (do not rewrite it)

* `scripts/run_real_data.py` + `src/vuhf/data/real.py`: readers for SigMF (`.sigmf-meta`, with annotations), SDRangel `.sdriq`, 2-channel
  `.wav`, MATLAB `.mat` (v5 and v7.3; 2-D arrays are treated as independent snippets), raw `.cu8`/`.cs8`/`.cf32`; windowing; the
  three detectors; predictions in seconds and absolute Hz; a figure; scoring (recall at IoU 0.5 and 0.3, false alarms, fragments,
  median IoU) whenever annotations exist (SigMF annotations or `--annotations-csv`). `python scripts/run_real_data.py --help`.
* `tests/test_real_data.py`: 7 tests with synthetic files. The whole suite is 26 tests and passed when this was written.
* The readers were tested on **synthetic files written by the tests, not on any real dataset**. Expect to fix format details
  (field names, scaling, centre frequency) when the first real file is opened. Say what you changed.

## Setup

```
git clone -b claude/real-data-detection https://github.com/kaankirdar05/EEE-493 && cd EEE-493
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest ultralytics h5py        # TorchSig is NOT needed for this task
pytest                                          # expect 26 passed
```
Check that `results/yolo/best.pt` is the real 6.4 MB file. A GPU is optional (YOLO is about 0.4 s per window on one CPU thread).

## The datasets (from the PDF, with what a web search found; verify every link yourself)

The PDF gives no URLs. The remote session that prepared this task had no access to these sites, so **none of the data was opened**.

| PDF entry | Where it seems to be | My assessment for detection |
|---|---|---|
| IQEngine.org | https://iqengine.org (community SigMF recordings; SigMF annotations hold time/frequency boxes) | **Best**: the only source that can have ground-truth boxes. Use recordings that carry annotations. |
| SDRangel IQ files | no official collection found; `.sdriq` / `.wav`, 250 kS/s to 4 MS/s, real over-the-air, several signals per file | **Good** wideband multi-signal data, no boxes. Figures, box counts, occupancy; optionally label by hand. |
| SigIDWiki | https://www.sigidwiki.com (GSM BCCH/traffic, FM, many others, with waterfall pictures) | Qualitative: known system present, no boxes. |
| Sub-GHz IQ dataset | the PDF says GitHub, but the only place found is IEEE DataPort, "IQ signals captured from multiple Sub-GHz technologies", DOI 10.21227/2m0z-5s90, https://ieee-dataport.org/documents/iq-signals-captured-multiple-sub-ghz-technologies (RTL-SDR, 2.048 MS/s, coax, 864.0 and 867.4 MHz; Sigfox, LoRa, 802.15.4g, SUN-OFDM, 802.11ah, plus a noise class; `.mat`, array `IQ_samples`). Login needed. | Good for **burst presence versus noise** and false alarms on the noise class; no boxes. Ask the user for the GitHub link if there is one. |
| LoRa datasets (Zenodo / IEEE DataPort) | Zenodo records 17708397 and 20341802 turned up for LoRa SigMF datasets (one seems to be "LoRaIQ", EPFL, 75,000 frames, USRP-2920); a 25-device USRP B210 fingerprinting set exists but no link was found | **Weak**: one transmitter, short frames, labelled by device. Use only as a burst-presence check. |
| Technology-recognition sets (IEEE DataPort) | IDLab "Dataset: IQ samples of LTE, 5G NR, Wi-Fi, ITS-G5, and C-V2X PC5", DOI 10.21227/72qq-z464 (1 to 25 Msps, 7,500 examples per technology, noise class). The PDF also lists DVB-T; the search did not confirm it. Login needed. | **Not proper for box detection**: one technology per clip, built for classification. At most a clip-level "signal vs noise" check. |

Decision rule: a dataset is proper for detection only if it is a wideband capture where signals occupy only part of the time-frequency
plane (so that "where" is a meaningful question). Single-signal clips cannot be scored as boxes: in Dataset 1 (the team's synthetic
set, one centred signal per clip) a data-blind box scored recall 0.99, so recall alone proves nothing there. Always report
false alarms on noise/empty windows next to recall.

If a dataset needs a login, a licence click-through or a subscription, **stop and ask the user**; do not work around it.
Record for every dataset used, in `results/real/<name>/SOURCE.md`: URL, DOI, licence, download date, file names, sample rate,
centre frequency. The paper requires that licences are checked before anything enters a deliverable. Never commit the data (the
`.gitignore` excludes `data_real/`); commit only small result files (summary, predictions, figures).

## What to do

1. Find and download the datasets above in the priority order of the table. Start with 2 to 3 annotated IQEngine/SigMF recordings, 2 to 3 SDRangel or SigIDWiki wideband recordings, and the Sub-GHz IQ files including its noise class.
2. For each file get the sample rate and centre frequency right (SigMF metadata; `.sdriq` header; otherwise the dataset documentation), and run
   `python scripts/run_real_data.py <file> [--fs ... --fc ...] --out results/real/<name>`.
   Use `--noise-only` for recordings known to contain no signal. Open every `windows.png` and look at it.
3. Window size: the synthetic scenes were 262,144 samples (26 ms at 10 MS/s), signals 2.5 to 25 % of the band wide, lasting 20 to 100 % of the window. For
   low-rate or bursty data (LoRa, Sigfox) try `--window 65536` as well and say which you report. Do **not** choose settings on the windows you report:
   if you re-tune anything (Pfa, linking, YOLO confidence), tune on separate files and report both the untuned and the tuned numbers.
4. If the receiver shows a DC spike (the PDF says to discard the centre bins), report the detectors untouched first. If the spike causes false alarms, add a
   clearly separate, documented option (for example masking the centre bins) and report with and without it. The detectors currently have no such option.
5. Report per dataset: files, windows, annotated signals, recall at IoU 0.3 and 0.5, false alarms and fragments per window, boxes per window, and for noise-only
   recordings the share of windows with any box. State plainly when there are no labels and the result is only counts and pictures, and which detections you judged
   right or wrong by eye. Annotations in public recordings are often incomplete, so a detection with no matching annotation can be a real, unlabelled signal.
6. Put the numbers next to what we already know (below) and say whether the real data behave like the clean synthetic scenes or like the impaired ones.
7. Commit the small results and a short `README` section to the branch `claude/real-data-detection` and push it. Do not open a pull request unless asked.

## Known results on synthetic data (for comparison; recall at IoU 0.3)

| data | energy | CA-CFAR | YOLO |
|---|---|---|---|
| TorchSig scenes, cabled (README table), 750 signals | 0.75 | 0.77 | 0.96 |
| Cem's generator, clean scenes, 667 signals | 0.86 | 0.84 | 0.96 |
| Cem's generator, impaired scenes (receiver effects), 673 signals | 0.19 | 0.38 | 0.74, but 6.7 false boxes per scene |

On the impaired scenes the noise floor jumps or decays inside a scene and has a curved passband; all three detectors flagged the bright noise. **Expect real
captures to look like that**, and expect CFAR to beat the energy detector there, as its design predicts. YOLO was trained on flat-noise scenes with long signals;
on short bursts or non-flat floors treat its output with suspicion and say so.

## Rules

* Report what happened, including failures. Do not invent, interpolate or "fix" results; do not substitute synthetic data for real data.
* Do not change the existing synthetic results or the detector settings used for them. New options go in as additional, documented flags.
* Keep the code style of `src/vuhf` (short, commented where the reason is not obvious) and keep `pytest` green.
