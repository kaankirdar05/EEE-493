# Source of the data in `subghz_sigfox_w262144_dcmask4`

| | |
|---|---|
| Dataset | "IQ signals captured from multiple Sub-GHz technologies" (Fontaine, Shahid, et al., IMEC / Ghent University) |
| URL | https://ieee-dataport.org/documents/iq-signals-captured-multiple-sub-ghz-technologies (needs an IEEE DataPort subscription, **not used**); the files were taken from the public share linked in the authors' GitHub README, https://github.com/JaronFontaine/Sub-GHz-IQ-signals-dataset -> https://cloud.ilabt.imec.be/index.php/s/bqXtdp9QsfXLbb3 (no login, no password) |
| DOI | 10.21227/2m0z-5s90 |
| Licence | CC BY-NC-SA 4.0 (stated in the GitHub README; the IEEE DataPort page states none). Non-commercial. Authors ask that the publication Fontaine et al., VTC2020-Fall, "Multi-band sub-GHz technology recognition on NVIDIA's Jetson Nano" is cited. |
| Downloaded | 2026-10-26 |
| Files | `867.4/Sigfox/sigfox_g0.0dB_att25dB_freq867.4MHz_1.mat` (131,072,208 bytes each, MATLAB v5, array `IQ_samples`, 1 x 8,192,000 complex doubles) |
| Sample rate | 2.048 MS/s (README) |
| Centre frequency | 867.4 MHz (from the file name and the README) |
| Receiver | RTL-SDR over coaxial cable (not over the air) |
| Notes | Sigfox, 100 Hz BPSK, one 4 s capture. The share holds 16.4 GB; only the files listed here were downloaded. |

The recording itself is not in this repository (`data_real/` is ignored).
