# Source of the data in `iqengine_cellular_scored`

| | |
|---|---|
| Dataset | IQEngine public library (IQEngine.org), recording `cellular_downlink_880MHz` |
| URL | https://iqengine.org  (public library; data via https://iqengine.org/api/datasources/local/local/cellular_downlink_880MHz.sigmf-data and .sigmf-meta) |
| DOI | none |
| Licence | CC BY-SA (core:license) |
| Downloaded | 2026-10-26 |
| Files | cellular_downlink_880MHz.sigmf-data / .sigmf-meta (80,000,000 bytes, whole file) |
| Sample rate | 40 MS/s |
| Centre frequency | 880 MHz |
| Receiver / author | USRP B210; Jacob Gilbert (OmniSIG) |
| Notes | 8 SigMF annotations (LTE, CDMA, LMR, unknown) lying in samples 100000-1150000 of the 20,000,000-sample file; this run scores the first 5 windows (the first run scored window 0 only, 5 of the 8 signals, and was replaced). |

The recording itself is not in this repository (`data_real/` is ignored). No login or licence click-through was needed.
