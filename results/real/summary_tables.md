## Recordings with annotations (scored)

| run | detector | windows | signals | recall IoU 0.3 | recall IoU 0.5 | median IoU | covered >= 30 % | boxes / window | false alarms / window | fragments / window |
|---|---|---|---|---|---|---|---|---|---|---|
| diagnostic_dect6_mask8 | energy | 152 | 1992 | 0.00 | 0.00 | - | 1.00 | 30.1 | 30.0 | 0.09 |
| diagnostic_dect6_mask8 | CA-CFAR | 152 | 1992 | 0.00 | 0.00 | - | 1.00 | 27.2 | 27.2 | 0.05 |
| diagnostic_dect6_mask8 | YOLO | 152 | 1992 | 0.00 | 0.00 | - | 1.00 | 3.2 | 3.2 | 0.00 |
| iqengine_bluetooth | energy | 152 | 184 | 0.85 | 0.59 | 0.67 | 1.00 | 1.5 | 0.8 | 0.00 |
| iqengine_bluetooth | CA-CFAR | 152 | 184 | 0.73 | 0.40 | 0.67 | 1.00 | 1.4 | 0.9 | 0.00 |
| iqengine_bluetooth | YOLO | 152 | 184 | 0.22 | 0.21 | 0.71 | 0.97 | 1.6 | 1.3 | 0.01 |
| iqengine_cellular_scored | energy | 5 | 8 | 0.50 | 0.38 | 0.61 | 0.75 | 4.0 | 3.4 | 0.00 |
| iqengine_cellular_scored | CA-CFAR | 5 | 8 | 0.50 | 0.38 | 0.54 | 0.75 | 11.2 | 10.0 | 0.60 |
| iqengine_cellular_scored | YOLO | 5 | 8 | 0.50 | 0.50 | 0.58 | 0.88 | 25.4 | 23.6 | 1.00 |
| iqengine_dect6 | energy | 152 | 1992 | 0.08 | 0.00 | 0.52 | 1.00 | 15.1 | 15.1 | 0.00 |
| iqengine_dect6 | CA-CFAR | 152 | 1992 | 0.00 | 0.00 | - | 1.00 | 10.7 | 10.7 | 0.00 |
| iqengine_dect6 | YOLO | 152 | 1992 | 0.00 | 0.00 | - | 1.00 | 1.8 | 1.8 | 0.00 |
| iqengine_dect6_tuned_on_bluetooth | energy | 152 | 1992 | 0.40 | 0.00 | 0.54 | 1.00 | 22.4 | 22.4 | 0.00 |
| iqengine_dect6_tuned_on_bluetooth | CA-CFAR | 152 | 1992 | 0.35 | 0.00 | 0.56 | 0.99 | 22.4 | 22.4 | 0.00 |
| iqengine_dect6_tuned_on_bluetooth | YOLO | 152 | 1992 | 0.00 | 0.00 | - | 1.00 | 1.0 | 1.0 | 0.00 |

## Noise-only recordings (every box is a false alarm)

| run | window | DC bins masked | detector | windows | windows with a box | boxes / window | share of spectrogram inside boxes |
|---|---|---|---|---|---|---|---|
| subghz_noise_w262144 | 262144 | 0 | energy | 62 | 1.00 | 5.1 | 0.08 |
| subghz_noise_w262144 | 262144 | 0 | CA-CFAR | 62 | 1.00 | 2.1 | 0.07 |
| subghz_noise_w262144 | 262144 | 0 | YOLO | 62 | 1.00 | 5.5 | 0.96 |
| subghz_noise_w262144_dcmask4 | 262144 | 9 | energy | 62 | 0.94 | 2.4 | 0.01 |
| subghz_noise_w262144_dcmask4 | 262144 | 9 | CA-CFAR | 62 | 0.69 | 1.1 | 0.01 |
| subghz_noise_w262144_dcmask4 | 262144 | 9 | YOLO | 62 | 1.00 | 4.2 | 0.97 |
| subghz_noise_w65536 | 65536 | 0 | energy | 250 | 1.00 | 1.3 | 0.03 |
| subghz_noise_w65536 | 65536 | 0 | CA-CFAR | 250 | 1.00 | 1.2 | 0.04 |
| subghz_noise_w65536 | 65536 | 0 | YOLO | 250 | 1.00 | 5.9 | 0.89 |
| subghz_noise_w65536_dcmask4 | 65536 | 9 | energy | 250 | 0.15 | 0.2 | 0.00 |
| subghz_noise_w65536_dcmask4 | 65536 | 9 | CA-CFAR | 250 | 0.15 | 0.2 | 0.01 |
| subghz_noise_w65536_dcmask4 | 65536 | 9 | YOLO | 250 | 0.93 | 1.7 | 0.51 |

## Recordings without labels (counts only; judge the pictures, not the numbers)

| run | window | DC bins masked | detector | windows | windows with a box | boxes / window | share of spectrogram inside boxes |
|---|---|---|---|---|---|---|---|
| iqengine_cellular_unscored | 262144 | 0 | energy | 76 | 1.00 | 4.6 | 0.69 |
| iqengine_cellular_unscored | 262144 | 0 | CA-CFAR | 76 | 1.00 | 8.6 | 0.63 |
| iqengine_cellular_unscored | 262144 | 0 | YOLO | 76 | 1.00 | 30.1 | 0.81 |
| iqengine_fm_sca | 262144 | 0 | energy | 91 | 1.00 | 9.0 | 0.58 |
| iqengine_fm_sca | 262144 | 0 | CA-CFAR | 91 | 1.00 | 5.1 | 0.36 |
| iqengine_fm_sca | 262144 | 0 | YOLO | 91 | 1.00 | 36.9 | 0.90 |
| iqengine_fm_sca_dcmask4 | 262144 | 9 | energy | 91 | 1.00 | 9.1 | 0.58 |
| iqengine_fm_sca_dcmask4 | 262144 | 9 | CA-CFAR | 91 | 1.00 | 5.1 | 0.41 |
| iqengine_fm_sca_dcmask4 | 262144 | 9 | YOLO | 91 | 1.00 | 38.6 | 0.89 |
| iqengine_ism_band_24 | 262144 | 0 | energy | 106 | 1.00 | 12.8 | 0.35 |
| iqengine_ism_band_24 | 262144 | 0 | CA-CFAR | 106 | 1.00 | 9.8 | 0.53 |
| iqengine_ism_band_24 | 262144 | 0 | YOLO | 106 | 1.00 | 14.0 | 0.93 |
| iqengine_rfd900p | 262144 | 0 | energy | 43 | 0.95 | 1.5 | 0.04 |
| iqengine_rfd900p | 262144 | 0 | CA-CFAR | 43 | 0.98 | 2.0 | 0.05 |
| iqengine_rfd900p | 262144 | 0 | YOLO | 43 | 0.98 | 7.2 | 0.94 |
| subghz_80211ah_w262144 | 262144 | 0 | energy | 31 | 1.00 | 2.1 | 0.69 |
| subghz_80211ah_w262144 | 262144 | 0 | CA-CFAR | 31 | 1.00 | 2.3 | 0.79 |
| subghz_80211ah_w262144 | 262144 | 0 | YOLO | 31 | 1.00 | 5.6 | 1.00 |
| subghz_80211ah_w262144_dcmask4 | 262144 | 9 | energy | 31 | 1.00 | 4.3 | 0.62 |
| subghz_80211ah_w262144_dcmask4 | 262144 | 9 | CA-CFAR | 31 | 1.00 | 2.3 | 0.84 |
| subghz_80211ah_w262144_dcmask4 | 262144 | 9 | YOLO | 31 | 1.00 | 5.2 | 1.00 |
| subghz_80211ah_w65536 | 65536 | 0 | energy | 125 | 1.00 | 2.1 | 0.66 |
| subghz_80211ah_w65536 | 65536 | 0 | CA-CFAR | 125 | 1.00 | 2.3 | 0.74 |
| subghz_80211ah_w65536 | 65536 | 0 | YOLO | 125 | 1.00 | 4.1 | 0.98 |
| subghz_80211ah_w65536_dcmask4 | 65536 | 9 | energy | 125 | 1.00 | 2.4 | 0.61 |
| subghz_80211ah_w65536_dcmask4 | 65536 | 9 | CA-CFAR | 125 | 1.00 | 2.2 | 0.78 |
| subghz_80211ah_w65536_dcmask4 | 65536 | 9 | YOLO | 125 | 1.00 | 2.6 | 0.97 |
| subghz_802154g_w262144 | 262144 | 0 | energy | 31 | 1.00 | 4.9 | 0.14 |
| subghz_802154g_w262144 | 262144 | 0 | CA-CFAR | 31 | 1.00 | 5.8 | 0.19 |
| subghz_802154g_w262144 | 262144 | 0 | YOLO | 31 | 1.00 | 12.5 | 0.90 |
| subghz_802154g_w262144_dcmask4 | 262144 | 9 | energy | 31 | 1.00 | 3.6 | 0.12 |
| subghz_802154g_w262144_dcmask4 | 262144 | 9 | CA-CFAR | 31 | 1.00 | 5.6 | 0.15 |
| subghz_802154g_w262144_dcmask4 | 262144 | 9 | YOLO | 31 | 1.00 | 11.2 | 0.90 |
| subghz_802154g_w65536 | 65536 | 0 | energy | 125 | 1.00 | 2.1 | 0.14 |
| subghz_802154g_w65536 | 65536 | 0 | CA-CFAR | 125 | 1.00 | 2.3 | 0.16 |
| subghz_802154g_w65536 | 65536 | 0 | YOLO | 125 | 1.00 | 6.1 | 0.76 |
| subghz_802154g_w65536_dcmask4 | 65536 | 9 | energy | 125 | 0.82 | 1.0 | 0.12 |
| subghz_802154g_w65536_dcmask4 | 65536 | 9 | CA-CFAR | 125 | 0.81 | 1.5 | 0.13 |
| subghz_802154g_w65536_dcmask4 | 65536 | 9 | YOLO | 125 | 0.98 | 2.7 | 0.45 |
| subghz_lora_sf12_w262144 | 262144 | 0 | energy | 31 | 1.00 | 2.2 | 0.26 |
| subghz_lora_sf12_w262144 | 262144 | 0 | CA-CFAR | 31 | 1.00 | 4.0 | 0.21 |
| subghz_lora_sf12_w262144 | 262144 | 0 | YOLO | 31 | 1.00 | 14.1 | 0.98 |
| subghz_lora_sf12_w262144_dcmask4 | 262144 | 9 | energy | 31 | 1.00 | 1.2 | 0.24 |
| subghz_lora_sf12_w262144_dcmask4 | 262144 | 9 | CA-CFAR | 31 | 1.00 | 3.1 | 0.16 |
| subghz_lora_sf12_w262144_dcmask4 | 262144 | 9 | YOLO | 31 | 1.00 | 14.4 | 0.97 |
| subghz_lora_sf12_w65536 | 65536 | 0 | energy | 125 | 1.00 | 2.0 | 0.24 |
| subghz_lora_sf12_w65536 | 65536 | 0 | CA-CFAR | 125 | 1.00 | 2.3 | 0.17 |
| subghz_lora_sf12_w65536 | 65536 | 0 | YOLO | 125 | 1.00 | 6.3 | 0.67 |
| subghz_lora_sf12_w65536_dcmask4 | 65536 | 9 | energy | 125 | 0.96 | 1.0 | 0.23 |
| subghz_lora_sf12_w65536_dcmask4 | 65536 | 9 | CA-CFAR | 125 | 0.95 | 1.3 | 0.14 |
| subghz_lora_sf12_w65536_dcmask4 | 65536 | 9 | YOLO | 125 | 0.99 | 5.4 | 0.57 |
| subghz_lora_sf7_w262144 | 262144 | 0 | energy | 31 | 1.00 | 2.8 | 0.13 |
| subghz_lora_sf7_w262144 | 262144 | 0 | CA-CFAR | 31 | 1.00 | 3.3 | 0.14 |
| subghz_lora_sf7_w262144 | 262144 | 0 | YOLO | 31 | 1.00 | 6.6 | 0.93 |
| subghz_lora_sf7_w262144_dcmask4 | 262144 | 9 | energy | 31 | 0.97 | 1.5 | 0.09 |
| subghz_lora_sf7_w262144_dcmask4 | 262144 | 9 | CA-CFAR | 31 | 0.97 | 2.3 | 0.08 |
| subghz_lora_sf7_w262144_dcmask4 | 262144 | 9 | YOLO | 31 | 1.00 | 5.5 | 0.95 |
| subghz_lora_sf7_w65536 | 65536 | 0 | energy | 125 | 1.00 | 1.7 | 0.11 |
| subghz_lora_sf7_w65536 | 65536 | 0 | CA-CFAR | 125 | 1.00 | 1.7 | 0.09 |
| subghz_lora_sf7_w65536 | 65536 | 0 | YOLO | 125 | 1.00 | 5.3 | 0.77 |
| subghz_lora_sf7_w65536_dcmask4 | 65536 | 9 | energy | 125 | 0.57 | 0.6 | 0.08 |
| subghz_lora_sf7_w65536_dcmask4 | 65536 | 9 | CA-CFAR | 125 | 0.58 | 0.7 | 0.06 |
| subghz_lora_sf7_w65536_dcmask4 | 65536 | 9 | YOLO | 125 | 0.96 | 2.6 | 0.50 |
| subghz_sigfox_w262144 | 262144 | 0 | energy | 31 | 1.00 | 10.8 | 0.30 |
| subghz_sigfox_w262144 | 262144 | 0 | CA-CFAR | 31 | 1.00 | 11.4 | 0.19 |
| subghz_sigfox_w262144 | 262144 | 0 | YOLO | 31 | 1.00 | 14.5 | 0.98 |
| subghz_sigfox_w262144_dcmask4 | 262144 | 9 | energy | 31 | 1.00 | 13.2 | 0.27 |
| subghz_sigfox_w262144_dcmask4 | 262144 | 9 | CA-CFAR | 31 | 1.00 | 11.2 | 0.16 |
| subghz_sigfox_w262144_dcmask4 | 262144 | 9 | YOLO | 31 | 1.00 | 15.5 | 0.95 |
| subghz_sigfox_w65536 | 65536 | 0 | energy | 125 | 1.00 | 4.7 | 0.26 |
| subghz_sigfox_w65536 | 65536 | 0 | CA-CFAR | 125 | 1.00 | 4.6 | 0.14 |
| subghz_sigfox_w65536 | 65536 | 0 | YOLO | 125 | 1.00 | 8.5 | 0.94 |
| subghz_sigfox_w65536_dcmask4 | 65536 | 9 | energy | 125 | 1.00 | 4.9 | 0.24 |
| subghz_sigfox_w65536_dcmask4 | 65536 | 9 | CA-CFAR | 125 | 0.99 | 4.2 | 0.12 |
| subghz_sigfox_w65536_dcmask4 | 65536 | 9 | YOLO | 125 | 0.99 | 7.1 | 0.82 |
| subghz_sunofdm_w262144 | 262144 | 0 | energy | 31 | 1.00 | 4.2 | 0.12 |
| subghz_sunofdm_w262144 | 262144 | 0 | CA-CFAR | 31 | 1.00 | 4.6 | 0.16 |
| subghz_sunofdm_w262144 | 262144 | 0 | YOLO | 31 | 1.00 | 11.0 | 0.98 |
| subghz_sunofdm_w262144_dcmask4 | 262144 | 9 | energy | 31 | 1.00 | 3.1 | 0.09 |
| subghz_sunofdm_w262144_dcmask4 | 262144 | 9 | CA-CFAR | 31 | 1.00 | 3.6 | 0.10 |
| subghz_sunofdm_w262144_dcmask4 | 262144 | 9 | YOLO | 31 | 1.00 | 9.5 | 0.98 |
| subghz_sunofdm_w65536 | 65536 | 0 | energy | 125 | 1.00 | 2.2 | 0.10 |
| subghz_sunofdm_w65536 | 65536 | 0 | CA-CFAR | 125 | 1.00 | 2.1 | 0.12 |
| subghz_sunofdm_w65536 | 65536 | 0 | YOLO | 125 | 1.00 | 6.3 | 0.88 |
| subghz_sunofdm_w65536_dcmask4 | 65536 | 9 | energy | 125 | 0.58 | 1.1 | 0.07 |
| subghz_sunofdm_w65536_dcmask4 | 65536 | 9 | CA-CFAR | 125 | 0.58 | 1.1 | 0.09 |
| subghz_sunofdm_w65536_dcmask4 | 65536 | 9 | YOLO | 125 | 0.98 | 3.4 | 0.55 |
