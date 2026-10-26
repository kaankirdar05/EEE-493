#!/usr/bin/env bash
# Exact commands of the untuned run (detector settings of the synthetic evaluation; nothing re-tuned).
# Data lives in data_real/ (not committed). Run from the repository root.
set -e
R="python scripts/run_real_data.py"
IQ=data_real/iqengine
SG=data_real/subghz

# --- IQEngine, annotated (SigMF annotations are the ground truth)
$R $IQ/dect6/dect6.sigmf-meta                                   --max-windows 200 --out results/real/iqengine_dect6
$R $IQ/bluetooth/bluetooth.sigmf-meta                           --max-windows 200 --out results/real/iqengine_bluetooth
# cellular: the 8 annotations lie in samples 100000..1150000 of a 20,000,000-sample file, so only the first 5 windows are scored
# (a first run scored window 0 only, 5 of the 8 signals; it was replaced after noticing that the annotations extend further)
$R $IQ/cellular_downlink_880MHz/cellular_downlink_880MHz.sigmf-meta --max-windows 5   --out results/real/iqengine_cellular_scored
$R $IQ/cellular_downlink_880MHz/cellular_downlink_880MHz.sigmf-meta --max-windows 200 --ignore-annotations --out results/real/iqengine_cellular_unscored

# --- IQEngine, wideband but without boxes (counts and pictures only)
$R $IQ/rfd900p/rfd900p.sigmf-meta                               --max-windows 200 --out results/real/iqengine_rfd900p
$R $IQ/ism_band_24/ism_band_24.sigmf-meta                       --max-windows 200 --out results/real/iqengine_ism_band_24
$R $IQ/FM-with-SCA-90M9CF-2M4FS-SGU-videos/FM-with-SCA-90M9CF-2M4FS-SGU-videos.sigmf-meta --max-windows 200 --out results/real/iqengine_fm_sca

# --- Sub-GHz IQ dataset (RTL-SDR, 2.048 MS/s, cabled): two window sizes, both reported
for W in 262144 65536; do
  $R $SG/noise/noise_g0.0dB_att10dB_freq867.4MHz_0.mat $SG/noise/noise_g0.0dB_att10dB_freq867.4MHz_1.mat \
     --fs 2.048e6 --fc 867.4e6 --noise-only --window $W --max-windows 400 --out results/real/subghz_noise_w$W
  $R $SG/lora/lorasf7_g0.0dB_att12dB_freq867.4MHz_0.mat    --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --out results/real/subghz_lora_sf7_w$W
  $R $SG/lora/lorasf12_g0.0dB_att24dB_freq867.4MHz_0.mat   --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --out results/real/subghz_lora_sf12_w$W
  $R $SG/sigfox/sigfox_g0.0dB_att25dB_freq867.4MHz_1.mat   --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --out results/real/subghz_sigfox_w$W
  $R $SG/ieee802154g/zigbee_g0.0dB_att22dB_freq867.4MHz_1.mat --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --out results/real/subghz_802154g_w$W
  $R $SG/ieee80211ah/80211ah_mcs0_chan6_g0.0dB_att3dB_freq867.4MHz_0.mat --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --out results/real/subghz_80211ah_w$W
  $R $SG/sunofdm/sunofdm_O4_mcs2_g0.0dB_att20dB_freq864.0MHz_1.mat --fs 2.048e6 --fc 864.0e6 --window $W --max-windows 400 --out results/real/subghz_sunofdm_w$W
done
