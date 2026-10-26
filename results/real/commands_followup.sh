#!/usr/bin/env bash
# Follow-up runs after the untuned ones (commands_untuned.sh). Run from the repository root; data is in data_real/.
set -e
R="python scripts/run_real_data.py"
IQ=data_real/iqengine
SG=data_real/subghz

# 1. cross-file tuning (selection rule of the synthetic tuning scripts; windows used for tuning are never the ones reported)
python scripts/tune_real.py $IQ/bluetooth/bluetooth.sigmf-meta --out results/real/tuning_on_bluetooth.json
python scripts/tune_real.py $IQ/dect6/dect6.sigmf-meta         --out results/real/tuning_on_dect6.json     # no setting within the false-alarm budget
#    settings chosen on bluetooth (link 1 2, min cells 40, YOLO confidence 0.5) applied to dect6:
$R $IQ/dect6/dect6.sigmf-meta --max-windows 200 --link 1 2 --min-cells 40 --yolo-conf 0.5 --out results/real/iqengine_dect6_tuned_on_bluetooth

# 2. receiver centre spike: mask the 9 centre bins (2N+1 with N = 4, fixed before looking at any result) and compare with the untuned runs
for W in 262144 65536; do
  $R $SG/noise/noise_g0.0dB_att10dB_freq867.4MHz_0.mat $SG/noise/noise_g0.0dB_att10dB_freq867.4MHz_1.mat \
     --fs 2.048e6 --fc 867.4e6 --noise-only --window $W --max-windows 400 --mask-dc 4 --out results/real/subghz_noise_w${W}_dcmask4
  $R $SG/lora/lorasf7_g0.0dB_att12dB_freq867.4MHz_0.mat    --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --mask-dc 4 --out results/real/subghz_lora_sf7_w${W}_dcmask4
  $R $SG/lora/lorasf12_g0.0dB_att24dB_freq867.4MHz_0.mat   --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --mask-dc 4 --out results/real/subghz_lora_sf12_w${W}_dcmask4
  $R $SG/sigfox/sigfox_g0.0dB_att25dB_freq867.4MHz_1.mat   --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --mask-dc 4 --out results/real/subghz_sigfox_w${W}_dcmask4
  $R $SG/ieee802154g/zigbee_g0.0dB_att22dB_freq867.4MHz_1.mat --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --mask-dc 4 --out results/real/subghz_802154g_w${W}_dcmask4
  $R $SG/ieee80211ah/80211ah_mcs0_chan6_g0.0dB_att3dB_freq867.4MHz_0.mat --fs 2.048e6 --fc 867.4e6 --window $W --max-windows 400 --mask-dc 4 --out results/real/subghz_80211ah_w${W}_dcmask4
  $R $SG/sunofdm/sunofdm_O4_mcs2_g0.0dB_att20dB_freq864.0MHz_1.mat --fs 2.048e6 --fc 864.0e6 --window $W --max-windows 400 --mask-dc 4 --out results/real/subghz_sunofdm_w${W}_dcmask4
done
$R $IQ/FM-with-SCA-90M9CF-2M4FS-SGU-videos/FM-with-SCA-90M9CF-2M4FS-SGU-videos.sigmf-meta --max-windows 200 --mask-dc 4 --out results/real/iqengine_fm_sca_dcmask4

# 3. diagnostic only (not a result): dect6 has a spur 6-7 bins above the centre, not at DC; masking 17 centre bins removes it, to test
#    whether the spur is what links consecutive bursts into oversized boxes under the default linking
$R $IQ/dect6/dect6.sigmf-meta --max-windows 200 --mask-dc 8 --out results/real/diagnostic_dect6_mask8
