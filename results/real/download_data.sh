#!/usr/bin/env bash
# The exact downloads of the real-data run (2026-10-26, about 1.7 GB). Data goes to data_real/, which git ignores.
# IQEngine recordings: public, no login. The Sub-GHz files come from the public share linked in the dataset authors' GitHub README
# (https://github.com/JaronFontaine/Sub-GHz-IQ-signals-dataset), no login or password; the IEEE DataPort copy needs a subscription and was not used.
set -e
U=https://iqengine.org/api/datasources/local/local
for n in dect6 bluetooth cellular_downlink_880MHz rfd900p; do
  mkdir -p data_real/iqengine/$n
  curl -s -o data_real/iqengine/$n/$n.sigmf-data "$U/$n.sigmf-data"
  curl -s -o data_real/iqengine/$n/$n.sigmf-meta "$U/$n.sigmf-meta"
done
# the two large recordings: only the first part of the data file (a SigMF data file is a plain sample stream, so a prefix is valid)
n=ism_band_24;                          mkdir -p data_real/iqengine/$n; curl -s "$U/$n.sigmf-data" | head -c 112000000 > data_real/iqengine/$n/$n.sigmf-data; curl -s -o data_real/iqengine/$n/$n.sigmf-meta "$U/$n.sigmf-meta"
n=FM-with-SCA-90M9CF-2M4FS-SGU-videos;  mkdir -p data_real/iqengine/$n; curl -s "$U/$n.sigmf-data" | head -c 96000000 > data_real/iqengine/$n/$n.sigmf-data;  curl -s -o data_real/iqengine/$n/$n.sigmf-meta "$U/$n.sigmf-meta"

T=bqXtdp9QsfXLbb3; B=https://cloud.ilabt.imec.be/public.php/webdav
get() { mkdir -p data_real/subghz/$1; curl -s -H "X-Requested-With: XMLHttpRequest" -u "$T:" -o "data_real/subghz/$1/$(basename $2)" "$B/$2"; }
get noise        867.4/Noise/noise_g0.0dB_att10dB_freq867.4MHz_0.mat
get noise        867.4/Noise/noise_g0.0dB_att10dB_freq867.4MHz_1.mat
get lora         867.4/Lora/lorasf7_g0.0dB_att12dB_freq867.4MHz_0.mat
get lora         867.4/Lora/lorasf12_g0.0dB_att24dB_freq867.4MHz_0.mat
get sigfox       867.4/Sigfox/sigfox_g0.0dB_att25dB_freq867.4MHz_1.mat
get ieee802154g  867.4/802154g/zigbee_g0.0dB_att22dB_freq867.4MHz_1.mat
get ieee80211ah  867.4/80211ah/80211ah_mcs0_chan6_g0.0dB_att3dB_freq867.4MHz_0.mat
get sunofdm      864/sunOFDM/sunofdm_O4_mcs2_g0.0dB_att20dB_freq864.0MHz_1.mat
