#!/usr/bin/env bash
# Retraining YOLO on the labelled real recordings and scoring it on held-out test blocks (2026-10-26).
# Run from the repository root; needs data_real/ (results/real/download_data.sh) and ultralytics. Training is CPU-only here:
# about 65 min for the joint model, 30 min for each single-recording model.
set -e
Y=data_real/yolo_real            # use a SHORT path on Windows (260-character limit), e.g. C:/yr
IQ=data_real/iqengine

# 1. train / validation images (the test blocks are never written as images)
python scripts/make_real_yolo_dataset.py $IQ/dect6/dect6.sigmf-meta         --out $Y/dect6
python scripts/make_real_yolo_dataset.py $IQ/bluetooth/bluetooth.sigmf-meta --out $Y/bluetooth
cp results/yolo/best.pt $Y/init_synthetic.pt
cat > $Y/e1_joint.yaml     <<YAML
path: $PWD/$Y
train: [dect6/images/train, bluetooth/images/train]
val: [dect6/images/val, bluetooth/images/val]
names:
  0: signal
YAML
sed 's#, bluetooth/images/train##; s#, bluetooth/images/val##' $Y/e1_joint.yaml > $Y/e2_dect6.yaml
sed 's#dect6/images/train, ##; s#dect6/images/val, ##'         $Y/e1_joint.yaml > $Y/e3_bluetooth.yaml

# 2. fine-tune the synthetic-trained model (same settings as scripts/train_yolo.py), 30 epochs each
for e in e1_joint e2_dect6 e3_bluetooth; do
  python scripts/train_yolo.py --data $Y/$e.yaml --model $Y/init_synthetic.pt --epochs 30 --project $Y/runs --name $e
done

# 3. score every model on every test block, plus the classical detectors (one extra fixed setting: link (1, 2), 40 cells, chosen on bluetooth windows)
python scripts/eval_real_yolo.py $IQ/dect6/dect6.sigmf-meta $IQ/bluetooth/bluetooth.sigmf-meta \
  --models synthetic=results/yolo/best.pt joint=$Y/runs/e1_joint/weights/best.pt dect6only=$Y/runs/e2_dect6/weights/best.pt bluetoothonly=$Y/runs/e3_bluetooth/weights/best.pt \
  --trained-on joint:dect6,bluetooth dect6only:dect6 bluetoothonly:bluetooth --figure-models synthetic joint --fixed-link 1 2 40 --out results/real_yolo
