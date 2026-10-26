### dect6: test = 42 windows in blocks [(31, 44), (91, 104), (136, 149)] of 152

| detector | signals | recall IoU 0.3 (95 % interval) | recall IoU 0.5 | median IoU | boxes / window | false alarms / window | fragments / window |
|---|---|---|---|---|---|---|---|
| Energy, untuned (link (4, 16), 40 cells) | 551 | 0.09 (0.07 to 0.11) | 0.00 | - | 15.3 | 15.3 | 0.00 |
| Energy, link (8, 32), 40 cells chosen on validation (none within budget) | 551 | 0.00 (0.00 to 0.01) | 0.00 | - | 2.5 | 2.5 | 0.00 |
| Energy, fixed link (1, 2), 40 cells (chosen on other windows) | 551 | 0.40 (0.36 to 0.45) | 0.00 | - | 22.4 | 22.4 | 0.00 |
| CA-CFAR, untuned (link (4, 16), 40 cells) | 551 | 0.00 (0.00 to 0.01) | 0.00 | - | 11.0 | 11.0 | 0.00 |
| CA-CFAR, link (8, 32), 40 cells chosen on validation (none within budget) | 551 | 0.00 (0.00 to 0.01) | 0.00 | - | 1.3 | 1.3 | 0.00 |
| CA-CFAR, fixed link (1, 2), 40 cells (chosen on other windows) | 551 | 0.36 (0.32 to 0.40) | 0.00 | - | 21.9 | 21.9 | 0.00 |
| YOLO synthetic (synthetic scenes only), conf 0.05 | 551 | 0.00 (0.00 to 0.01) | 0.00 | - | 1.8 | 1.8 | 0.00 |
| YOLO synthetic (synthetic scenes only), conf 0.05 chosen on validation | 551 | 0.00 (0.00 to 0.01) | 0.00 | - | 1.8 | 1.8 | 0.00 |
| YOLO joint (trained on this recording), conf 0.05 | 551 | 1.00 (0.99 to 1.00) | 0.99 | 0.88 | 14.4 | 0.1 | 1.29 |
| YOLO joint (trained on this recording), conf 0.02 chosen on validation | 551 | 1.00 (0.99 to 1.00) | 0.99 | 0.88 | 14.8 | 0.2 | 1.60 |
| YOLO dect6only (trained on this recording), conf 0.05 | 551 | 1.00 (0.99 to 1.00) | 0.99 | 0.87 | 16.1 | 0.4 | 2.69 |
| YOLO dect6only (trained on this recording), conf 0.02 chosen on validation | 551 | 1.00 (0.99 to 1.00) | 0.99 | 0.87 | 17.2 | 0.6 | 3.57 |
| YOLO bluetoothonly (cross-recording: not trained on this one), conf 0.05 | 551 | 0.00 (0.00 to 0.01) | 0.00 | - | 0.2 | 0.1 | 0.10 |
| YOLO bluetoothonly (cross-recording: not trained on this one), conf 0.02 chosen on validation | 551 | 0.00 (0.00 to 0.01) | 0.00 | - | 0.7 | 0.4 | 0.29 |

### bluetooth: test = 42 windows in blocks [(31, 44), (91, 104), (136, 149)] of 152

| detector | signals | recall IoU 0.3 (95 % interval) | recall IoU 0.5 | median IoU | boxes / window | false alarms / window | fragments / window |
|---|---|---|---|---|---|---|---|
| Energy, untuned (link (4, 16), 40 cells) | 39 | 0.79 (0.64 to 0.89) | 0.59 | 0.68 | 1.0 | 0.4 | 0.00 |
| Energy, link (1, 2), 40 cells chosen on validation | 39 | 1.00 (0.91 to 1.00) | 0.95 | 0.67 | 1.0 | 0.1 | 0.00 |
| Energy, fixed link (1, 2), 40 cells (chosen on other windows) | 39 | 1.00 (0.91 to 1.00) | 0.95 | 0.67 | 1.0 | 0.1 | 0.00 |
| CA-CFAR, untuned (link (4, 16), 40 cells) | 39 | 0.74 (0.59 to 0.85) | 0.41 | 0.67 | 1.0 | 0.6 | 0.00 |
| CA-CFAR, link (1, 2), 40 cells chosen on validation | 39 | 1.00 (0.91 to 1.00) | 0.97 | 0.68 | 1.0 | 0.1 | 0.00 |
| CA-CFAR, fixed link (1, 2), 40 cells (chosen on other windows) | 39 | 1.00 (0.91 to 1.00) | 0.97 | 0.68 | 1.0 | 0.1 | 0.00 |
| YOLO synthetic (synthetic scenes only), conf 0.05 | 39 | 0.15 (0.07 to 0.30) | 0.10 | 0.66 | 1.4 | 1.3 | 0.00 |
| YOLO synthetic (synthetic scenes only), conf 0.05 chosen on validation | 39 | 0.15 (0.07 to 0.30) | 0.10 | 0.66 | 1.4 | 1.3 | 0.00 |
| YOLO joint (trained on this recording), conf 0.05 | 39 | 1.00 (0.91 to 1.00) | 1.00 | 0.87 | 1.0 | 0.0 | 0.00 |
| YOLO joint (trained on this recording), conf 0.02 chosen on validation | 39 | 1.00 (0.91 to 1.00) | 1.00 | 0.87 | 1.1 | 0.1 | 0.05 |
| YOLO dect6only (cross-recording: not trained on this one), conf 0.05 | 39 | 0.00 (0.00 to 0.09) | 0.00 | - | 0.0 | 0.0 | 0.00 |
| YOLO dect6only (cross-recording: not trained on this one), conf 0.02 chosen on validation | 39 | 0.00 (0.00 to 0.09) | 0.00 | - | 0.0 | 0.0 | 0.00 |
| YOLO bluetoothonly (trained on this recording), conf 0.05 | 39 | 1.00 (0.91 to 1.00) | 1.00 | 0.92 | 1.0 | 0.1 | 0.00 |
| YOLO bluetoothonly (trained on this recording), conf 0.02 chosen on validation | 39 | 1.00 (0.91 to 1.00) | 1.00 | 0.92 | 1.0 | 0.1 | 0.00 |
