"""Collect results/real/*/summary.json into markdown tables (results/real/summary_tables.md).

    python scripts/summarize_real.py

Three tables: recordings with annotations (scored), noise-only recordings (every box is a false alarm) and recordings without
labels (counts only).  Nothing is computed here beyond what run_real_data.py wrote.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"energy_median": "energy", "cfar_freq_g24_t48": "CA-CFAR", "yolo": "YOLO"}


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    runs = []
    for p in sorted((ROOT / "results/real").glob("*/summary.json")):
        s = json.loads(p.read_text())
        a = s["args"]
        runs.append(dict(name=p.parent.name, s=s, noise=a.get("noise_only") == "True", window=a["window"], dc=a.get("mask_dc", "0")))
    out = []

    out += ["## Recordings with annotations (scored)", "",
            "| run | detector | windows | signals | recall IoU 0.3 | recall IoU 0.5 | median IoU | covered >= 30 % | boxes / window | false alarms / window | fragments / window |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if r["noise"] or not r["s"]["annotated"]:
            continue
        for k, d in r["s"]["detectors"].items():
            out.append(f"| {r['name']} | {NAMES[k]} | {r['s']['n_windows']} | {d.get('n_annotated_signals', '-')} | {fmt(d.get('recall_iou30'))} | "
                       f"{fmt(d.get('recall_iou50'))} | {fmt(d.get('median_iou'))} | {fmt(d.get('covered_ge30'))} | {fmt(d['boxes_per_window'], 1)} | "
                       f"{fmt(d.get('false_alarms_per_window'), 1)} | {fmt(d.get('fragments_per_window'), 2)} |")

    out += ["", "## Noise-only recordings (every box is a false alarm)", "",
            "| run | window | DC bins masked | detector | windows | windows with a box | boxes / window | share of spectrogram inside boxes |",
            "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if not r["noise"]:
            continue
        for k, d in r["s"]["detectors"].items():
            out.append(f"| {r['name']} | {r['window']} | {2 * int(r['dc']) + 1 if int(r['dc']) else 0} | {NAMES[k]} | {r['s']['n_windows']} | "
                       f"{fmt(d['windows_with_a_box'])} | {fmt(d['boxes_per_window'], 1)} | {fmt(d['mean_share_of_spectrogram_inside_boxes'])} |")

    out += ["", "## Recordings without labels (counts only; judge the pictures, not the numbers)", "",
            "| run | window | DC bins masked | detector | windows | windows with a box | boxes / window | share of spectrogram inside boxes |",
            "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if r["noise"] or r["s"]["annotated"]:
            continue
        for k, d in r["s"]["detectors"].items():
            out.append(f"| {r['name']} | {r['window']} | {2 * int(r['dc']) + 1 if int(r['dc']) else 0} | {NAMES[k]} | {r['s']['n_windows']} | "
                       f"{fmt(d['windows_with_a_box'])} | {fmt(d['boxes_per_window'], 1)} | {fmt(d['mean_share_of_spectrogram_inside_boxes'])} |")
    (ROOT / "results/real/summary_tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
