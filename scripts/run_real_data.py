"""Run the energy detector, CA-CFAR and YOLO on real I/Q recordings (SigMF, .sdriq, .wav, .mat, raw .cu8) and write
predictions, a summary and a picture.  If the recording has box annotations (SigMF annotations, or --annotations-csv) the
detectors are also scored; if not, only counts and figures are produced.

    python scripts/run_real_data.py capture.sigmf-meta --out results/real/capture
    python scripts/run_real_data.py snapshot.mat --fs 2.048e6 --fc 864e6 --window 65536 --out results/real/subghz
    python scripts/run_real_data.py silence.cu8 --fs 2.4e6 --fc 150e6 --noise-only --out results/real/noise

Detectors and settings are those of the synthetic evaluation (results/torchsig_imp*): energy_median (3 x 9 averaging) and
cfar_freq_g24_t48, both calibrated for a cell false-alarm probability of 1e-3 on noise, then the validation-chosen box linking
(4, 16) with at least 40 cells; YOLO = results/yolo/best.pt at confidence 0.05.  Frequency is in cycles/sample inside the
detectors; the outputs are converted to seconds and absolute Hz (centre frequency + f * sample rate).

--annotations-csv: columns file,t_start_s,t_stop_s,f_low_hz,f_high_hz,label  (file = the recording's file name; times from its start).
Annotations of public recordings are often incomplete: a detection that matches no annotation can be a real, unlabelled signal.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask, score_boxes  # noqa: E402
from vuhf.data.real import Annotation, iter_windows, open_recording, window_boxes  # noqa: E402
from vuhf.detectors import mask_to_boxes  # noqa: E402
from vuhf.detectors.boxes import boxes_to_mask  # noqa: E402
from vuhf.geometry import SpecGeometry  # noqa: E402
from vuhf.spectrogram import power_spectrogram  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DETECTORS = [("energy_median", "Energy detector", "#2a78d6"), ("cfar_freq_g24_t48", "CA-CFAR", "#eb6834"), ("yolo", "YOLO", "#1baf7a")]
MIN_SAMPLES = 4096


def load_annotations_csv(path: Path, rec) -> list[Annotation]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["file"] not in (rec.path.name, "*"):
                continue
            s = int(round(float(r["t_start_s"]) * rec.fs))
            out.append(Annotation(s, int(round((float(r["t_stop_s"]) - float(r["t_start_s"])) * rec.fs)),
                                  float(r["f_low_hz"]), float(r["f_high_hz"]), r.get("label", "")))
    return out


def plot_windows(chosen, names, path, title):
    n = len(chosen)
    fig, axes = plt.subplots(n, len(names), figsize=(4.6 * len(names), 2.7 * n + 1.0), squeeze=False)
    halo = [pe.withStroke(linewidth=3.2, foreground="black")]
    for r, w in enumerate(chosen):
        P, fs, fc = w["P"], w["fs"], w["fc"]
        db = 10 * np.log10(np.maximum(P, 1e-12) / (np.median(P) / np.log(2)))
        t_ms = w["n"] / fs * 1e3
        f_lo, f_hi = (fc - fs / 2) / 1e6, (fc + fs / 2) / 1e6
        for c, (name, label, color) in enumerate(names):
            ax = axes[r, c]
            ax.imshow(np.clip(db, -3, 20), origin="lower", aspect="auto", cmap="gray", vmin=-3, vmax=24, extent=(0, t_ms, f_lo, f_hi))
            for g in w["gts"]:
                ax.add_patch(plt.Rectangle((g.t0 / fs * 1e3, fc / 1e6 + g.f0 * fs / 1e6), g.duration / fs * 1e3, g.bandwidth * fs / 1e6,
                                           fill=False, ec="white", ls="--", lw=1.3, path_effects=halo))
            for b in w["preds"][name]:
                ax.add_patch(plt.Rectangle((b.t0 / fs * 1e3, fc / 1e6 + b.f0 * fs / 1e6), b.duration / fs * 1e3, b.bandwidth * fs / 1e6,
                                           fill=False, ec=color, ls="-", lw=2.0, path_effects=halo))
            ax.text(0.01, 0.03, f"{len(w['preds'][name])} boxes", transform=ax.transAxes, fontsize=9.5, color="white", va="bottom",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])
            if r == 0:
                ax.set_title(label, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{w['name']}\nwindow @ {w['start'] / fs:.3f} s\nfrequency (MHz)", fontsize=8.5)
            if r == n - 1:
                ax.set_xlabel("time in window (ms)")
    fig.suptitle(title + "   (white dashed = annotation, if any; solid = detection)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=105, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--fs", type=float, help="sample rate in Hz (needed for raw .cu8/.cs8/.cf32 and .mat files without one)")
    ap.add_argument("--fc", type=float, help="centre frequency in Hz")
    ap.add_argument("--format", help="force the format: sigmf, sdriq, wav, mat, cu8, cs8, cf32")
    ap.add_argument("--window", type=int, default=262144, help="samples per detection window (the synthetic scenes used 262144)")
    ap.add_argument("--hop", type=int, default=None, help="window hop in samples (default = window, no overlap)")
    ap.add_argument("--max-windows", type=int, default=50, help="per file")
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--yolo-weights", type=Path, default=ROOT / "results/yolo/best.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.05)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--noise-only", action="store_true", help="the recordings contain no signal: every box counts as a false alarm")
    ap.add_argument("--annotations-csv", type=Path)
    ap.add_argument("--n-figures", type=int, default=6, help="windows drawn in the figure (random, fixed seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/real/run")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args.out.mkdir(parents=True, exist_ok=True)

    names = [d for d in DETECTORS if not (d[0] == "yolo" and args.no_yolo)]
    specs = {s.name: s for s in DEFAULT_SPECS}
    post = Post()
    rng = np.random.default_rng(args.seed)
    thr_cache, yolo_cache = {}, {}
    stats = {n: defaultdict(list) for n, _, _ in names}
    rows_out, chosen, seen, t_start = [], [], 0, time.time()
    any_ann = False

    for path in args.inputs:
        rec = open_recording(path, fs=args.fs, fc=args.fc, fmt=args.format)
        if args.annotations_csv:
            rec.annotations = load_annotations_csv(args.annotations_csv, rec)
        has_ann = bool(rec.annotations) or args.noise_only
        any_ann |= has_ann
        print(f"{path.name}: {rec.fmt}, fs {rec.fs:g} Hz, fc {rec.fc:g} Hz, {rec.n_samples} samples ({rec.n_samples / rec.fs:.3f} s), "
              f"{len(rec.annotations)} annotations", flush=True)
        for start, iq in iter_windows(rec, args.window, args.hop, args.max_windows):
            if len(iq) < MIN_SAMPLES:
                continue
            iq = np.nan_to_num(iq)
            n = len(iq)
            geom = SpecGeometry(n, 512, 256)
            if n not in thr_cache:
                thr_cache[n] = {k: calibrate(specs[k], geom, args.pfa) for k in ("energy_median", "cfar_freq_g24_t48")}
                if not args.no_yolo:
                    from vuhf.detectors.yolo import YoloDetector

                    yolo_cache[n] = YoloDetector(args.yolo_weights, geom, min_conf=args.yolo_conf)
            P = power_spectrogram(iq, geom)
            preds, conf = {}, {}
            for k in ("energy_median", "cfar_freq_g24_t48"):
                preds[k] = mask_to_boxes(detect_mask(specs[k], P, thr_cache[n][k]), geom, link=post.link, min_cells=post.min_cells)
            if not args.no_yolo:
                raw = [(b, c) for b, c in yolo_cache[n].predict_raw([P])[0] if c >= args.yolo_conf]
                preds["yolo"], conf["yolo"] = [b for b, _ in raw], [c for _, c in raw]
            gts = window_boxes(rec, start, n) if rec.annotations else []
            for name, _, _ in names:
                st, pb = stats[name], preds[name]
                st["n_boxes"].append(len(pb))
                st["occupancy"].append(float(boxes_to_mask(pb, geom).mean()))
                if has_ann:
                    sc = score_boxes(gts, pb, boxes_to_mask(pb, geom), geom, iou_thrs=(0.5, 0.3))
                    st["fa"].append(sc["n_fa"][0.5]); st["frag"].append(sc["n_frag"][0.5])
                    for r in sc["rows"]:
                        st["det50"].append(r["det@0.5"]); st["det30"].append(r["det@0.3"])
                        if r["det@0.5"]:
                            st["iou"].append(r["iou"])
                for i, b in enumerate(pb):
                    rows_out.append(dict(file=path.name, window_start_sample=start, detector=name,
                                         t_start_s=(start + b.t0) / rec.fs, t_stop_s=(start + b.t1) / rec.fs,
                                         f_low_hz=rec.fc + b.f0 * rec.fs, f_high_hz=rec.fc + b.f1 * rec.fs,
                                         confidence=conf.get(name, [None] * len(pb))[i] if name in conf else ""))
            seen += 1                                                       # reservoir sample of windows for the figure
            item = dict(P=P.astype(np.float32), preds=preds, gts=gts, fs=rec.fs, fc=rec.fc, n=n, start=start, name=path.name)
            if len(chosen) < args.n_figures:
                chosen.append(item)
            elif (j := rng.integers(0, seen)) < args.n_figures:
                chosen[j] = item
            if seen % 10 == 0:
                print(f"  {seen} windows ({time.time() - t_start:.0f}s)", flush=True)

    if not seen:
        raise SystemExit("no window was processed (recordings shorter than 4096 samples?)")
    summary = dict(args={k: str(v) for k, v in vars(args).items()}, n_windows=seen, annotated=any_ann, detectors={})
    for name, _, _ in names:
        st = stats[name]
        d = dict(boxes_per_window=float(np.mean(st["n_boxes"])), windows_with_a_box=float(np.mean(np.array(st["n_boxes"]) > 0)),
                 mean_share_of_spectrogram_inside_boxes=float(np.mean(st["occupancy"])))
        if any_ann and st["fa"]:
            d.update(false_alarms_per_window=float(np.mean(st["fa"])), fragments_per_window=float(np.mean(st["frag"])))
            if st["det50"]:
                d.update(n_annotated_signals=len(st["det50"]), recall_iou50=float(np.mean(st["det50"])),
                         recall_iou30=float(np.mean(st["det30"])), median_iou=float(np.median(st["iou"])) if st["iou"] else None)
        summary["detectors"][name] = d
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    if rows_out:
        with open(args.out / "predictions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    plot_windows(sorted(chosen, key=lambda w: (w["name"], w["start"])), names, args.out / "windows.png",
                 f"{', '.join(p.name for p in args.inputs[:3])}{' ...' if len(args.inputs) > 3 else ''}")

    print(f"\n{seen} windows; scored against annotations: {any_ann}")
    for name, label, _ in names:
        print(f"  {label:<16}" + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in summary['detectors'][name].items()))
    print(f"wrote {args.out}/summary.json, predictions.csv, windows.png")


if __name__ == "__main__":
    main()
