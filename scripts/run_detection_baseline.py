"""Energy detector and CA-CFAR on wideband scenes: detection baseline.

For every scene: I/Q -> power spectrogram -> detector mask -> boxes, then compare the
boxes with the ground-truth boxes.  Per detector it reports

* recall (probability of detection) overall, per SNR bin, per bandwidth bin and per class
    - ``recall@IoU``  a predicted box matches the signal's box (depends on box forming)
    - ``covered``     >= 30 % of the signal's cells are flagged (detector sensitivity only)
* false alarms per scene (predicted boxes mostly outside every signal) and fragments per
  scene (extra pieces of a signal that did not reach the IoU threshold)
* cell false-alarm rate away from signals (the detector's actual Pfa)
* median localisation error of matched boxes, and run time per scene

Examples
    python scripts/run_detection_baseline.py --source synthetic --n-scenes 40
    python scripts/run_detection_baseline.py --source torchsig --n-scenes 60 --impairment-level 1 --seed 1000 \
        --link 4 16 --min-cells 40
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask, score_mask  # noqa: E402
from vuhf.data.sources import cached, synthetic_source, torchsig_source  # noqa: E402
from vuhf.geometry import SpecGeometry  # noqa: E402
from vuhf.spectrogram import power_spectrogram  # noqa: E402

BW_EDGES = np.array([0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5])


def rate_by_bin(x, flag, edges):
    x, flag = np.asarray(x, float), np.asarray(flag, bool)
    idx = np.digitize(x, edges) - 1
    return np.array([flag[idx == k].mean() if np.any(idx == k) else np.nan for k in range(len(edges) - 1)]), \
           np.array([int(np.sum(idx == k)) for k in range(len(edges) - 1)])


def plot_curves(out, names, data, snr_edges, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    colors = plt.get_cmap("tab10")
    snr_c = 0.5 * (snr_edges[:-1] + snr_edges[1:])
    bw_c = np.sqrt(np.maximum(BW_EDGES[:-1], 1e-3) * BW_EDGES[1:])
    for r, (flag, label) in enumerate([("det@0.3", "box matches (IoU >= 0.3)"), ("covered", "signal covered (>= 30 % of cells)")]):
        for i, n in enumerate(names):
            d = data[n]
            axes[r, 0].plot(snr_c, rate_by_bin(d["snr_db"], d[flag], snr_edges)[0], marker="o", color=colors(i), label=n)
            axes[r, 1].plot(bw_c, rate_by_bin(d["bandwidth"], d[flag], BW_EDGES)[0], marker="o", color=colors(i))
        axes[r, 0].set(ylabel=f"Pd: {label}", ylim=(0, 1.02))
        axes[r, 1].set(xscale="log")
    axes[1, 0].set_xlabel("signal SNR (dB)")
    axes[1, 1].set_xlabel("signal bandwidth (cycles/sample)")
    axes[0, 0].set_title("vs SNR")
    axes[0, 1].set_title("vs bandwidth")
    for a in axes.ravel():
        a.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_example(out, scene, P, preds_by_det, names):
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    db = 10 * np.log10(P / np.median(P) * np.log(2))
    for ax, n in zip(axes, names):
        ax.imshow(db, origin="lower", aspect="auto", extent=(0, scene.iq.shape[0], -0.5, 0.5), vmin=-3, vmax=20, cmap="viridis")
        for b in scene.boxes:
            ax.add_patch(plt.Rectangle((b.t0, b.f0), b.t1 - b.t0, b.f1 - b.f0, fill=False, ec="white", ls="--", lw=1.2))
        for b in preds_by_det[n]:
            ax.add_patch(plt.Rectangle((b.t0, b.f0), b.t1 - b.t0, b.f1 - b.f0, fill=False, ec="red", lw=1.2))
        ax.set_title(n, fontsize=9)
        ax.set_xlabel("time (samples)")
    axes[0].set_ylabel("frequency (cycles/sample)")
    fig.suptitle("white dashed = ground truth, red = detections", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "torchsig"], default="synthetic")
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pfa", type=float, default=1e-3, help="design cell false-alarm probability")
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--link", type=int, nargs=2, default=[4, 16], help="cells (freq, time) within which detections are joined")
    ap.add_argument("--min-cells", type=int, default=40, help="smallest accepted blob, in detected cells")
    ap.add_argument("--impairment-level", type=int, default=0, choices=[0, 1, 2], help="TorchSig: 0 perfect, 1 cabled, 2 wireless")
    ap.add_argument("--snr-db", type=float, nargs=2, default=[0.0, 20.0])
    ap.add_argument("--num-signals", type=int, nargs=2, default=[1, 4])
    ap.add_argument("--n-samples", type=int, default=262144)
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tag = args.source + (f"_imp{args.impairment_level}" if args.source == "torchsig" else "")
    out = args.out or Path("results") / tag
    out.mkdir(parents=True, exist_ok=True)
    key = f"{tag}_snr{args.snr_db[0]:g}-{args.snr_db[1]:g}_seed{args.seed}"
    if args.source == "synthetic":
        factory = lambda: synthetic_source(args.n_scenes, args.seed, n_samples=args.n_samples, n_signals=args.num_signals[1],
                                           snr_db_range=tuple(args.snr_db))
    else:
        factory = lambda: torchsig_source(args.n_scenes, args.seed, impairment_level=args.impairment_level, n_samples=args.n_samples,
                                          num_signals=tuple(args.num_signals), snr_db=tuple(args.snr_db))

    geom = SpecGeometry(args.n_samples, args.n_fft, args.hop)
    post = Post(tuple(args.link), args.min_cells)
    specs = DEFAULT_SPECS
    thresholds = {s.name: calibrate(s, geom, args.pfa) for s in specs}
    names = [s.name for s in specs]
    data = {n: defaultdict(list) for n in names}
    per_scene = {n: dict(fa=[], frag=[], cell_fa=[], cell_pd=[], t=[], n_pred=[]) for n in names}
    errs = {n: defaultdict(list) for n in names}
    csv_rows, example, t_all = [], None, time.time()

    for i, scene in enumerate(cached(factory, key, args.n_scenes, args.cache_dir)):
        P = power_spectrogram(scene.iq, geom)
        preds_by = {}
        for s in specs:
            t0 = time.perf_counter()
            mask = detect_mask(s, P, thresholds[s.name])
            sc = score_mask(scene.boxes, mask, geom, post, iou_thrs=(0.5, 0.3))
            per_scene[s.name]["t"].append(time.perf_counter() - t0)
            preds_by[s.name] = sc["preds"]
            ps = per_scene[s.name]
            ps["fa"].append(sc["n_fa"][0.5]); ps["frag"].append(sc["n_frag"][0.5])
            ps["cell_fa"].append(sc["cell_fa"]); ps["cell_pd"].append(sc["cell_pd"]); ps["n_pred"].append(sc["n_pred"])
            for r in sc["rows"]:
                for k in ("snr_db", "bandwidth", "label", "det@0.5", "det@0.3", "covered"):
                    data[s.name][k].append(r[k])
                if r["det@0.5"]:
                    for k in ("iou", "fc_err", "bw_err", "t_start_err", "t_stop_err"):
                        errs[s.name][k].append(r[k])
                csv_rows.append(dict(detector=s.name, scene=i, label=r["label"], snr_db=r["snr_db"], bandwidth=r["bandwidth"],
                                     duration=r["duration"], det50=int(r["det@0.5"]), det30=int(r["det@0.3"]), covered=int(r["covered"])))
        if example is None and len(scene.boxes) >= 3:
            example = (scene, P, preds_by)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n_scenes} scenes ({time.time() - t_all:.0f}s)", flush=True)

    snr_edges = np.linspace(args.snr_db[0], args.snr_db[1], 9)
    summary = {}
    for n in names:
        d, ps = data[n], per_scene[n]
        med = lambda v: float(np.median(v)) if len(v) else float("nan")
        by_class = defaultdict(lambda: [0, 0, 0])
        for lab, a, b in zip(d["label"], d["det@0.3"], d["covered"]):
            by_class[lab][0] += 1; by_class[lab][1] += int(a); by_class[lab][2] += int(b)
        summary[n] = dict(
            n_signals=len(d["snr_db"]),
            recall_iou50=float(np.mean(d["det@0.5"])), recall_iou30=float(np.mean(d["det@0.3"])), covered=float(np.mean(d["covered"])),
            false_alarms_per_scene=float(np.mean(ps["fa"])), fragments_per_scene=float(np.mean(ps["frag"])),
            predicted_boxes_per_scene=float(np.mean(ps["n_pred"])),
            cell_false_alarm_rate_far_from_signals=float(np.nanmean(ps["cell_fa"])),
            median_iou=med(errs[n]["iou"]), median_fc_err_over_bw=med(errs[n]["fc_err"]), median_bw_err_rel=med(errs[n]["bw_err"]),
            median_t_start_err_rel=med(errs[n]["t_start_err"]), median_t_stop_err_rel=med(errs[n]["t_stop_err"]),
            runtime_ms_per_scene=1000 * float(np.mean(ps["t"])),
            recall_iou30_by_class={k: dict(n=v[0], recall_iou30=v[1] / v[0], covered=v[2] / v[0]) for k, v in sorted(by_class.items())},
        )
    (out / "summary.json").write_text(json.dumps(dict(args={k: str(v) for k, v in vars(args).items()}, design_cell_pfa=args.pfa,
                                                      post=dict(link=post.link, min_cells=post.min_cells), detectors=summary), indent=2))
    with open(out / "per_signal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    plot_curves(out / "pd_curves.png", names, data, snr_edges,
                f"{tag}: {args.n_scenes} scenes, design Pfa {args.pfa:g}, link {post.link}, min cells {post.min_cells}")
    if example is not None:
        sc, P, pb = example
        plot_example(out / "example_scene.png", sc, P, pb, ["energy_median", "energy_q25", "cfar_freq_g24_t48"])

    print(f"\n{tag}: {args.n_scenes} scenes, {summary[names[0]]['n_signals']} signals, spectrogram {geom.shape}, design cell Pfa {args.pfa:g}, "
          f"link {post.link}, min cells {post.min_cells}")
    print(f"{'detector':<20}{'rec@.5':>8}{'rec@.3':>8}{'covered':>9}{'FA/sc':>7}{'frag/sc':>9}{'cell Pfa':>10}{'IoU med':>9}{'ms/sc':>7}")
    for n in names:
        s = summary[n]
        print(f"{n:<20}{s['recall_iou50']:>8.3f}{s['recall_iou30']:>8.3f}{s['covered']:>9.3f}{s['false_alarms_per_scene']:>7.2f}"
              f"{s['fragments_per_scene']:>9.2f}{s['cell_false_alarm_rate_far_from_signals']:>10.1e}{s['median_iou']:>9.2f}{s['runtime_ms_per_scene']:>7.0f}")
    print(f"\nwrote {out}/summary.json, per_signal.csv, pd_curves.png, example_scene.png")


if __name__ == "__main__":
    main()
