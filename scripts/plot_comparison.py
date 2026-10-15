"""Visual comparison of the energy detector, CA-CFAR and YOLO on the same test scenes.

Writes two figures to ``--out-dir``:

* ``comparison_scenes_imp<L>.png``  randomly chosen scenes (fixed ``--pick-seed``, not hand-picked), one row per
  scene, one column per detector.  White dashed = true signal, solid colour = detection that matches a true
  signal (IoU >= 0.3), dotted colour = detection that matches nothing (false alarm or fragment).
* ``comparison_scenes_overlap_imp<L>.png``  the same, but drawn at random from the scenes that contain overlapping
  signals (the case the first gallery rarely shows).
* ``comparison_summary.png``  all test scenes of all impairment levels pooled: recall against SNR, against
  bandwidth, for isolated against overlapping signals, and recall against false alarms per scene.
  The YOLO sweep in the last panel uses the test scenes only to draw the curve; the threshold used everywhere
  else (0.05) was chosen on validation scenes.

    python scripts/plot_comparison.py --yolo-weights results/yolo/best.pt --yolo-conf 0.05
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask  # noqa: E402
from vuhf.data.sources import cached  # noqa: E402
from vuhf.detectors import mask_to_boxes  # noqa: E402
from vuhf.eval.matching import evaluate_scene, match_boxes  # noqa: E402
from vuhf.geometry import SpecGeometry  # noqa: E402
from vuhf.spectrogram import power_spectrogram  # noqa: E402

# fixed order = fixed colour per detector (first three slots of the validated categorical palette)
DETECTORS = [("energy_median", "Energy detector", "#2a78d6"), ("cfar_freq_g24_t48", "CA-CFAR", "#eb6834"), ("yolo", "YOLO", "#1baf7a")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd7"
FS = 10e6
CONF_SWEEP = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85]


def touches_other(boxes, j, freq_gap=0.01):
    g = boxes[j]
    return any(k != j and min(g.t1, h.t1) - max(g.t0, h.t0) > 0 and min(g.f1, h.f1) - max(g.f0, h.f0) > -freq_gap
               for k, h in enumerate(boxes))


def collect(args, geom):
    from vuhf.detectors.yolo import YoloDetector

    specs = {s.name: s for s in DEFAULT_SPECS}
    thr = {n: calibrate(specs[n], geom, 1e-3) for n in ("energy_median", "cfar_freq_g24_t48")}
    yolo = YoloDetector(args.yolo_weights, geom, min_conf=min(CONF_SWEEP))
    post = Post()
    out = dict(rows=[], scenes=[], yolo_raw=[])
    for lv in args.levels:
        key = f"torchsig_imp{lv}_snr0-20_seed{args.seed}"
        for i, sc in enumerate(cached(None, key, args.n_scenes, args.cache_dir)):
            P = power_spectrogram(sc.iq, geom)
            raw = yolo.predict_raw([P])[0]
            preds = {"yolo": [b for b, c in raw if c >= args.yolo_conf]}
            for n in thr:
                preds[n] = mask_to_boxes(detect_mask(specs[n], P, thr[n]), geom, link=post.link, min_cells=post.min_cells)
            gts = sc.boxes
            matched = {n: {j for _, j, _ in match_boxes(preds[n], gts, 0.3)} for n, _, _ in DETECTORS}
            for j, g in enumerate(gts):
                out["rows"].append(dict(level=lv, snr=g.snr_db, bw=g.bandwidth, touch=touches_other(gts, j),
                                        **{n: j in matched[n] for n, _, _ in DETECTORS}))
            out["yolo_raw"].append((lv, gts, raw))
            out["fa"] = out.get("fa", {n: [] for n, _, _ in DETECTORS})
            for n, _, _ in DETECTORS:
                out["fa"][n].append(evaluate_scene(preds[n], gts, 0.3).n_false_alarms)
            if lv == args.scene_level:
                out["scenes"].append(dict(index=i, level=lv, gts=gts, preds=preds,
                                          overlap=any(touches_other(gts, j) for j in range(len(gts)))))
        print(f"level {lv} done", flush=True)
    return out


def style(ax):
    ax.set_facecolor("white")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4af")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def pick(out, args, overlap):
    pool = [s for s in out["scenes"] if s["overlap"] == overlap] if overlap else list(out["scenes"])
    rng = np.random.default_rng(args.pick_seed)
    idx = sorted(rng.choice(len(pool), size=min(args.n_show, len(pool)), replace=False).tolist())
    chosen = [pool[k] for k in idx]
    want = {s["index"] for s in chosen}
    key = f"torchsig_imp{args.scene_level}_snr0-20_seed{args.seed}"
    geom = SpecGeometry(262144, 512, 256)
    for i, sc in enumerate(cached(None, key, args.n_scenes, args.cache_dir)):
        if i in want:
            next(s for s in chosen if s["index"] == i)["P"] = power_spectrogram(sc.iq, geom).astype(np.float32)
    return chosen


def plot_scenes(sc_list, geom, path, title):
    sc_list = sorted(sc_list, key=lambda s: s["index"])
    n = len(sc_list)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 2.7 * n + 1.0), sharex=True, sharey=True, squeeze=False)
    t_ms = geom.n_samples / FS * 1e3
    halo = [pe.withStroke(linewidth=3.2, foreground="black")]
    for r, sc in enumerate(sc_list):
        P = sc["P"]
        db = 10 * np.log10(P / (np.median(P) / np.log(2)))
        gts = sc["gts"]
        for c, (name, label, color) in enumerate(DETECTORS):
            ax = axes[r, c]
            ax.imshow(np.clip(db, -3, 20), origin="lower", aspect="auto", cmap="gray", vmin=-3, vmax=24,
                      extent=(0, t_ms, -FS / 2e6, FS / 2e6))
            for g in gts:
                ax.add_patch(plt.Rectangle((g.t0 / FS * 1e3, g.f0 * FS / 1e6), g.duration / FS * 1e3, g.bandwidth * FS / 1e6, fill=False,
                                           ec="white", ls="--", lw=1.3, path_effects=halo))
            preds = sc["preds"][name]
            ok = {i for i, _, _ in match_boxes(preds, gts, 0.3)}
            found = len({j for _, j, _ in match_boxes(preds, gts, 0.3)})
            for i, b in enumerate(preds):
                ax.add_patch(plt.Rectangle((b.t0 / FS * 1e3, b.f0 * FS / 1e6), b.duration / FS * 1e3, b.bandwidth * FS / 1e6, fill=False,
                                           ec=color, ls="-" if i in ok else ":", lw=2.0, path_effects=halo))
            extra = len(preds) - len(ok)
            ax.text(0.01, 0.03, f"found {found} of {len(gts)}   extra boxes {extra}", transform=ax.transAxes, fontsize=9.5, color="white",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")], va="bottom")
            if r == 0:
                ax.set_title(label, fontsize=12, color=color if False else INK, fontweight="bold", pad=8)
                ax.plot([], [], color=color, lw=3, label=label)
            if c == 0:
                ax.set_ylabel(f"scene {sc['index']}\nfrequency (MHz)", fontsize=9.5)
            ax.tick_params(labelsize=8.5)
            if r == n - 1:
                ax.set_xlabel("time (ms)", fontsize=9.5)
    handles = [plt.Line2D([], [], color="white", ls="--", lw=1.6, path_effects=halo),
               plt.Line2D([], [], color="#888", lw=2),
               plt.Line2D([], [], color="#888", lw=2, ls=":")]
    fig.legend(handles, ["true signal (ground truth)", "detection that matches a true signal (IoU >= 0.3)",
                         "detection that matches nothing (false alarm or fragment)"], loc="upper center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.03, fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for c, (name, label, color) in enumerate(DETECTORS):  # coloured bar above each column = the detector's identity colour
        pos = axes[0, c].get_position()
        fig.add_artist(plt.Line2D([pos.x0, pos.x1], [pos.y1 + 0.032] * 2, color=color, lw=4, transform=fig.transFigure))
    fig.savefig(path, dpi=105, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def recall_by(rows, key, edges, name):
    v = np.array([r[key] for r in rows], float)
    d = np.array([r[name] for r in rows], bool)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi)
        out.append(d[m].mean() if m.any() else np.nan)
    return np.array(out)


def plot_summary(out, args, geom, path):
    from vuhf.baseline import score_boxes
    from vuhf.detectors.boxes import boxes_to_mask

    rows = out["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    ax = axes[0, 0]
    edges = [0, 5, 10, 15, 20.01]
    labels = ["0 to 5", "5 to 10", "10 to 15", "15 to 20"]
    x = np.arange(4)
    for name, label, color in DETECTORS:
        y = recall_by(rows, "snr", edges, name)
        ax.plot(x, y, marker="o", ms=7, lw=2, color=color, label=label, mfc=color, mec="white", mew=1.5)
    ax.set_xticks(x, labels)
    ax.set_xlabel("signal SNR (dB)")
    ax.set_ylabel("share of signals found (IoU >= 0.3)")
    ax.set_title("Weak signals: YOLO holds up at low SNR", loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, 1.02)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[0, 1]
    touch = np.array([r["touch"] for r in rows], bool)
    w = 0.26
    for k, (name, label, color) in enumerate(DETECTORS):
        d = np.array([r[name] for r in rows], bool)
        vals = [d[~touch].mean(), d[touch].mean()]
        bars = ax.bar(np.arange(2) + (k - 1) * (w + 0.02), vals, w, color=color, label=label, zorder=3, edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1], [f"signal on its own\n(n = {int((~touch).sum())})", f"overlaps a neighbouring signal\n(n = {int(touch.sum())})"])
    ax.set_ylabel("share of signals found (IoU >= 0.3)")
    ax.set_title("Overlap is where the classical detectors fail", loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, 1.2)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)

    ax = axes[1, 0]
    edges = [0, 0.02, 0.05, 0.1, 0.51]
    labels = ["< 0.2", "0.2 to 0.5", "0.5 to 1", "> 1"]
    for name, label, color in DETECTORS:
        y = recall_by(rows, "bw", edges, name)
        ax.plot(x, y, marker="o", ms=7, lw=2, color=color, label=label, mfc=color, mec="white", mew=1.5)
    ax.set_xticks(x, labels)
    ax.set_xlabel("signal bandwidth (MHz, at 10 MS/s)")
    ax.set_ylabel("share of signals found (IoU >= 0.3)")
    ax.set_title("Narrow signals are hard for everyone", loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, 1.02)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1, 1]
    n_scenes = len(out["yolo_raw"])
    pts = []
    for conf in CONF_SWEEP:
        rec, fa = [], []
        for lv, gts, raw in out["yolo_raw"]:
            preds = [b for b, c in raw if c >= conf]
            res = evaluate_scene(preds, gts, 0.3)
            rec += [g.detected for g in res.gt]
            fa.append(res.n_false_alarms)
        pts.append((np.mean(fa), np.mean(rec)))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=DETECTORS[2][2], lw=2, marker="o", ms=5, label="YOLO, confidence threshold swept")
    i05 = CONF_SWEEP.index(args.yolo_conf)
    ax.plot(*pts[i05], marker="o", ms=11, mfc=DETECTORS[2][2], mec=INK, mew=1.8, ls="none", label=f"YOLO as evaluated (confidence {args.yolo_conf})")
    for name, label, color in DETECTORS[:2]:
        d = np.array([r[name] for r in rows], bool)
        ax.plot(np.mean(out["fa"][name]), d.mean(), marker="s", ms=10, color=color, mec="white", mew=1.5, ls="none", label=f"{label} (design Pfa 1e-3)")
    ax.set_xlabel("false-alarm boxes per scene")
    ax.set_ylabel("share of signals found (IoU >= 0.3)")
    ax.set_title("Same false-alarm level, far more signals found", loc="left", fontsize=11, color=INK)
    ax.set_xlim(left=0)
    ax.set_ylim(0.5, 1.0)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle(f"Energy, CA-CFAR and YOLO on the same {n_scenes} test scenes ({len(rows)} signals, three impairment levels pooled)",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--yolo-conf", type=float, default=0.05)
    ap.add_argument("--n-scenes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--scene-level", type=int, default=1, help="impairment level of the scene gallery")
    ap.add_argument("--n-show", type=int, default=6)
    ap.add_argument("--pick-seed", type=int, default=0, help="seed of the random choice of gallery scenes")
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--reuse", action="store_true", help="reuse the detections of the last run (data_cache/compare.pkl)")
    args = ap.parse_args()

    geom = SpecGeometry(262144, 512, 256)
    pkl = args.cache_dir / "compare.pkl"
    if args.reuse and pkl.exists():
        out = pickle.loads(pkl.read_bytes())
    else:
        out = collect(args, geom)
        pkl.write_bytes(pickle.dumps(out))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lv = args.scene_level
    plot_scenes(pick(out, args, False), geom, args.out_dir / f"comparison_scenes_imp{lv}.png",
                f"Same test scenes, three detectors (impairment level {lv}, scenes drawn at random)")
    plot_scenes(pick(out, args, True), geom, args.out_dir / f"comparison_scenes_overlap_imp{lv}.png",
                f"Scenes with overlapping signals (impairment level {lv}, drawn at random among the {sum(s['overlap'] for s in out['scenes'])} such scenes)")
    plot_summary(out, args, geom, args.out_dir / "comparison_summary.png")
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
