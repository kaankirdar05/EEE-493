"""Score YOLO weights and the classical detectors on the TEST block of annotated real recordings.

    python scripts/eval_real_yolo.py data_real/iqengine/dect6/dect6.sigmf-meta data_real/iqengine/bluetooth/bluetooth.sigmf-meta \\
        --models synthetic=results/yolo/best.pt joint=C:/yr/runs/e1_joint/weights/best.pt --out results/real_yolo

Each recording is split in time by ``block_split`` (interleaved train / validation / test blocks with a gap window between them;
see make_real_yolo_dataset.py for the train images).  Everything here reads only the validation and test blocks:

* the validation block picks, with the selection rule of tune_real.py (highest recall at IoU 0.3 among the settings with at most
  ``--budget`` false alarms per window, false alarms counted at IoU 0.3), the YOLO confidence of each model and the box linking
  shared by the two classical detectors (energy and CA-CFAR, ranked by the mean of their recalls).  If no setting is within the budget
  the one with the fewest false alarms is used and the table says so;
* the test block is scored once: classical detectors untuned (link (4, 16), 40 cells) and with the linking chosen on the validation
  block, every model at the validation-chosen confidence and at the default 0.05.  A model that was not trained on a recording is
  also scored on it (cross-recording); its confidence is then chosen on the validation block of the recording it is scored on.

The tables report false alarms at IoU 0.5 (as the real-data tables of the README); the 95 % interval on recall is Wilson's, over signals
(windows of one recording are not independent, so it is optimistic).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_data import plot_windows  # noqa: E402

from vuhf.baseline import DEFAULT_SPECS, Post, calibrate, detect_mask, score_boxes  # noqa: E402
from vuhf.data.real import block_split, open_recording, runs, window_boxes  # noqa: E402
from vuhf.detectors import mask_to_boxes  # noqa: E402
from vuhf.detectors.boxes import boxes_to_mask  # noqa: E402
from vuhf.detectors.yolo import YoloDetector  # noqa: E402
from vuhf.geometry import SpecGeometry  # noqa: E402
from vuhf.spectrogram import power_spectrogram  # noqa: E402

W = 262144
CLASSICAL = ("energy_median", "cfar_freq_g24_t48")
LINKS = [(1, 2), (2, 4), (2, 6), (4, 16), (8, 32)]
MIN_CELLS = [12, 40]
CONFS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LABELS = {"energy_median": "Energy (tuned link)", "cfar_freq_g24_t48": "CA-CFAR (tuned link)"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def score(wins, preds, geom) -> dict:
    sc = [score_boxes(w["gts"], p, boxes_to_mask(p, geom), geom, iou_thrs=(0.5, 0.3)) for w, p in zip(wins, preds)]
    rows = [r for s in sc for r in s["rows"]]
    d30, d50 = [r["det@0.3"] for r in rows], [r["det@0.5"] for r in rows]
    lo, hi = wilson(int(sum(d30)), len(d30))
    return dict(n_signals=len(rows), recall30=float(np.mean(d30)) if d30 else None, recall30_ci=[lo, hi],
                recall50=float(np.mean(d50)) if d50 else None, median_iou=float(np.median([r["iou"] for r in rows if r["det@0.5"]])) if any(d50) else None,
                fa30=float(np.mean([s["n_fa"][0.3] for s in sc])), fa50=float(np.mean([s["n_fa"][0.5] for s in sc])),
                frag50=float(np.mean([s["n_frag"][0.5] for s in sc])), boxes=float(np.mean([len(p) for p in preds])))


def pick(cands: list[tuple[object, float, float]], budget: float):
    """cands = (setting, recall30, false alarms): best recall within the budget, else the fewest false alarms."""
    ok = [c for c in cands if c[2] <= budget]
    if ok:
        return max(ok, key=lambda c: c[1])[0], True
    return min(cands, key=lambda c: (c[2], -c[1]))[0], False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--models", nargs="+", required=True, help="name=weights.pt (the first one is the reference, e.g. synthetic=results/yolo/best.pt)")
    ap.add_argument("--trained-on", nargs="*", default=[], help="name:rec1,rec2  which recordings (file stems) a model was trained on, for the table")
    ap.add_argument("--figure-models", nargs="*", default=None, help="models drawn in the figures (default: all)")
    ap.add_argument("--fixed-link", nargs=3, type=int, metavar=("DF", "DT", "MIN_CELLS"), default=None,
                    help="also score the classical detectors on every test block with this fixed setting (for example the one chosen on another recording)")
    ap.add_argument("--budget", type=float, default=2.0, help="false alarms per validation window allowed when picking a setting")
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--n-figures", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args.out.mkdir(parents=True, exist_ok=True)

    models = dict(m.split("=", 1) for m in args.models)
    trained = {m.split(":")[0]: set(m.split(":")[1].split(",")) for m in args.trained_on}
    geom = SpecGeometry(W, 512, 256)
    specs = {s.name: s for s in DEFAULT_SPECS}
    thr = {k: calibrate(specs[k], geom, args.pfa) for k in CLASSICAL}
    dets = {n: YoloDetector(p, geom, min_conf=min(CONFS)) for n, p in models.items()}
    results, rng = {}, np.random.default_rng(0)

    for path in args.recordings:
        rec = open_recording(path)
        n_all = (rec.n_samples - W) // W + 1
        sp = block_split(n_all)
        data = {}
        for split in ("val", "test"):
            idx = sp[split]
            ws = []
            for i in idx:
                s = i * W
                P = power_spectrogram(np.nan_to_num(rec.read(s, W)), geom)
                ws.append(dict(start=s, P=P.astype(np.float32), gts=window_boxes(rec, s, W),
                               masks={k: detect_mask(specs[k], P, thr[k]) for k in CLASSICAL},
                               yolo={n: d.predict_raw([P])[0] for n, d in dets.items()}))
            data[split] = ws
        stem = path.stem.split(".")[0]
        print(f"{stem}: {n_all} windows; val {len(sp['val'])} windows / {sum(len(w['gts']) for w in data['val'])} signals, "
              f"test {len(sp['test'])} windows / {sum(len(w['gts']) for w in data['test'])} signals", flush=True)
        res = dict(split=dict(n_windows=n_all, **{k: runs(v) for k, v in sp.items()}, n_val_signals=sum(len(w["gts"]) for w in data["val"]),
                              n_test_signals=sum(len(w["gts"]) for w in data["test"])), classical={}, yolo={})

        def boxes_of(ws, k, post):
            return [mask_to_boxes(w["masks"][k], geom, link=post.link, min_cells=post.min_cells) for w in ws]

        # classical detectors: untuned, and with the linking chosen on the validation block
        cands = []
        for link, mc in itertools.product(LINKS, MIN_CELLS):
            r = {k: score(data["val"], boxes_of(data["val"], k, Post(link, mc)), geom) for k in CLASSICAL}
            cands.append(((link, mc), float(np.mean([r[k]["recall30"] for k in CLASSICAL])), float(max(r[k]["fa30"] for k in CLASSICAL))))
        (link, mc), feasible = pick(cands, args.budget)
        res["classical_choice"] = dict(link=list(link), min_cells=mc, within_budget=feasible, val_grid=[dict(link=list(c[0][0]), min_cells=c[0][1], mean_recall30=c[1], max_fa30=c[2]) for c in cands])
        for k in CLASSICAL:
            res["classical"][k] = dict(untuned=score(data["test"], boxes_of(data["test"], k, Post()), geom),
                                       tuned=score(data["test"], boxes_of(data["test"], k, Post(link, mc)), geom))
            if args.fixed_link:
                res["classical"][k]["fixed"] = score(data["test"], boxes_of(data["test"], k, Post((args.fixed_link[0], args.fixed_link[1]), args.fixed_link[2])), geom)

        # YOLO models: confidence chosen on the validation block, scored on the test block
        for n in models:
            cands = []
            for c in CONFS:
                r = score(data["val"], [[b for b, cf in w["yolo"][n] if cf >= c] for w in data["val"]], geom)
                cands.append((c, r["recall30"], r["fa30"]))
            conf, feasible = pick(cands, args.budget)
            res["yolo"][n] = dict(trained_on_this=stem in trained.get(n, set()), chosen_conf=conf, within_budget=feasible,
                                  val_sweep=[dict(conf=c[0], recall30=c[1], fa30=c[2]) for c in cands],
                                  test_at_chosen=score(data["test"], [[b for b, cf in w["yolo"][n] if cf >= conf] for w in data["test"]], geom),
                                  test_at_005=score(data["test"], [[b for b, cf in w["yolo"][n] if cf >= 0.05] for w in data["test"]], geom))
        results[stem] = res

        # figure: a few test windows
        names = [(k, LABELS[k], c) for k, c in zip(CLASSICAL, ("#2a78d6", "#eb6834"))]
        fig_models = [m for m in (args.figure_models if args.figure_models is not None else list(models))]
        palette = ["#1baf7a", "#9b59b6", "#c0392b", "#7f8c8d"]
        names += [(f"yolo:{m}", f"YOLO {m} (conf {res['yolo'][m]['chosen_conf']:g})", palette[i % len(palette)]) for i, m in enumerate(fig_models)]
        pick_idx = sorted(rng.choice(len(data["test"]), size=min(args.n_figures, len(data["test"])), replace=False).tolist())
        chosen = []
        for i in pick_idx:
            w = data["test"][i]
            preds = {k: mask_to_boxes(w["masks"][k], geom, link=Post(link, mc).link, min_cells=Post(link, mc).min_cells) for k in CLASSICAL}
            for m in fig_models:
                preds[f"yolo:{m}"] = [b for b, cf in w["yolo"][m] if cf >= res["yolo"][m]["chosen_conf"]]
            chosen.append(dict(P=w["P"], preds=preds, gts=w["gts"], fs=rec.fs, fc=rec.fc, n=W, start=w["start"], name=stem))
        png = args.out / f"test_windows_{stem}.png"
        plot_windows(chosen, names, png, f"{stem}: test block, windows drawn at random (white dashed = annotation)")
        from PIL import Image

        Image.open(png).convert("RGB").save(png.with_suffix(".jpg"), quality=80, subsampling=0, optimize=True)

    (args.out / "test_results.json").write_text(json.dumps(results, indent=2))

    # markdown table
    def f(v, nd=2):
        return "-" if v is None else f"{v:.{nd}f}"

    def row(label, r, extra=""):
        lo, hi = r["recall30_ci"]
        return (f"| {label} | {r['n_signals']} | {f(r['recall30'])} ({lo:.2f} to {hi:.2f}) | {f(r['recall50'])} | {f(r['median_iou'])} | "
                f"{f(r['boxes'], 1)} | {f(r['fa50'], 1)} | {f(r['frag50'], 2)} |{extra}")

    out = []
    for stem, res in results.items():
        sp = res["split"]
        out += [f"### {stem}: test = {len(sum([list(range(a, b + 1)) for a, b in sp['test']], []))} windows in blocks {sp['test']} of {sp['n_windows']}", "",
                "| detector | signals | recall IoU 0.3 (95 % interval) | recall IoU 0.5 | median IoU | boxes / window | false alarms / window | fragments / window |",
                "|---|---|---|---|---|---|---|---|"]
        cc = res["classical_choice"]
        for k in CLASSICAL:
            out.append(row(f"{LABELS[k].split(' (')[0]}, untuned (link (4, 16), 40 cells)", res["classical"][k]["untuned"]))
            out.append(row(f"{LABELS[k].split(' (')[0]}, link {tuple(cc['link'])}, {cc['min_cells']} cells chosen on validation" + ("" if cc["within_budget"] else " (none within budget)"),
                           res["classical"][k]["tuned"]))
            if "fixed" in res["classical"][k]:
                fl = args.fixed_link
                out.append(row(f"{LABELS[k].split(' (')[0]}, fixed link ({fl[0]}, {fl[1]}), {fl[2]} cells (chosen on other windows)", res["classical"][k]["fixed"]))
        for n, y in res["yolo"].items():
            where = "trained on this recording" if y["trained_on_this"] else ("synthetic scenes only" if n == list(models)[0] else "cross-recording: not trained on this one")
            out.append(row(f"YOLO {n} ({where}), conf 0.05", y["test_at_005"]))
            out.append(row(f"YOLO {n} ({where}), conf {y['chosen_conf']:g} chosen on validation" + ("" if y["within_budget"] else " (none within budget)"), y["test_at_chosen"]))
        out.append("")
    (args.out / "test_table.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
