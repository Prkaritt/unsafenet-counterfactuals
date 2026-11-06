#!/usr/bin/env python
"""Find the panel from the populations, not from a pair of clips.

A single class-2 clip differenced against a single class-6 clip is dominated by whichever
worker happened to be in one of them. That is what defeated the first identification: the
largest difference between `2_tr89` and `6_tr16` was a man in a hi-vis vest, and the one
fixture that did differ, a mesh guard on the press, turned out to be shut in 96 of the 120
class-2 clips. See DECISIONS.md P6.

Averaging each class's per-clip medians over scores of clips suppresses what differs from
clip to clip, which is the people, and keeps what differs consistently with the label. The
difference of those two averages is the measurement this script produces. It also reports,
for the strongest regions it finds, how separable the two classes are **per clip** in that
region, because a difference visible in the average is worthless if it does not hold clip by
clip.

Nothing is selected here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsafenet_cf import clipset as cs       # noqa: E402
from unsafenet_cf import manifest as mf      # noqa: E402
from unsafenet_cf import manualpath as mp    # noqa: E402
from unsafenet_cf import panelpatch as pp    # noqa: E402

OUT = REPO / "outputs" / "panel_cover"
W, H = 960, 540          # half resolution; the full-res box scales by 2
SMOOTH = 9
MIN_BLOB = 80


def log(m: str) -> None:
    print(m, flush=True)


def clip_median(path: Path, stride: int, cap: int) -> np.ndarray | None:
    frames = []
    for i, f in enumerate(cs.iter_frames(path, width=W, height=H)):
        if i % stride == 0:
            frames.append(f.copy())
        if len(frames) >= cap:
            break
    if not frames:
        return None
    return np.median(np.stack(frames).astype(np.float32), axis=0)


def save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="outputs/panel_cover/selection.json")
    ap.add_argument("--out", default="outputs/panel_cover/population")
    ap.add_argument("--align-min", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--top", type=int, default=6, help="regions carried to the per-clip test")
    args = ap.parse_args()

    sel = json.loads((REPO / args.selection).read_text())
    rows = [r for r in sel["clips"]
            if "error" not in r and r["alignment_ncc"] >= args.align_min]
    by_class = {2: [r for r in rows if r["dataset_class"] == 2],
                6: [r for r in rows if r["dataset_class"] == 6]}
    log(f"aligned at >= {args.align_min}: class 2 {len(by_class[2])}, "
        f"class 6 {len(by_class[6])}")
    if not by_class[2] or not by_class[6]:
        log("one class has no aligned clips; cannot compare populations")
        return 2

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    meds: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, r in enumerate(by_class[2] + by_class[6], 1):
        m = clip_median(REPO / r["path"], args.stride, args.cap)
        if m is not None:
            meds[r["clip_id"]] = m
        if i % 25 == 0:
            log(f"  medians {i}/{len(rows)} in {time.time() - t0:.0f}s")
    log(f"medians built for {len(meds)} clips in {time.time() - t0:.0f}s")

    def mean_of(cls: int) -> np.ndarray:
        ids = [r["clip_id"] for r in by_class[cls] if r["clip_id"] in meds]
        return np.mean(np.stack([meds[i] for i in ids]), axis=0), ids

    mean2, ids2 = mean_of(2)
    mean6, ids6 = mean_of(6)
    diff = pp.box_mean(np.abs(mean2 - mean6).mean(axis=2), SMOOTH)
    log(f"population difference: max {diff.max():.2f}, p99 {np.percentile(diff, 99):.2f}, "
        f"p99.9 {np.percentile(diff, 99.9):.2f}, median {np.median(diff):.2f}")

    thresh = float(max(8.0, np.percentile(diff, 99.5)))
    comps = mp.connected_components(diff >= thresh, min_area=MIN_BLOB)
    log(f"regions above {thresh:.2f}: {len(comps)}")

    regions = []
    for c in comps[: args.top]:
        x0, y0, x1, y1 = c["bbox"]
        box = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
        # Per-clip separability: correlate each clip's region with each class mean.
        r2 = mean2[y0:y1 + 1, x0:x1 + 1].mean(axis=2)
        r6 = mean6[y0:y1 + 1, x0:x1 + 1].mean(axis=2)
        scores = {2: [], 6: []}
        for cls, ids in ((2, ids2), (6, ids6)):
            for cid in ids:
                reg = meds[cid][y0:y1 + 1, x0:x1 + 1].mean(axis=2)
                scores[cls].append(pp.ncc(reg, r6) - pp.ncc(reg, r2))
        s2 = np.array(scores[2])
        s6 = np.array(scores[6])
        # A threshold midway between the class means, and the accuracy it gives.
        cut = float((s2.mean() + s6.mean()) / 2.0)
        acc = float((np.mean(s2 < cut) + np.mean(s6 >= cut)) / 2.0)
        regions.append({
            "box_half_res_xywh": box,
            "box_full_res_xywh": [v * 2 for v in box],
            "area_px": int(c["area"]),
            "mean_population_diff": round(float(diff[y0:y1 + 1, x0:x1 + 1].mean()), 3),
            "max_population_diff": round(float(diff[y0:y1 + 1, x0:x1 + 1].max()), 3),
            "per_clip_separation": {
                "statistic": "ncc(region, class6 mean) - ncc(region, class2 mean)",
                "class2_mean": round(float(s2.mean()), 4),
                "class2_std": round(float(s2.std()), 4),
                "class6_mean": round(float(s6.mean()), 4),
                "class6_std": round(float(s6.std()), 4),
                "midpoint_threshold": round(cut, 4),
                "balanced_accuracy_of_that_threshold": round(acc, 4),
                "_meaning": ("A region whose balanced accuracy is near 0.5 does not "
                             "separate the classes clip by clip, however large its "
                             "difference looks in the population average."),
            },
        })
        log(f"  region {box} area {c['area']:6d} meandiff "
            f"{regions[-1]['mean_population_diff']:6.2f} sep acc {acc:.3f} "
            f"(class2 {s2.mean():+.4f}+-{s2.std():.4f}, class6 {s6.mean():+.4f}+-{s6.std():.4f})")

    save_png(mean2, out / "mean_class2.png")
    save_png(mean6, out / "mean_class6.png")
    heat = np.zeros(diff.shape + (3,), np.float32)
    d = diff / max(1e-6, diff.max())
    heat[..., 0] = 255 * d
    overlay = (0.55 * mean6 + 0.45 * heat)
    for r in regions:
        x, y, w, h = r["box_half_res_xywh"]
        overlay[y:y + 2, x:x + w] = [255, 255, 0]
        overlay[y + h - 2:y + h, x:x + w] = [255, 255, 0]
        overlay[y:y + h, x:x + 2] = [255, 255, 0]
        overlay[y:y + h, x + w - 2:x + w] = [255, 255, 0]
    save_png(np.vstack([mean2, mean6, heat, overlay]), out / "population_difference.png")

    report = {
        "_what": ("Mean of per-clip medians for each class, their difference, and how "
                  "well each strong region separates the classes clip by clip."),
        "generated": mf.utc_now(),
        "alignment_min_ncc": args.align_min,
        "clips_used": {"class_2": len(ids2), "class_6": len(ids6)},
        "resolution": [W, H],
        "sampling": {"stride": args.stride, "cap": args.cap},
        "difference_threshold": round(thresh, 3),
        "regions": regions,
        "_caveat": ("A region that separates the classes here is a region that differs "
                    "with the label. It is not proof that the region holds a panel "
                    "cover, and it must be looked at before anything is selected."),
        "environment": mf.environment_snapshot(),
    }
    (out / "population.json").write_text(json.dumps(report, indent=1))
    log(f"\nwrote {out / 'population.json'}")
    log(f"images: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
