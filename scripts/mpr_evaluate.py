#!/usr/bin/env python
"""Evaluate source, composite and refined clip for every case, against one fixed lane.

Three arms per clip, measured identically. The lane mask is the one annotated from the
*source* in the annotation stage and is applied unchanged to all three, so a clip that
repaints the floor cannot satisfy the foot-placement measure by moving the paint.

Separate questions kept separate, because conflating them is how this project has gone
wrong before:

1. **Did the feet end up on the lane?** Distance from the tracked worker's foot point to
   the nearest painted lane pixel, and whether a 9x9 window at the foot point touches
   paint. No fixed x threshold, so the measure survives different crops and views.
2. **Did the compositing work, or did the refinement work?** These are different
   successes. The composite is the control: it is where the relocation actually happened.
   The refined clip is judged against the composite, not only against the source.
3. **Was the lane itself preserved?** Each arm's own green mask against the source lane,
   as intersection over union.
4. **Was the rest preserved?** Whole-frame PSNR and mean absolute difference, recorded
   with the standing warning that they are dominated by static background.

Identity, gait, shadows and occlusion are **not** scored here. A scalar per frame cannot
see them: that is the lesson the LoRA run in this project paid for. They are assessed by a
human from the contact sheets and the review page, and this file records only what was
measured automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from unsafenet_cf import compare as cmp  # noqa: E402
from unsafenet_cf.clipset import iter_frames  # noqa: E402
from evaluate_multi_walkway import detect_worker, lane_mask  # noqa: E402
from verify_lora_intervals import GREEN_THRESHOLD  # noqa: E402

OUT = REPO / "outputs" / "manual_path"


def load(p: Path) -> np.ndarray:
    return np.stack([f.copy() for f in iter_frames(p)])


def measure(arr, lane_yx, lane, device, intervention_frame):
    ly, lx = lane_yx
    dets = detect_worker(arr, device)
    rows, misses = [], 0
    for i, d in enumerate(dets):
        if d is None:
            misses += 1
            rows.append({"frame": i, "foot_xy": None, "dist_to_lane_px": None, "on_lane": None})
            continue
        (x1, y1, x2, y2), score = d
        fx, fy = (x1 + x2) / 2.0, y2
        dist = float(np.hypot(lx - fx, ly - fy).min())
        h, w = lane.shape
        yy = int(np.clip(round(fy), 0, h - 1))
        xx = int(np.clip(round(fx), 0, w - 1))
        rows.append({
            "frame": i, "foot_xy": [round(fx, 1), round(fy, 1)],
            "dist_to_lane_px": round(dist, 1),
            "on_lane": bool(lane[max(0, yy - 4):yy + 5, max(0, xx - 4):xx + 5].any()),
            "height_px": round(y2 - y1, 1), "score": round(score, 3)})
    ds = [r["dist_to_lane_px"] for r in rows if r["dist_to_lane_px"] is not None]
    post = [r for r in rows if r["frame"] >= intervention_frame]
    post_ds = [r["dist_to_lane_px"] for r in post if r["dist_to_lane_px"] is not None]
    return {
        "frames": int(len(arr)),
        "detector_misses": misses,
        "dist_to_lane_px_median": round(float(np.median(ds)), 1) if ds else None,
        "dist_to_lane_px_min": round(min(ds), 1) if ds else None,
        "frames_on_lane": sum(1 for r in rows if r["on_lane"]),
        "after_intervention": {
            "from_frame": intervention_frame,
            "frames": len(post),
            "dist_to_lane_px_median": round(float(np.median(post_ds)), 1) if post_ds else None,
            "frames_on_lane": sum(1 for r in post if r["on_lane"]),
            "detector_misses": sum(1 for r in post if r["foot_xy"] is None),
        },
        "per_frame": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="configs/manual_path_sources.json")
    ap.add_argument("--clips", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/manual_path/evaluation.json")
    args = ap.parse_args()

    specs = json.loads((REPO / args.sources).read_text())["sources"]
    only = {c for c in args.clips.split(",") if c}
    results = {}

    for spec in specs:
        cid = spec["clip_id"]
        if only and cid not in only:
            continue
        d = OUT / cid
        ann = json.loads((d / "annotation.json").read_text())
        src = load(d / "videos" / f"{cid}_source.mp4")
        lane = lane_mask(src)
        ly, lx = np.nonzero(lane)
        if not len(lx):
            results[cid] = {"error": "no lane pixels in source"}
            continue
        first = ann["intervention"]["first_frame"]

        entry = {
            "lane_reference": {
                "pixels": int(lane.sum()),
                "x_min": int(lx.min()), "x_max": int(lx.max()),
                "components_chosen": ann["lane"]["components_chosen"],
                "from": "annotated from the source clip; applied unchanged to every arm",
            },
            "intervention_first_frame": first,
        }

        arms = {"source": src}
        for arm in ("composite", "refined"):
            p = d / "videos" / f"{cid}_{arm}.mp4"
            if p.exists():
                arms[arm] = load(p)
            else:
                print(f"{cid}: missing {p}")

        for arm, arr in arms.items():
            m = measure(arr, (ly, lx), lane, args.device, first)
            own = lane_mask(arr)
            inter = int((own & lane).sum())
            union = int((own | lane).sum())
            m["lane_preservation"] = {
                "own_lane_pixels": int(own.sum()),
                "iou_with_source_lane": round(inter / union, 4) if union else None,
                "_reading": "IoU near 1 means the marking kept its position and extent",
            }
            if arm != "source":
                m["psnr_vs_source_db"] = cmp.psnr(src, arr)
                m["mean_abs_diff_vs_source"] = cmp.mean_abs_diff(src, arr)
            if arm == "refined" and "composite" in arms:
                m["psnr_vs_composite_db"] = cmp.psnr(arms["composite"], arr)
                m["mean_abs_diff_vs_composite"] = cmp.mean_abs_diff(arms["composite"], arr)
            entry[arm] = m
            ai = m["after_intervention"]
            print(f"{cid:9s} {arm:9s} dist_med {m['dist_to_lane_px_median']} "
                  f"on_lane {m['frames_on_lane']}/{m['frames']} "
                  f"| after f{first}: dist_med {ai['dist_to_lane_px_median']} "
                  f"on_lane {ai['frames_on_lane']}/{ai['frames']} "
                  f"misses {m['detector_misses']} "
                  f"laneIoU {m['lane_preservation']['iou_with_source_lane']}", flush=True)

        sm = entry["source"]["dist_to_lane_px_median"]
        for arm in ("composite", "refined"):
            if arm in entry and entry[arm]["dist_to_lane_px_median"] is not None and sm:
                entry[arm]["delta_dist_vs_source_px"] = round(
                    entry[arm]["dist_to_lane_px_median"] - sm, 1)
        if "composite" in entry and "refined" in entry:
            cm = entry["composite"]["dist_to_lane_px_median"]
            rm = entry["refined"]["dist_to_lane_px_median"]
            entry["refined"]["delta_dist_vs_composite_px"] = (
                round(rm - cm, 1) if (cm is not None and rm is not None) else None)
            entry["relocation_retained_by_refinement"] = {
                "_question": "did the refinement keep the worker where the composite put him",
                "composite_on_lane_after_intervention":
                    entry["composite"]["after_intervention"]["frames_on_lane"],
                "refined_on_lane_after_intervention":
                    entry["refined"]["after_intervention"]["frames_on_lane"],
                "_note": "the pipeline enforces no trajectory constraint; this is measured",
            }
        results[cid] = entry

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_method": __doc__,
        "_negative_delta_means": "feet closer to the lane than in the source",
        "_not_safety": ("Distance to paint is image-plane geometry. Standing on a marked "
                        "lane is not proof of safe behaviour, and no prediction flip would "
                        "be either."),
        "_not_scored_here": ["worker identity", "gait", "cast shadows", "occlusion",
                             "background restoration quality"],
        "green_threshold": GREEN_THRESHOLD,
        "clips": results}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
