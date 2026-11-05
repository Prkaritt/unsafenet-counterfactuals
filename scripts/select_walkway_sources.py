#!/usr/bin/env python
"""Characterise candidate walkway-violation clips so three can be chosen on evidence.

For each candidate this measures, at full source resolution:

* where the foreground worker's feet are, per sampled frame (COCO person detector);
* where the marked lane is, as *dense* green paint: the per-pixel colour test alone
  leaves speckle across the whole frame at full resolution (measured: the densest 200
  columns held only 70-87% of it), which makes a minimum-distance measure meaningless
  because one stray pixel beside a foot reports zero. Pixels are kept only where at
  least half of a 15x15 neighbourhood is green;
* the minimum distance in pixels from the worker's feet to the nearest lane pixel,
  which is the quantity that says "near the lane" or "far from the lane" without
  reusing a fixed x threshold across different crops and camera views;
* the worker's direction of travel, as the sign of the change in foot x;
* a crop window that contains both the worker's track and the lane, so the crop is
  adapted to each source instead of copying 0_tr143's.

Nothing here is a safety judgement. Distance from paint is geometry. A clip is a
*candidate* violation because the dataset labels it class 0; this script says whether
the violation is still visible in the window that will actually be used.
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

from unsafenet_cf.clipset import iter_frames  # noqa: E402
from verify_lora_intervals import GREEN_THRESHOLD, dense_green_mask  # noqa: E402

CROP_W, CROP_H = 1620, 1080
SRC_W = 1920


def detect(frames: np.ndarray, device, score_thresh: float = 0.5):
    import torch
    from eval_classifier import load_person_detector

    model, person_idx = load_person_detector(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(frames), 4):
            chunk = frames[start : start + 4].astype(np.float32) / 255.0
            batch = [torch.from_numpy(f).permute(2, 0, 1).to(device) for f in chunk]
            for det in model(batch):
                keep = (det["labels"] == person_idx) & (det["scores"] >= score_thresh)
                people = [
                    (det["boxes"][i].cpu().numpy().tolist(), float(det["scores"][i]))
                    for i in torch.nonzero(keep).flatten().tolist()
                ]
                out.append(people)
    return out


def analyse(path: Path, device, n_frames: int, step: int):
    idx = [i * step for i in range(n_frames)]
    frames, keep = [], []
    for i, f in enumerate(iter_frames(path)):
        if i in idx:
            frames.append(f.copy())
            keep.append(i)
        if i > max(idx):
            break
    if not frames:
        return None
    arr = np.stack(frames)
    dets = detect(arr, device)

    rows = []
    for k, (fi, people) in enumerate(zip(keep, dets)):
        gm = dense_green_mask(arr[k])
        gy, gx = np.nonzero(gm)
        if people:
            box, score = max(people, key=lambda p: p[0][3] - p[0][1])
            x1, y1, x2, y2 = box
            fx, fy = (x1 + x2) / 2.0, y2
            if len(gx):
                d = np.hypot(gx - fx, gy - fy)
                j = int(np.argmin(d))
                dist = float(d[j]); near = [int(gx[j]), int(gy[j])]
            else:
                dist, near = None, None
            rows.append({"frame": fi, "foot_xy": [round(fx, 1), round(fy, 1)],
                         "height_px": round(y2 - y1, 1), "score": round(score, 3),
                         "dist_to_lane_px": round(dist, 1) if dist is not None else None,
                         "nearest_lane_xy": near,
                         "n_people": len(people)})
        else:
            rows.append({"frame": fi, "foot_xy": None, "height_px": None,
                         "score": None, "dist_to_lane_px": None,
                         "nearest_lane_xy": None, "n_people": 0})

    gm_any = np.zeros(arr.shape[1:3], bool)
    for k in range(len(arr)):
        gm_any |= dense_green_mask(arr[k])
    gy, gx = np.nonzero(gm_any)
    lane = {"pixels": int(gm_any.sum()),
            "x_min": int(gx.min()) if len(gx) else None,
            "x_max": int(gx.max()) if len(gx) else None,
            "x_median": int(np.median(gx)) if len(gx) else None}

    fx = [r["foot_xy"][0] for r in rows if r["foot_xy"]]
    dists = [r["dist_to_lane_px"] for r in rows if r["dist_to_lane_px"] is not None]
    misses = sum(1 for r in rows if r["foot_xy"] is None)

    # Crop: 1620 of 1920, so x in [0, 300]. Choose the x that keeps the worker's track
    # and the lane inside the window; if both cannot fit, keep the worker and say so.
    best, best_score = 0, None
    for x in range(0, SRC_W - CROP_W + 1, 10):
        lo, hi = x, x + CROP_W
        w_in = sum(1 for v in fx if lo <= v <= hi) / max(len(fx), 1)
        l_in = (float(((gx >= lo) & (gx <= hi)).mean()) if len(gx) else 0.0)
        sc = 2.0 * w_in + l_in
        if best_score is None or sc > best_score:
            best, best_score = x, sc
    lo, hi = best, best + CROP_W
    crop = {"x": best, "y": 0, "width": CROP_W, "height": CROP_H,
            "worker_frames_inside": round(sum(1 for v in fx if lo <= v <= hi) / max(len(fx), 1), 3),
            "lane_pixels_inside": round(float(((gx >= lo) & (gx <= hi)).mean()) if len(gx) else 0.0, 3)}

    return {
        "path": str(path),
        "frames_sampled": keep,
        "detector_misses": misses,
        "lane": lane,
        "crop_proposed": crop,
        "foot_x_first": round(fx[0], 1) if fx else None,
        "foot_x_last": round(fx[-1], 1) if fx else None,
        "direction": ("right" if fx and fx[-1] > fx[0] + 30 else
                      "left" if fx and fx[-1] < fx[0] - 30 else "little lateral motion")
                     if fx else None,
        "dist_to_lane_px_min": round(min(dists), 1) if dists else None,
        "dist_to_lane_px_median": round(float(np.median(dists)), 1) if dists else None,
        "worker_height_px_median": round(float(np.median(
            [r["height_px"] for r in rows if r["height_px"]])), 1) if fx else None,
        "per_frame": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="/tmp/candidates.json")
    ap.add_argument("--raw-dir", default="data/raw/class0_train")
    ap.add_argument("--out", default="outputs/multi_walkway/source_selection.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--step", type=int, default=12)
    args = ap.parse_args()

    cands = json.loads(Path(args.candidates).read_text())
    results = {}
    for cluster, clip in cands:
        p = REPO / args.raw_dir / f"{clip}.mp4"
        if not p.exists():
            print(f"{clip}: MISSING"); continue
        r = analyse(p, args.device, args.n_frames, args.step)
        if r is None:
            print(f"{clip}: no frames decoded"); continue
        r["scene_cluster"] = cluster
        results[clip] = r
        print(f"{clip:9s} [{cluster:8s}] lane_x {r['lane']['x_min']}-{r['lane']['x_max']} "
              f"dist_min {r['dist_to_lane_px_min']} dist_med {r['dist_to_lane_px_median']} "
              f"h {r['worker_height_px_median']} dir {r['direction']} "
              f"misses {r['detector_misses']}/{args.n_frames} "
              f"crop_x {r['crop_proposed']['x']} "
              f"(worker in {r['crop_proposed']['worker_frames_inside']}, "
              f"lane in {r['crop_proposed']['lane_pixels_inside']})", flush=True)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_method": (
            f"COCO person detector, tallest detection per frame as the foreground worker; "
            f"lane = green paint with G - max(R,B) > {GREEN_THRESHOLD}; distance is the "
            "minimum Euclidean pixel distance from the worker's feet to a lane pixel, at "
            "full source resolution 1920x1080."),
        "_why_distance_not_x": (
            "A fixed x threshold cannot be compared across different crops and camera "
            "views. Distance to that clip's own lane pixels is defined per source."),
        "_not_safety": "Distance from paint is geometry, not a safety judgement.",
        "clips": results}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
