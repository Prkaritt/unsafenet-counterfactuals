#!/usr/bin/env python
"""Evaluate each source's base and LoRA edit against that source's own lane.

Three measurements per clip, kept separate because they answer different questions.

1. **Did the worker's feet move toward or into the lane?** The lane is annotated from
   the *preprocessed source* clip, as the union of its green-paint pixels over all
   frames, and that one mask is then applied to the source, the base edit and the LoRA
   edit alike. Using the source's lane rather than each clip's own is deliberate: it
   asks whether the worker moved to where the lane really is, and cannot be satisfied
   by a model that repaints the floor. No fixed x threshold is used anywhere, so the
   measure is comparable across different crops and camera views.

2. **Was the lane itself preserved?** Each clip's own green mask is measured separately
   -- pixel count, bounding box, and intersection-over-union against the source's lane.
   A low IoU means the edit moved, widened or erased the marking, which is the spurious
   change this project exists to reject.

3. **Was the rest preserved?** Whole-frame PSNR and mean absolute difference, recorded
   with the warning that they are dominated by the static background and are nearly
   blind to a worker occupying a small part of the frame.

Detection failures are counted, never silently filled. Everything here is sampled from
decoded frames; it is not a substitute for watching the clip.
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
from verify_lora_intervals import GREEN_THRESHOLD, dense_green_mask  # noqa: E402


def load(path: Path) -> np.ndarray:
    return np.stack([f.copy() for f in iter_frames(path)])


def lane_mask(arr: np.ndarray) -> np.ndarray:
    m = np.zeros(arr.shape[1:3], bool)
    for f in arr:
        m |= dense_green_mask(f)
    return m


def detect_all(arr: np.ndarray, device, score_thresh: float = 0.5):
    """Every person detection per frame, not only the largest."""
    import torch
    from eval_classifier import load_person_detector

    model, person_idx = load_person_detector(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(arr), 8):
            chunk = arr[start : start + 8].astype(np.float32) / 255.0
            batch = [torch.from_numpy(f).permute(2, 0, 1).to(device) for f in chunk]
            for det in model(batch):
                keep = (det["labels"] == person_idx) & (det["scores"] >= score_thresh)
                out.append([(det["boxes"][i].cpu().numpy().tolist(), float(det["scores"][i]))
                            for i in torch.nonzero(keep).flatten().tolist()])
    return out


def track_worker(per_frame, max_jump: float = 120.0, min_height_ratio: float = 0.5):
    """Follow ONE person through the clip instead of taking the largest each frame.

    Taking the largest detection per frame looks reasonable and is wrong here: these
    clips contain distant figures as well as the foreground worker, and from frame to
    frame the "largest" flips between them. The resulting path jumps across the frame
    and any distance statistic computed from it is meaningless. Seen directly in the
    first foot-path overlays, which showed long straight lines between unrelated people.

    The track is seeded on the frame holding the single largest detection, then grown
    forwards and backwards, each step taking the detection nearest the previous foot
    position, subject to a jump gate and a height-consistency gate. A frame with no
    acceptable candidate is a miss and stays empty; nothing is interpolated.
    """
    def foot(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) / 2.0, y2, y2 - y1)

    seed_i, seed_d, best_h = None, None, -1.0
    for i, ppl in enumerate(per_frame):
        for b, sc in ppl:
            h = b[3] - b[1]
            if h > best_h:
                seed_i, seed_d, best_h = i, (b, sc), h
    track = [None] * len(per_frame)
    if seed_i is None:
        return track
    track[seed_i] = seed_d

    def grow(order):
        prev = seed_d
        for i in order:
            if prev is None:
                break
            px, py, ph = foot(prev[0])
            best, bd = None, None
            for b, sc in per_frame[i]:
                cx, cy, ch = foot(b)
                if ph > 0 and not (min_height_ratio <= ch / ph <= 1.0 / min_height_ratio):
                    continue
                d = float(np.hypot(cx - px, cy - py))
                if d > max_jump:
                    continue
                if bd is None or d < bd:
                    best, bd = (b, sc), d
            track[i] = best
            if best is not None:
                prev = best
    grow(range(seed_i + 1, len(per_frame)))
    grow(range(seed_i - 1, -1, -1))
    return track


def detect_worker(arr: np.ndarray, device, score_thresh: float = 0.5):
    return track_worker(detect_all(arr, device, score_thresh))


def measure(arr, ref_lane_yx, ref_lane_mask, device):
    ly, lx = ref_lane_yx
    dets = detect_worker(arr, device)
    rows, misses = [], 0
    for i, d in enumerate(dets):
        if d is None:
            misses += 1
            rows.append({"frame": i, "foot_xy": None, "dist_to_lane_px": None,
                         "on_lane": None, "height_px": None})
            continue
        (x1, y1, x2, y2), score = d
        fx, fy = (x1 + x2) / 2.0, y2
        dd = np.hypot(lx - fx, ly - fy)
        dist = float(dd.min())
        h, w = ref_lane_mask.shape
        yy, xx = int(np.clip(round(fy), 0, h - 1)), int(np.clip(round(fx), 0, w - 1))
        y0, y1b = max(0, yy - 4), min(h, yy + 5)
        x0, x1b = max(0, xx - 4), min(w, xx + 5)
        rows.append({"frame": i, "foot_xy": [round(fx, 1), round(fy, 1)],
                     "dist_to_lane_px": round(dist, 1),
                     "on_lane": bool(ref_lane_mask[y0:y1b, x0:x1b].any()),
                     "height_px": round(y2 - y1, 1), "score": round(score, 3)})
    ds = [r["dist_to_lane_px"] for r in rows if r["dist_to_lane_px"] is not None]
    return {
        "frames": len(arr), "detector_misses": misses,
        "dist_to_lane_px_median": round(float(np.median(ds)), 1) if ds else None,
        "dist_to_lane_px_min": round(min(ds), 1) if ds else None,
        "frames_on_lane": sum(1 for r in rows if r["on_lane"]),
        "per_frame": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="outputs/multi_walkway/generation_plan.json")
    ap.add_argument("--out", default="outputs/multi_walkway/evaluation.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    plan = json.loads((REPO / args.plan).read_text())
    results = {}
    for src in plan["sources"]:
        cid = src["clip_id"]
        s_arr = load(REPO / src["preprocessed"])
        ref_mask = lane_mask(s_arr)
        ly, lx = np.nonzero(ref_mask)
        if not len(lx):
            print(f"{cid}: NO LANE PIXELS in the source; skipping, cannot evaluate")
            results[cid] = {"error": "no green lane pixels found in the preprocessed source"}
            continue
        ref_yx = (ly, lx)
        entry = {"lane_reference": {
            "source_lane_pixels": int(ref_mask.sum()),
            "x_min": int(lx.min()), "x_max": int(lx.max()),
            "from": "union of green pixels over all frames of the preprocessed source"}}

        arms = {"source": s_arr}
        for arm in ("base", "lora"):
            p = REPO / "outputs" / "multi_walkway" / "videos" / f"{cid}_{arm}.mp4"
            if p.exists():
                arms[arm] = load(p)
            else:
                print(f"{cid}: missing {p}")

        for arm, arr in arms.items():
            m = measure(arr, ref_yx, ref_mask, args.device)
            own = lane_mask(arr)
            inter = int((own & ref_mask).sum()); union = int((own | ref_mask).sum())
            m["lane_preservation"] = {
                "own_lane_pixels": int(own.sum()),
                "iou_with_source_lane": round(inter / union, 4) if union else None,
                "_reading": "IoU near 1 means the marking kept its position and extent"}
            if arm != "source":
                m["psnr_vs_source_db"] = cmp.psnr(s_arr, arr)
                m["mean_abs_diff_vs_source"] = cmp.mean_abs_diff(s_arr, arr)
            entry[arm] = m
            print(f"{cid:9s} {arm:6s} dist_med {m['dist_to_lane_px_median']} "
                  f"dist_min {m['dist_to_lane_px_min']} on_lane {m['frames_on_lane']}/{m['frames']} "
                  f"misses {m['detector_misses']} laneIoU {m['lane_preservation']['iou_with_source_lane']}",
                  flush=True)

        if "base" in entry and "lora" in entry:
            sm = entry["source"]["dist_to_lane_px_median"]
            for arm in ("base", "lora"):
                am = entry[arm]["dist_to_lane_px_median"]
                entry[arm]["delta_dist_vs_source_px"] = (
                    round(am - sm, 1) if (am is not None and sm is not None) else None)
        results[cid] = entry

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_method": __doc__,
        "_negative_delta_means": "feet closer to the lane than in the source",
        "_not_safety": ("Distance to paint is geometry. Moving onto a marked lane is not "
                        "proof of safe walking, and a prediction flip would not be either."),
        "green_threshold": GREEN_THRESHOLD,
        "clips": results}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
