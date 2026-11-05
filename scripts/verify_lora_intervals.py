#!/usr/bin/env python
"""Verify that each selected training interval actually shows safe walkway use.

The dataset label says "Safe Walkway". That is a label on the published clip, not
a guarantee about the particular 49-frame window this project cut out of it. This
script checks the window directly, so captions are written from what is in the
window rather than from the class name.

Two measurements per frame, both recorded:

* **People.** The COCO person detector already used for the Stage C worker mask
  (``fasterrcnn_mobilenet_v3_large_fpn``). Every person above the score threshold
  is kept, not only the highest-scoring one, because these clips contain seated
  machine operators as well as pedestrians.
* **Walkway contact.** The marked walkway is green paint, which separates cleanly
  from concrete by ``G - max(R, B)``. A person is counted as *on the walkway* when
  green pixels appear in a small window at the bottom-centre of their box, i.e.
  under their feet.

Neither measurement is a safety judgement. A person standing on green paint is
not thereby working safely, and a frame with no detection is not thereby empty.
The output is evidence for writing an accurate caption and for deciding whether
an interval shows the behaviour at all.
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

GREEN_THRESHOLD = 15
FOOT_WINDOW = 9


def green_mask(frame: np.ndarray, threshold: int = GREEN_THRESHOLD) -> np.ndarray:
    a = frame.astype(np.int16)
    return (a[:, :, 1] - np.maximum(a[:, :, 0], a[:, :, 2])) > threshold


def detect_all_people(clip: np.ndarray, device, score_thresh: float = 0.5):
    """Every person detection per frame, as (box, score) lists."""
    import torch
    from eval_classifier import load_person_detector

    model, person_idx = load_person_detector(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(clip), 8):
            chunk = clip[start : start + 8].astype(np.float32) / 255.0
            batch = [torch.from_numpy(f).permute(2, 0, 1).to(device) for f in chunk]
            for det in model(batch):
                keep = (det["labels"] == person_idx) & (det["scores"] >= score_thresh)
                out.append(
                    [
                        (det["boxes"][i].cpu().numpy().tolist(), float(det["scores"][i]))
                        for i in torch.nonzero(keep).flatten().tolist()
                    ]
                )
    return out


def analyse_clip(clip: np.ndarray, device, score_thresh: float) -> dict:
    h, w = clip.shape[1:3]
    per_frame = detect_all_people(clip, device, score_thresh)
    frames = []
    for i, dets in enumerate(per_frame):
        gm = green_mask(clip[i])
        people = []
        for box, score in dets:
            x1, y1, x2, y2 = box
            fx, fy = (x1 + x2) / 2.0, y2
            r = FOOT_WINDOW // 2
            y0 = int(np.clip(round(fy) - r, 0, h - 1))
            y1w = int(np.clip(round(fy) + r + 1, 1, h))
            x0 = int(np.clip(round(fx) - r, 0, w - 1))
            x1w = int(np.clip(round(fx) + r + 1, 1, w))
            on_green = bool(gm[y0:y1w, x0:x1w].any())
            people.append(
                {
                    "box": [round(v, 1) for v in box],
                    "score": round(score, 3),
                    "height_px": round(y2 - y1, 1),
                    "foot_xy": [round(fx, 1), round(fy, 1)],
                    "on_walkway": on_green,
                }
            )
        frames.append(
            {
                "frame": i,
                "n_people": len(people),
                "n_on_walkway": sum(p["on_walkway"] for p in people),
                "green_fraction": round(float(gm.mean()), 5),
                "people": people,
            }
        )

    n = len(frames)
    any_person = sum(f["n_people"] > 0 for f in frames)
    any_walkway = sum(f["n_on_walkway"] > 0 for f in frames)
    heights = [p["height_px"] for f in frames for p in f["people"] if p["on_walkway"]]
    all_heights = [p["height_px"] for f in frames for p in f["people"]]
    return {
        "n_frames": n,
        "frames_with_person": any_person,
        "frames_with_person_on_walkway": any_walkway,
        "fraction_frames_person_on_walkway": round(any_walkway / n, 4) if n else 0.0,
        "max_people_in_a_frame": max((f["n_people"] for f in frames), default=0),
        "median_walkway_person_height_px": round(float(np.median(heights)), 1) if heights else None,
        "median_person_height_px": round(float(np.median(all_heights)), 1) if all_heights else None,
        "mean_green_fraction": round(float(np.mean([f["green_fraction"] for f in frames])), 5),
        "per_frame": frames,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lora_safe_walkway")
    ap.add_argument("--out", default="outputs/lora/interval_verification.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    args = ap.parse_args(argv)

    ds = REPO / args.dataset
    rels = [l.strip() for l in (ds / "videos.txt").read_text().splitlines() if l.strip()]
    results = {}
    for rel in rels:
        clip_id = Path(rel).stem
        arr = np.stack([f.copy() for f in iter_frames(ds / rel)])
        res = analyse_clip(arr, args.device, args.score_thresh)
        results[clip_id] = res
        print(
            f"{clip_id:8s} frames_with_person={res['frames_with_person']:2d}/{res['n_frames']} "
            f"on_walkway={res['frames_with_person_on_walkway']:2d} "
            f"({res['fraction_frames_person_on_walkway']:.2f}) "
            f"max_people={res['max_people_in_a_frame']} "
            f"median_h={res['median_person_height_px']}",
            flush=True,
        )

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "_method": (
                    "COCO person detector (fasterrcnn_mobilenet_v3_large_fpn), all people "
                    f"above score {args.score_thresh}; walkway contact = any green pixel "
                    f"(G - max(R,B) > {GREEN_THRESHOLD}) in a {FOOT_WINDOW}x{FOOT_WINDOW} "
                    "window at the bottom-centre of the person's box."
                ),
                "_not_a_safety_judgement": (
                    "Standing on green paint is not working safely, and a frame with no "
                    "detection is not an empty frame. This is caption evidence, not a verdict."
                ),
                "score_threshold": args.score_thresh,
                "green_threshold": GREEN_THRESHOLD,
                "clips": results,
            },
            indent=1,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def dense_green_mask(frame, threshold: int = GREEN_THRESHOLD,
                     win: int = 15, min_frac: float = 0.5):
    """Green paint with isolated speckle removed.

    ``green_mask`` alone is a per-pixel colour test, and at full source resolution it
    leaves a tail of scattered pixels across the whole frame: measured on three clips,
    the densest 200 columns held only 70-87% of it. That tail is harmless when counting
    area, and ruinous for any *distance* to the nearest lane pixel, because one stray
    pixel beside a worker's foot reports a distance of zero.

    A pixel is kept when at least ``min_frac`` of a ``win`` x ``win`` neighbourhood is
    green, computed with an integral image. Painted lane markings are solid and survive;
    isolated speckle does not.
    """
    import numpy as np

    m = green_mask(frame, threshold).astype(np.float32)
    pad = win // 2
    p = np.pad(m, pad + 1, mode="constant")
    ii = p.cumsum(0).cumsum(1)
    h, w = m.shape
    y0 = np.arange(h)[:, None]
    x0 = np.arange(w)[None, :]
    y1 = y0 + win
    x1 = x0 + win
    total = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return (total / float(win * win)) >= min_frac
