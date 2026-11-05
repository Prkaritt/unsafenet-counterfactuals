#!/usr/bin/env python
"""Measure where the worker is in the source, the base edit and the LoRA edit.

Same instrument as scripts/verify_lora_intervals.py, pointed at the three clips
that matter: the COCO person detector locates every person, and a person counts
as on the walkway when green paint appears under their feet. Applying the same
measurement to the training data and to the outputs means the claim "the worker
did or did not move onto the walkway" rests on one method, not on an impression.

The limits carry over. Green paint is not the same thing as the marked lane, the
detector can miss a person, and a person standing on paint is not thereby safe.
This measures position, not safety.
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
from verify_lora_intervals import analyse_clip  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/lora_edit/worker_position.json")
    args = ap.parse_args()

    clips = {
        "source": REPO / "outputs" / "preprocessed" / "0_tr143_720x480x49.mp4",
        "vae_reconstruction": REPO / "outputs" / "stage3_vae" / "reconstruction.mp4",
        "base": REPO / "outputs" / "lora_edit" / "base_seed20261103_s0.6_g6.0.mp4",
        "lora": REPO / "outputs" / "lora_edit" / "lora_seed20261103_s0.6_g6.0.mp4",
    }
    results = {}
    for name, path in clips.items():
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        arr = np.stack([f.copy() for f in iter_frames(path)])
        res = analyse_clip(arr, args.device, args.score_thresh)
        # The largest detected person is the foreground worker in this framing.
        tallest = []
        for f in res["per_frame"]:
            if f["people"]:
                p = max(f["people"], key=lambda q: q["height_px"])
                tallest.append({"frame": f["frame"], "foot_xy": p["foot_xy"],
                                "height_px": p["height_px"], "on_walkway": p["on_walkway"]})
        res["foreground_worker_per_frame"] = tallest
        xs = [t["foot_xy"][0] for t in tallest]
        res["foreground_worker"] = {
            "frames_detected": len(tallest),
            "foot_x_min": round(min(xs), 1) if xs else None,
            "foot_x_median": round(float(np.median(xs)), 1) if xs else None,
            "foot_x_max": round(max(xs), 1) if xs else None,
            "frames_on_walkway": sum(t["on_walkway"] for t in tallest),
        }
        results[name] = res
        fw = res["foreground_worker"]
        print(f"{name:20s} any_person_on_walkway={res['frames_with_person_on_walkway']:2d}/"
              f"{res['n_frames']}  foreground worker: detected {fw['frames_detected']}, "
              f"foot_x median {fw['foot_x_median']}, on_walkway {fw['frames_on_walkway']}",
              flush=True)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_method": "see scripts/verify_lora_intervals.py; identical detector and green test",
        "_reading": (
            "foot_x is the horizontal position of the foreground worker's feet in the "
            "720-pixel-wide frame. In this camera the marked walkway occupies roughly "
            "x>450. A counterfactual that moved the worker onto the walkway would show "
            "foot_x rising toward that band and frames_on_walkway rising with it."
        ),
        "_not_safety": "Position is not safety. This measures where a person is, nothing more.",
        "clips": results,
    }, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
