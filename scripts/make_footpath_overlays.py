#!/usr/bin/env python
"""Draw each clip's worker foot path over its own first frame, with the lane marked.

One image per source: three panels, source / base / LoRA, each showing that clip's
lane in cyan and the worker's foot positions as a track from dark to bright. A frame
where the detector found nobody draws no point, so gaps in a track are detection
failures and are visible rather than interpolated away.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from unsafenet_cf.clipset import iter_frames  # noqa: E402
from verify_lora_intervals import dense_green_mask  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="outputs/multi_walkway/generation_plan.json")
    ap.add_argument("--eval", default="outputs/multi_walkway/evaluation.json")
    ap.add_argument("--out", default="outputs/multi_walkway/overlays")
    args = ap.parse_args()

    plan = json.loads((REPO / args.plan).read_text())
    ev = json.loads((REPO / args.eval).read_text())["clips"]
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    for s in plan["sources"]:
        cid = s["clip_id"]
        e = ev.get(cid)
        if not e or "source" in e and "error" in e:
            print(f"{cid}: no evaluation, skipping")
            continue
        paths = {
            "source": REPO / s["preprocessed"],
            "base": REPO / "outputs/multi_walkway/videos" / f"{cid}_base.mp4",
            "lora": REPO / "outputs/multi_walkway/videos" / f"{cid}_lora.mp4",
        }
        panels, misses = [], {}
        for arm, p in paths.items():
            if not p.exists() or arm not in e:
                continue
            first = next(iter(iter_frames(p))).copy()
            lane = np.zeros(first.shape[:2], bool)
            for f in iter_frames(p):
                lane |= dense_green_mask(f)
            img = first.copy()
            img[lane] = (0, 255, 255)
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            rows = e[arm]["per_frame"]
            n = max(len(rows), 1)
            prev = None
            for i, r in enumerate(rows):
                if not r.get("foot_xy"):
                    prev = None
                    continue
                x, y = r["foot_xy"]
                t = i / n
                col = (int(60 + 195 * t), int(20 + 60 * t), int(255 - 180 * t))
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
                if prev:
                    d.line([prev, (x, y)], fill=col, width=2)
                prev = (x, y)
            d.text((6, 6), f"{cid} {arm}", fill=(255, 255, 0))
            misses[arm] = e[arm].get("detector_misses")
            panels.append(np.array(im))
        if not panels:
            continue
        Image.fromarray(np.concatenate(panels, axis=1)).save(out / f"{cid}_footpath.png")
        print(f"{cid}: wrote overlay, detector misses {misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
