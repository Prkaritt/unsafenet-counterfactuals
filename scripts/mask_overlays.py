#!/usr/bin/env python3
"""Render the worker, lane and control masks over real frames, for visual checking.

Motion-energy coverage is a number, and a number can be high or low for reasons that have
nothing to do with whether a mask is over the worker. Looking at the frames is the check
that settles it. This runs on CPU from the decode cache, so it costs no allocation.

Output: one PNG per sampled clip, rows for the three masks, columns for sampled times,
with each mask drawn as an outlined rectangle over the real frame.
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

from unsafenet_cf import classifier as C    # noqa: E402

OUT = REPO / "outputs" / "classifier" / "mask_overlays"


def draw_box(img: np.ndarray, box, colour, width: int = 2) -> None:
    """Outline a box in place."""
    y0, x0, bh, bw = box
    h, w = img.shape[:2]
    y1, x1 = min(y0 + bh, h), min(x0 + bw, w)
    for t in range(width):
        if y0 + t < h:
            img[y0 + t, x0:x1] = colour
        if y1 - 1 - t >= 0:
            img[y1 - 1 - t, x0:x1] = colour
        if x0 + t < w:
            img[y0:y1, x0 + t] = colour
        if x1 - 1 - t >= 0:
            img[y0:y1, x1 - 1 - t] = colour


def main(argv=None) -> int:
    from PIL import Image
    import eval_classifier as E
    from train_classifier import decode_cached

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=3,
                    help="clips to render from each dataset class")
    ap.add_argument("--times", type=int, default=5, help="frames sampled per clip")
    ap.add_argument("--mask-area", type=float, default=0.06)
    ap.add_argument("--device", default="cuda",
                    help="device for the person detector; 'cpu' works but is slow")
    args = ap.parse_args(argv)

    import torch
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    clf = REPO / "outputs" / "classifier"
    inventory = json.loads((clf / "clip_inventory.json").read_text())
    val = [e for e in inventory if e["split"] == "val"]
    chosen = []
    for cls in (0, 4):
        chosen += [e for e in val if e["dataset_class"] == cls][:args.per_class]
    if not chosen:
        print("no validation clips found")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    prep = (C.DEFAULT_FRAMES, C.DEFAULT_HEIGHT, C.DEFAULT_WIDTH)
    colours = {"worker": (255, 60, 60), "lane": (60, 220, 90),
               "control": (90, 160, 255)}
    summary = []

    for entry in chosen:
        clip = decode_cached(entry, *prep)
        t_idx = np.linspace(0, len(clip) - 1, args.times).round().astype(int)
        rows = []
        clip_row = {"clip_id": entry["clip_id"], "dataset_class": entry["dataset_class"]}
        for kind in ("worker", "lane", "control"):
            if kind == "worker":
                boxes, missed, scores = E.worker_boxes_from_detector(
                    clip, args.mask_area, device)
                clip_row["frames_with_no_person_detected"] = int(missed)
                clip_row["frames_total"] = len(clip)
                clip_row["detector_scores_sampled"] = [
                    round(float(scores[t]), 3) for t in t_idx]
            else:
                boxes = E.per_frame_boxes(clip, kind, args.mask_area)
            clip_row[f"{kind}_energy_removed"] = E.motion_energy_removed(clip, boxes)
            panels = []
            for t in t_idx:
                img = clip[t].copy()
                draw_box(img, boxes[t], np.array(colours[kind], dtype=img.dtype))
                panels.append(img)
            rows.append(np.concatenate(panels, axis=1))
        sheet = np.concatenate(rows, axis=0)
        path = OUT / f"{entry['clip_id']}_class{entry['dataset_class']}.png"
        Image.fromarray(sheet).save(path)
        clip_row["sheet"] = str(path)
        clip_row["frames_shown"] = t_idx.tolist()
        summary.append(clip_row)
        print(f"{entry['clip_id']} class {entry['dataset_class']}: "
              f"person missed in {clip_row['frames_with_no_person_detected']}/"
              f"{clip_row['frames_total']} frames, "
              f"worker energy {clip_row['worker_energy_removed']:.1%}, "
              f"lane {clip_row['lane_energy_removed']:.1%}, "
              f"control {clip_row['control_energy_removed']:.1%} -> {path.name}",
              flush=True)

    (OUT / "overlay_summary.json").write_text(json.dumps({
        "rows_top_to_bottom": ["worker", "lane", "control"],
        "box_colours_rgb": colours,
        "mask_area_fraction": args.mask_area,
        "clips": summary,
        "worker_localisation": "COCO person detector (fasterrcnn_mobilenet_v3_large_fpn)",
        "note": ("Motion-energy coverage is supporting evidence about where a mask sits, "
                 "not proof that the mask covers the worker. These frames are the check "
                 "that settles it."),
    }, indent=2))
    print(f"wrote {OUT}/overlay_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
