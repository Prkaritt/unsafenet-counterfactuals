#!/usr/bin/env python
"""Annotate lanes, obstacles and agent-specified target paths. No model runs here.

Two modes, run in order.

``--mode survey`` decodes each preprocessed source, builds the dense-green mask, splits it
into connected components and writes one overlay per clip with every component numbered and
coloured. Green paint is a *candidate* for the pedestrian lane and nothing more: this
dataset also paints machine bases green, and an earlier run in this project was misled by
exactly that, scoring a clip as having walkway contact when the contact was on a green
machine base at the frame edge. The survey exists so a human can look at the numbered
components and say which ones are the lane.

``--mode finalize`` reads the frozen plan, rasterises the chosen components into a lane
polygon, interpolates the target path from its keyframes, and runs every plausibility check
that could invalidate an agent-specified intervention: teleportation, machinery crossing,
whether the path starts where the worker actually was, and whether it ends inside the lane.
Failures are recorded, never corrected silently.

The worker tracks come from measurements written by earlier GPU jobs. No detector is loaded
here, so this stage is pure array work and is allowed to run on a login node.
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

from unsafenet_cf import manualpath as mp  # noqa: E402
from unsafenet_cf.clipset import iter_frames  # noqa: E402
from verify_lora_intervals import GREEN_THRESHOLD, dense_green_mask  # noqa: E402

PALETTE = [
    (255, 64, 64), (64, 160, 255), (255, 208, 48), (144, 96, 255),
    (48, 224, 176), (255, 128, 192), (176, 176, 64), (96, 224, 96),
]


def load_clip(path: Path) -> np.ndarray:
    return np.stack([f.copy() for f in iter_frames(path)])


def union_green(frames: np.ndarray) -> np.ndarray:
    m = np.zeros(frames.shape[1:3], bool)
    for f in frames:
        m |= dense_green_mask(f)
    return m


def recorded_track(spec: dict) -> list[dict | None]:
    """Per-frame foot position and box height for the tracked worker.

    Two recorded formats. ``evaluation.json`` already holds a single tracked worker per
    frame. ``worker_position.json`` holds every detection per frame, so the same
    continuity-gated tracker that produced the other file is applied to its boxes here.
    No model is loaded either way: the boxes are measurements already on disk.
    """
    src = json.loads((REPO / spec["track_source"]).read_text())
    kind = spec["track_format"]
    if kind == "evaluation":
        rows = src["clips"][spec["clip_id"]]["source"]["per_frame"]
        return [
            None if r.get("foot_xy") is None else
            {"frame": r["frame"], "foot_xy": r["foot_xy"], "height_px": r.get("height_px"),
             "box": None, "score": r.get("score")}
            for r in rows
        ]
    if kind == "worker_position":
        from evaluate_multi_walkway import track_worker
        per_frame = [
            [(p["box"], p["score"]) for p in r["people"]]
            for r in src["clips"]["source"]["per_frame"]
        ]
        tracked = track_worker(per_frame)
        out: list[dict | None] = []
        for i, d in enumerate(tracked):
            if d is None:
                out.append(None)
                continue
            (x1, y1, x2, y2), score = d
            out.append({"frame": i, "foot_xy": [round((x1 + x2) / 2.0, 1), round(y2, 1)],
                        "height_px": round(y2 - y1, 1),
                        "box": [round(v, 1) for v in (x1, y1, x2, y2)],
                        "score": score})
        return out
    raise SystemExit(f"unknown track_format {kind!r}")


# ----------------------------------------------------------------------------- drawing

def _png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _dot(canvas: np.ndarray, x: float, y: float, colour, r: int = 3) -> None:
    h, w, _ = canvas.shape
    yy, xx = int(round(y)), int(round(x))
    y0, y1 = max(0, yy - r), min(h, yy + r + 1)
    x0, x1 = max(0, xx - r), min(w, xx + r + 1)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] = colour


def _line(canvas: np.ndarray, p, q, colour, r: int = 1) -> None:
    n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1])) + 1)
    for t in np.linspace(0.0, 1.0, n):
        _dot(canvas, p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), colour, r)


def _tint(canvas: np.ndarray, mask: np.ndarray, colour, strength: float = 0.55) -> None:
    sel = canvas[mask].astype(np.float32)
    canvas[mask] = (sel * (1 - strength) + np.array(colour, np.float32) * strength).astype(np.uint8)


# ------------------------------------------------------------------------------ survey

def survey(spec: dict, out_dir: Path) -> dict:
    frames = load_clip(REPO / spec["preprocessed"])
    green = union_green(frames)
    comps = mp.connected_components(green, min_area=spec.get("min_component_area", 60))
    base = frames[len(frames) // 2].copy()
    overlay = base.copy()

    recorded = []
    for i, c in enumerate(comps[:len(PALETTE)]):
        m = np.zeros(green.shape, bool)
        m[c["pixels"][:, 0], c["pixels"][:, 1]] = True
        _tint(overlay, m, PALETTE[i], 0.75)
        x0, y0, x1, y1 = c["bbox"]
        for yy in (y0, y1):
            _line(overlay, (x0, yy), (x1, yy), PALETTE[i], 1)
        for xx in (x0, x1):
            _line(overlay, (xx, y0), (xx, y1), PALETTE[i], 1)
        recorded.append({k: v for k, v in c.items() if k != "pixels"} | {
            "component": i, "colour_rgb": list(PALETTE[i]),
            "fill_ratio_in_bbox": round(c["area"] / max(1, c["x_span"] * c["y_span"]), 3),
        })

    track = recorded_track(spec)
    for t in track:
        if t is not None:
            _dot(overlay, t["foot_xy"][0], t["foot_xy"][1], (255, 255, 255), 2)

    _png(overlay, out_dir / "overlays" / "01_green_components.png")
    _png(base, out_dir / "overlays" / "00_source_midframe.png")

    strip = np.concatenate([frames[i] for i in (0, 12, 24, 36, 48)], axis=1)
    _png(strip, out_dir / "overlays" / "02_source_strip.png")

    fy = [t["foot_xy"][1] for t in track if t is not None and t.get("height_px")]
    hh = [t["height_px"] for t in track if t is not None and t.get("height_px")]
    fit = mp.fit_height_vs_y(fy, hh)

    feet = [t["foot_xy"] for t in track if t is not None]
    steps = [float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(feet, feet[1:])]

    result = {
        "clip_id": spec["clip_id"],
        "frames": int(len(frames)),
        "shape_hw": [int(frames.shape[1]), int(frames.shape[2])],
        "green_threshold": GREEN_THRESHOLD,
        "green_pixels_total": int(green.sum()),
        "components": recorded,
        "_components_warning": (
            "A green component is a candidate for the pedestrian lane and nothing more. "
            "This dataset paints machine bases green too, and this project has already been "
            "misled by one. Choose components by looking at 01_green_components.png."),
        "perspective_fit": fit,
        "worker_track": {
            "frames_with_worker": sum(1 for t in track if t is not None),
            "detector_misses": sum(1 for t in track if t is None),
            "foot_x_range": [round(min(p[0] for p in feet), 1), round(max(p[0] for p in feet), 1)] if feet else None,
            "foot_y_range": [round(min(p[1] for p in feet), 1), round(max(p[1] for p in feet), 1)] if feet else None,
            "median_step_px": round(float(np.median(steps)), 2) if steps else None,
            "p90_step_px": round(float(np.percentile(steps, 90)), 2) if steps else None,
            "max_step_px": round(max(steps), 2) if steps else None,
            "_source": spec["track_source"],
            "_note": "Recorded by an earlier GPU job. No detector was loaded in this stage.",
        },
        "per_frame_track": track,
    }
    (out_dir / "survey.json").write_text(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------------------- finalize

def finalize(spec: dict, plan: dict, out_dir: Path) -> dict:
    frames = load_clip(REPO / spec["preprocessed"])
    h, w = frames.shape[1], frames.shape[2]
    green = union_green(frames)
    comps = mp.connected_components(green, min_area=spec.get("min_component_area", 60))

    chosen = plan["lane_components"]
    lane_mask = np.zeros((h, w), bool)
    for i in chosen:
        c = comps[i]
        lane_mask[c["pixels"][:, 0], c["pixels"][:, 1]] = True
    pts = np.stack(np.nonzero(lane_mask)[::-1], axis=1)
    hull = mp.convex_hull(pts.astype(np.float64))
    hull_mask = mp.polygon_mask(hull, (h, w))

    obstacles = {name: np.array(poly, float) for name, poly in plan.get("obstacles", {}).items()}

    survey_data = json.loads((out_dir / "survey.json").read_text())
    track = survey_data["per_frame_track"]
    source_feet = [None if t is None else tuple(t["foot_xy"]) for t in track]

    path = mp.interpolate_path(plan["path_keyframes"], len(frames))

    # "follow_source" keeps each relocated frame on the foot row the worker actually
    # occupied in that frame, so only the horizontal position changes. That matters more
    # than it looks: a silhouette cut from frame f is the right size for the depth it was
    # photographed at, so holding its row fixed means the composite never rescales a
    # person, and the weak height-versus-row fits on two of these clips are never relied
    # on. The cost is that the intervention cannot move the worker nearer to or further
    # from the camera, only across the frame.
    y_mode = plan.get("y_mode", "keyframes")
    if y_mode == "follow_source":
        for p in path:
            if p is None:
                continue
            sf = source_feet[p["frame"]]
            if sf is not None:
                p["y"] = float(sf[1])

    checks = mp.path_checks(path, source_feet, hull, obstacles,
                            speed_limit_px=plan["speed_limit_px"])
    checks["y_mode"] = y_mode

    overlay = frames[len(frames) // 2].copy()
    _tint(overlay, lane_mask, (64, 224, 112), 0.8)
    _tint(overlay, hull_mask & ~lane_mask, (64, 224, 112), 0.22)
    for name, poly in obstacles.items():
        pm = mp.polygon_mask(poly, (h, w))
        _tint(overlay, pm, (255, 72, 72), 0.25)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        _line(overlay, a, b, (32, 255, 128), 1)
    for t in track:
        if t is not None:
            _dot(overlay, t["foot_xy"][0], t["foot_xy"][1], (235, 235, 235), 2)
    prev = None
    for p in path:
        if p is None:
            continue
        if prev is not None:
            _line(overlay, (prev["x"], prev["y"]), (p["x"], p["y"]), (255, 196, 32), 1)
        _dot(overlay, p["x"], p["y"], (255, 128, 0), 2)
        prev = p
    for k in plan["path_keyframes"]:
        _dot(overlay, k["x"], k["y"], (255, 255, 255), 4)
        _dot(overlay, k["x"], k["y"], (255, 48, 48), 2)
    _png(overlay, out_dir / "overlays" / "03_lane_path.png")

    result = {
        "clip_id": spec["clip_id"],
        "frames": int(len(frames)),
        "lane": {
            "components_chosen": chosen,
            "why": plan["lane_rationale"],
            "mask_pixels": int(lane_mask.sum()),
            "polygon_xy": [[round(float(x), 1), round(float(y), 1)] for x, y in hull],
            "polygon_pixels": int(hull_mask.sum()),
            "_note": ("The polygon is the convex hull of the chosen paint, so it spans gaps "
                      "between dashes and is larger than the painted area. Foot placement is "
                      "scored against the painted mask, not the hull."),
        },
        "obstacles": {name: [[float(x), float(y)] for x, y in poly] for name, poly in obstacles.items()},
        "intervention": {
            "first_frame": plan["path_keyframes"][0]["frame"],
            "frames_untouched_before": plan["path_keyframes"][0]["frame"],
            "keyframes": plan["path_keyframes"],
            "rationale": plan["path_rationale"],
            "_label": ("Agent-specified intervention. I proposed this path from image "
                       "evidence. It was not discovered as safe behaviour by any model and "
                       "is not present in the dataset."),
        },
        "path_checks": checks,
        "per_frame_path": path,
        "geometry_uncertainty": [
            "The path is drawn in the image plane of a 720x480 frame. There is no camera "
            "calibration and no ground plane, so no step, distance or clearance here is a "
            "real-world measurement.",
            "The only perspective evidence is the empirical fit of detector box height "
            "against foot row, recorded in survey.json. It holds across the foot rows the "
            "worker actually occupied and nowhere else.",
            "Placing a silhouette inside the lane polygon says where the worker appears to "
            "stand. It does not establish that walking there is safe.",
        ],
    }
    (out_dir / "annotation.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out_dir / "lane_mask.npz", lane=lane_mask, hull=hull_mask)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["survey", "finalize"], required=True)
    ap.add_argument("--sources", default="configs/manual_path_sources.json")
    ap.add_argument("--plan", default="configs/manual_path_plan.json")
    ap.add_argument("--out-root", default="outputs/manual_path")
    ap.add_argument("--clips", default="", help="comma-separated subset")
    args = ap.parse_args()

    specs = json.loads((REPO / args.sources).read_text())["sources"]
    only = {c for c in args.clips.split(",") if c}
    plans = {}
    if args.mode == "finalize":
        plans = json.loads((REPO / args.plan).read_text())["clips"]

    for spec in specs:
        cid = spec["clip_id"]
        if only and cid not in only:
            continue
        out_dir = REPO / args.out_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "survey":
            r = survey(spec, out_dir)
            print(f"\n=== {cid} === {r['frames']} frames, green {r['green_pixels_total']} px, "
                  f"{len(r['components'])} components")
            for c in r["components"]:
                print(f"  [{c['component']}] area {c['area']:6d} bbox {c['bbox']} "
                      f"centroid {c['centroid_xy']} fill {c['fill_ratio_in_bbox']} "
                      f"rgb {c['colour_rgb']}")
            wt = r["worker_track"]
            print(f"  track: {wt['frames_with_worker']}/{r['frames']} frames, "
                  f"foot x {wt['foot_x_range']} y {wt['foot_y_range']}, "
                  f"step med {wt['median_step_px']} p90 {wt['p90_step_px']} max {wt['max_step_px']}")
            pf = r["perspective_fit"]
            print(f"  perspective: {pf}")
        else:
            r = finalize(spec, plans[cid], out_dir)
            ch = r["path_checks"]
            print(f"\n=== {cid} === lane {r['lane']['mask_pixels']} px from components "
                  f"{r['lane']['components_chosen']}")
            print(f"  path: frames relocated {ch['frames_relocated']}, "
                  f"max lateral step {ch['max_lateral_step_px']} px "
                  f"(limit {ch['speed_limit_px']}, max total {ch['max_total_step_px']}), "
                  f"teleports {ch['teleports']}, machinery {ch['crosses_machinery']}")
            print(f"  start offset from source foot {ch['start_offset_from_source_foot_px']} px, "
                  f"ends in lane {ch['ends_inside_lane_polygon']}, "
                  f"frames with foot in lane polygon {ch['frames_with_foot_inside_lane_polygon']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
