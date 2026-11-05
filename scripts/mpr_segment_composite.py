#!/usr/bin/env python
"""Segment the worker, recover background, and build the rough relocated composite.

This is the stage that actually moves the worker. CogVideoX is not involved: everything
here is detection, segmentation and array compositing, and the result is a deliberately
rough prototype whose failures are meant to stay visible.

What the composite is
---------------------
For every frame from the intervention onward: start from the source frame, erase the
worker where he was by pasting the temporal-median background over his mask, then paste his
own silhouette back at the target foot position. Frames before the intervention are copied
through untouched, so the violation is on screen before anything changes.

What the composite cannot be
----------------------------
Translating a silhouette does not synthesise a body turn, a new gait, a new viewing angle,
or a correct cast shadow. The worker arrives at the lane facing the way he was photographed
walking somewhere else. The contact shadow under his feet is removed with him only to the
extent that the dilated erase mask covers it, and no new shadow is drawn at the destination.
These are properties of the method, not bugs to be tuned away, and they are what the
refinement stage is then asked to improve.

Segmentation backends
---------------------
``sam2`` uses Meta's SAM 2.1 video predictor, seeded with the tracked worker's detector box
and propagated across the clip. ``maskrcnn`` uses torchvision's instance masks per frame,
matched to the track. Which one ran is recorded per clip; neither is assumed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from unsafenet_cf import manualpath as mp  # noqa: E402
from unsafenet_cf.clipset import iter_frames  # noqa: E402

TARGET_FPS = 8


# ------------------------------------------------------------------- detection + track

def detect_people(frames: np.ndarray, device, score_thresh: float = 0.5):
    import torch
    from eval_classifier import load_person_detector

    model, person_idx = load_person_detector(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(frames), 8):
            chunk = frames[start:start + 8].astype(np.float32) / 255.0
            batch = [torch.from_numpy(f).permute(2, 0, 1).to(device) for f in chunk]
            for det in model(batch):
                keep = (det["labels"] == person_idx) & (det["scores"] >= score_thresh)
                out.append([(det["boxes"][i].cpu().numpy().tolist(), float(det["scores"][i]))
                            for i in torch.nonzero(keep).flatten().tolist()])
    return out


# ------------------------------------------------------------------------ segmentation

def masks_sam2(frames: np.ndarray, seed_frame: int, seed_box, work_dir: Path, device) -> np.ndarray:
    """SAM 2.1 video propagation from one box prompt. Returns (T, H, W) bool."""
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    jpg_dir = work_dir / "jpg"
    if jpg_dir.exists():
        shutil.rmtree(jpg_dir)
    jpg_dir.mkdir(parents=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(jpg_dir / f"{i:05d}.jpg", quality=95)

    ckpt = Path("/ocean/projects/cis260288p/puprety/checkpoints/sam2/sam2.1_hiera_small.pt")
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml",
                                           str(ckpt), device=device)
    out = np.zeros((len(frames), frames.shape[1], frames.shape[2]), bool)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(jpg_dir))
        predictor.add_new_points_or_box(inference_state=state, frame_idx=seed_frame,
                                        obj_id=1, box=np.array(seed_box, np.float32))
        # Propagate forwards from the seed, then backwards, so frames on both sides of the
        # seed are covered. The predictor must be reset between the two sweeps.
        for idx, _ids, logits in predictor.propagate_in_video(state, start_frame_idx=seed_frame):
            out[idx] = (logits[0] > 0.0).cpu().numpy().squeeze()
        if seed_frame > 0:
            for idx, _ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=seed_frame, reverse=True):
                out[idx] = (logits[0] > 0.0).cpu().numpy().squeeze()
    return out


def masks_maskrcnn(frames: np.ndarray, track, device) -> np.ndarray:
    """Per-frame torchvision instance masks, matched to the tracked worker's foot point."""
    import torch
    from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights,
                                              maskrcnn_resnet50_fpn)

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).eval().to(device)
    person_idx = weights.meta["categories"].index("person")
    out = np.zeros((len(frames), frames.shape[1], frames.shape[2]), bool)
    with torch.no_grad():
        for i, f in enumerate(frames):
            if track[i] is None:
                continue
            tx, ty = track[i]["foot_xy"]
            t = torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            det = model([t])[0]
            keep = (det["labels"] == person_idx) & (det["scores"] >= 0.5)
            best, bd = None, None
            for j in torch.nonzero(keep).flatten().tolist():
                x1, y1, x2, y2 = det["boxes"][j].tolist()
                d = float(np.hypot((x1 + x2) / 2.0 - tx, y2 - ty))
                if bd is None or d < bd:
                    best, bd = j, d
            if best is not None and bd is not None and bd < 60.0:
                out[i] = det["masks"][best, 0].cpu().numpy() > 0.5
    return out


# --------------------------------------------------------------------------- utilities

def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r, written out because scipy is not in this environment."""
    if r <= 0:
        return mask
    m = mask.copy()
    for _ in range(r):
        p = np.pad(m, 1, mode="constant")
        m = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[1:-1, 1:-1])
    return m


def mask_stats(masks: np.ndarray, track) -> dict:
    areas = [int(m.sum()) for m in masks]
    empty = [i for i, a in enumerate(areas) if a == 0]
    tracked = [i for i, t in enumerate(track) if t is not None]
    missing_where_tracked = [i for i in tracked if areas[i] == 0]
    return {
        "frames": int(len(masks)),
        "frames_with_mask": int(sum(1 for a in areas if a > 0)),
        "empty_mask_frames": empty,
        "frames_tracked_but_unsegmented": missing_where_tracked,
        "area_px_min": int(min(areas)) if areas else None,
        "area_px_median": int(np.median(areas)) if areas else None,
        "area_px_max": int(max(areas)) if areas else None,
        "_reading": ("An empty mask is a segmentation failure, counted and never filled. "
                     "A sudden area change between neighbouring frames usually means the "
                     "mask lost a limb or grabbed a second person; the overlay sheet is the "
                     "evidence for that, not this number."),
    }


def contact_sheet(frames, masks, path, out_png, every=6):
    from PIL import Image
    tiles = []
    for i in range(0, len(frames), every):
        t = frames[i].copy()
        edge = masks[i] & ~dilate(masks[i], 0) if False else masks[i]
        t[edge] = (0.45 * t[edge].astype(np.float32)
                   + 0.55 * np.array([255, 64, 180], np.float32)).astype(np.uint8)
        if path[i] is not None:
            y = int(np.clip(round(path[i]["y"]), 0, t.shape[0] - 1))
            x = int(np.clip(round(path[i]["x"]), 0, t.shape[1] - 1))
            t[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = (255, 200, 0)
        tiles.append(t)
    rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_png)


def save_video(frames, path: Path):
    """Write uint8 RGB frames as H.264.

    The frames are handed over as PIL images on purpose. ``export_to_video`` rescales a
    list of numpy arrays with ``frame * 255``, which is right for float frames in [0, 1]
    and silently catastrophic for uint8: the multiply wraps modulo 256 and every frame
    comes out colour-inverted. Caught here by reading a written clip back and finding it
    matched the inverted source to 3.1/255 while differing from the source itself by
    84.6/255.
    """
    from PIL import Image
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video([Image.fromarray(np.ascontiguousarray(f)) for f in frames],
                    str(path), fps=TARGET_FPS)


# ---------------------------------------------------------------------------- the work

def run_clip(spec: dict, out_dir: Path, args, device) -> dict:
    t0 = time.time()
    frames = np.stack([f.copy() for f in iter_frames(REPO / spec["preprocessed"])])
    ann = json.loads((out_dir / "annotation.json").read_text())
    survey = json.loads((out_dir / "survey.json").read_text())
    track = survey["per_frame_track"]
    path = ann["per_frame_path"]

    # Seed SAM 2 on the frame whose tracked box is largest, which is the frame where the
    # worker is nearest the camera and least likely to be confused with a distant figure.
    seed_frame, seed_box, best_h = None, None, -1.0
    boxes = None
    if any(t is not None and t.get("box") for t in track):
        boxes = {t["frame"]: t["box"] for t in track if t is not None and t.get("box")}
    else:
        per_frame = detect_people(frames, device)
        from evaluate_multi_walkway import track_worker
        tr = track_worker(per_frame)
        boxes = {i: d[0] for i, d in enumerate(tr) if d is not None}
        for i, d in enumerate(tr):
            if d is None:
                track[i] = None
            else:
                (x1, y1, x2, y2), sc = d
                track[i] = {"frame": i, "foot_xy": [round((x1 + x2) / 2, 1), round(y2, 1)],
                            "height_px": round(y2 - y1, 1), "box": [round(v, 1) for v in d[0]],
                            "score": sc}
    for i, b in boxes.items():
        h = b[3] - b[1]
        if h > best_h:
            seed_frame, seed_box, best_h = i, b, h

    backend = args.backend
    seg_error = None
    try:
        if backend == "sam2":
            masks = masks_sam2(frames, seed_frame, seed_box,
                               Path(args.scratch) / spec["clip_id"], device)
        else:
            masks = masks_maskrcnn(frames, track, device)
    except Exception as exc:  # noqa: BLE001
        seg_error = f"{type(exc).__name__}: {exc}"
        print(f"  {spec['clip_id']}: {backend} failed ({seg_error}); falling back to maskrcnn",
              flush=True)
        backend = "maskrcnn-fallback"
        masks = masks_maskrcnn(frames, track, device)

    stats = mask_stats(masks, track)
    print(f"  masks: {stats['frames_with_mask']}/{stats['frames']} frames, "
          f"area med {stats['area_px_median']} px, backend {backend}", flush=True)

    # Background: exclude every person the detector found, not only the tracked worker, and
    # dilate so that mask edges and the contact shadow do not bleed into the estimate.
    per_frame_people = detect_people(frames, device)
    exclude = np.zeros(masks.shape, bool)
    for i, people in enumerate(per_frame_people):
        for b, _sc in people:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x1, y1 = max(0, x1 - 6), max(0, y1 - 6)
            x2 = min(frames.shape[2], x2 + 6)
            y2 = min(frames.shape[1], y2 + 6)
            exclude[i, y1:y2, x1:x2] = True
        exclude[i] |= dilate(masks[i], args.erase_dilate)
    bg, unknown = mp.temporal_background(frames, exclude)

    from PIL import Image
    Image.fromarray(bg).save(out_dir / "overlays" / "04_background_estimate.png")
    contact_sheet(frames, masks, path, out_dir / "overlays" / "05_worker_masks.png")

    composite = frames.copy()
    records = []
    erase = args.erase_dilate
    for i in range(len(frames)):
        p = path[i]
        if p is None or masks[i].sum() == 0:
            records.append({"frame": i, "relocated": False,
                            "reason": "before intervention" if p is None else "empty mask"})
            continue
        m = masks[i]
        ys, xs = np.nonzero(m)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        patch = frames[i, y0:y1, x0:x1].copy()
        alpha = m[y0:y1, x0:x1].astype(np.float32)
        src_foot = ((x0 + x1) / 2.0, float(y1))

        canvas = composite[i]
        em = dilate(m, erase)
        canvas[em] = bg[em]

        scale = 1.0  # follow_source holds the foot row, so the silhouette is never resized.
        res = mp.paste_silhouette(canvas, patch, alpha, (p["x"], p["y"]), scale)
        records.append({"frame": i, "relocated": bool(res.get("pasted")),
                        "source_foot_xy": [round(src_foot[0], 1), round(src_foot[1], 1)],
                        "target_foot_xy": [round(p["x"], 1), round(p["y"], 1)],
                        "shift_px": round(float(np.hypot(p["x"] - src_foot[0],
                                                         p["y"] - src_foot[1])), 1),
                        "mask_area_px": int(m.sum()), **res})

    save_video(composite, out_dir / "videos" / f"{spec['clip_id']}_composite.mp4")
    save_video(frames, out_dir / "videos" / f"{spec['clip_id']}_source.mp4")
    np.savez_compressed(out_dir / "masks.npz", masks=masks)

    relocated = sum(1 for r in records if r.get("relocated"))
    result = {
        "clip_id": spec["clip_id"],
        "segmentation": {
            "backend_requested": args.backend,
            "backend_used": backend,
            "error_if_any": seg_error,
            "seed_frame": seed_frame,
            "seed_box": seed_box,
            "checkpoint": ("sam2.1_hiera_small.pt sha256 6d1aa6f30de5c92224f8172114de081d"
                           "104bbd23dd9dc5c58996f0cad5dc4d38" if backend == "sam2"
                           else "torchvision maskrcnn_resnet50_fpn DEFAULT weights"),
            **stats,
        },
        "background": {
            "method": "per-pixel temporal median over frames where no person box and no "
                      "dilated worker mask covered the pixel",
            "person_box_pad_px": 6,
            "erase_dilate_px": erase,
            "pixels_with_no_evidence": int(unknown.sum()),
            "pixels_with_no_evidence_fraction": round(float(unknown.mean()), 5),
            "_fallback": "pixels with no evidence fall back to the whole-clip median, flagged",
            "_machinery_caveat": (
                "Presses and arms move. A median over the clip returns whichever state held "
                "for more than half the frames, so the estimate is a plausible still rather "
                "than a true empty room, and machinery caught mid-stroke may be flattened to "
                "its resting position."),
            "_inpainting": "none; no generative fill was used at this stage",
        },
        "composite": {
            "frames_relocated": relocated,
            "frames_untouched": int(len(frames) - relocated),
            "silhouette_rescaled": False,
            "per_frame": records,
            "_limits": [
                "Translating a photographed silhouette does not synthesise a body turn, a "
                "new gait, or a different viewing angle. The worker arrives at the lane "
                "facing the way he was photographed walking elsewhere.",
                "No cast shadow is drawn at the destination. The original contact shadow is "
                f"removed only as far as the {erase} px erase dilation reaches.",
                "No occluder is modelled. All three paths were drawn over open floor, so "
                "nothing in the scene should pass in front of the relocated worker; if "
                "something does, the composite will show him in front of it.",
            ],
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "composite.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="configs/manual_path_sources.json")
    ap.add_argument("--out-root", default="outputs/manual_path")
    ap.add_argument("--backend", choices=["sam2", "maskrcnn"], default="sam2")
    ap.add_argument("--erase-dilate", type=int, default=7)
    ap.add_argument("--scratch", default="/ocean/projects/cis260288p/puprety/scratch/mpr")
    ap.add_argument("--clips", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}", flush=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available; this must run in a Slurm job")

    specs = json.loads((REPO / args.sources).read_text())["sources"]
    only = {c for c in args.clips.split(",") if c}
    for spec in specs:
        if only and spec["clip_id"] not in only:
            continue
        print(f"\n=== {spec['clip_id']} ===", flush=True)
        out_dir = REPO / args.out_root / spec["clip_id"]
        r = run_clip(spec, out_dir, args, args.device)
        print(f"  composite: {r['composite']['frames_relocated']} frames relocated, "
              f"{r['runtime_s']}s", flush=True)
        if args.device == "cuda":
            print(f"  peak GPU {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
