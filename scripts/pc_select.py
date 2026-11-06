#!/usr/bin/env python
"""Score every class-2 and class-6 clip against the identified panel, before selecting.

The panel was identified from the class populations, not from a clip pair. An earlier
version of this script scored a wire-mesh press guard that a single pair of medians
suggested; measuring it over the whole folder showed the guard shut in 96 of 120 aligned
class-2 clips, which disproved the identification. `scripts/pc_population.py` replaced it:
it averages per-clip temporal medians within each class and differences the averages, so
whichever worker happens to stand in one clip averages away and whatever tracks the label
survives. The strongest surviving region is an electrical cabinet door on the press column,
open across the class-2 average and closed across the class-6 average. The history is in
DECISIONS.md P5 and P6, and the box is frozen in `configs/panel_cover_regions.json`.

This script measures that box clip by clip:

``alignment_ncc``     Correlation of a large static background patch with the reference
                      clip's. The box is in one camera's pixel coordinates, so it is
                      meaningless for a clip shot from elsewhere; a low value drops the clip
                      from the means and the rankings rather than scoring it.
``open_ncc``          Correlation of the panel region with the class-2 mean of that region.
``closed_ncc``        The same against the class-6 mean. Both are reported so neither is
                      read alone, and `panel_separation` is closed minus open.
``region_activity``   Mean absolute deviation from the clip's own temporal median inside the
                      region. This is the occlusion and stability test: a worker reaching
                      into the cabinet, or a door that swings during the clip, moves these
                      pixels. A source must hold still there for a donor patch to be pasted
                      in, so a high value disqualifies a clip however good its correlations.
``max_stride_...``    49 output frames at stride 3 need 145 source frames. Most clips in
                      this class pair are shorter, so the stride a clip supports is part of
                      its eligibility and not a detail.

The class means are built from the aligned clips measured in this same run, so every clip is
scored partly against itself. With 100 and 12 clips that self-contribution is 1% and 8% of
the respective means; it inflates each clip's correlation with its own class slightly and so
makes the separation optimistic, which is why the ranking is evidence for a hand-frozen plan
and not a selection. Nothing is selected here.
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
from unsafenet_cf import panelpatch as pp    # noqa: E402

OUT = REPO / "outputs" / "panel_cover"
PANEL_CLASSES = (2, 6)

SHEET_COLS = 8          # contact-sheet columns; ids are logged in row-major order
CONTEXT_BOX = [0, 500, 760, 580]    # x, y, w, h: the press front and the floor around it


def log(m: str) -> None:
    print(m, flush=True)


def sampled_median(path: Path, stride: int, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (median frame, mean absolute deviation per pixel) at full resolution."""
    frames = []
    for i, f in enumerate(cs.iter_frames(path)):
        if i % stride == 0:
            frames.append(f.copy())
        if len(frames) >= cap:
            break
    if not frames:
        return np.zeros((0,)), np.zeros((0,))
    stack = np.stack(frames).astype(np.float32)
    grey = stack.mean(axis=3)
    med = np.median(stack, axis=0)
    dev = np.abs(grey - np.median(grey, axis=0)).mean(axis=0)
    return med.astype(np.uint8), dev.astype(np.float32)


def region(img: np.ndarray, box) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    return img[y:y + h, x:x + w]


def save_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def contact_sheet(crops: list[np.ndarray], cols: int = SHEET_COLS,
                  scale: int = 1) -> np.ndarray:
    """Tile equal-sized crops row-major on a grey background, 4 px gutters."""
    if not crops:
        return np.zeros((1, 1, 3), np.uint8)
    h, w = crops[0].shape[:2]
    rows = (len(crops) + cols - 1) // cols
    gap = 4
    sheet = np.full((rows * (h + gap) + gap, cols * (w + gap) + gap, 3), 90, np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        y, x = gap + r * (h + gap), gap + k * (w + gap)
        sheet[y:y + h, x:x + w] = c
    if scale > 1:
        sheet = np.repeat(np.repeat(sheet, scale, axis=0), scale, axis=1)
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="configs/panel_cover_regions.json")
    ap.add_argument("--roots", nargs="+",
                    default=["data/raw/class2_train", "data/raw/class6_train"])
    ap.add_argument("--out", default="outputs/panel_cover/selection.json")
    ap.add_argument("--sheets", default="outputs/panel_cover/selection")
    ap.add_argument("--stride", type=int, default=8,
                    help="decode every Nth frame when building the median")
    ap.add_argument("--cap", type=int, default=24, help="frames kept per clip")
    args = ap.parse_args()

    cfg = json.loads((REPO / args.regions).read_text())
    view = cfg["views"]["press_front"]
    panel_box = view["panel_box_full_res"]
    align_box = view["alignment_patch_full_res"]
    align_ref_id = view["alignment_reference_clip"]
    min_align = float(cfg.get("alignment_min_ncc", 0.95))

    roots = [REPO / r for r in args.roots]
    clips = cs.discover(roots, keep_classes=PANEL_CLASSES)
    log(f"clips: {len(clips)}")
    log(f"panel box (full res x,y,w,h): {panel_box}")
    log(f"alignment patch: {align_box}, reference {align_ref_id}, min ncc {min_align}")

    by_id = {c.clip_id: c for c in clips}
    if align_ref_id not in by_id:
        log(f"alignment reference clip {align_ref_id} not found under the given roots")
        return 2
    ref_med, _ = sampled_median(by_id[align_ref_id].path, args.stride, args.cap)
    align_ref = region(ref_med, align_box).mean(axis=2)

    # Pass 1: one median per clip, plus the alignment test.
    meds: dict[str, np.ndarray] = {}
    rows = []
    t0 = time.time()
    order = sorted(clips, key=lambda c: (c.dataset_class, c.index))
    for i, c in enumerate(order, 1):
        media = cs.ffprobe_media(c.path)
        med, dev = sampled_median(c.path, args.stride, args.cap)
        if med.size == 0:
            rows.append({"clip_id": c.clip_id, "error": "decoded 0 frames"})
            continue
        frames_total = media.get("frame_count") or 0
        meds[c.clip_id] = med
        rows.append({
            "clip_id": c.clip_id,
            "dataset_class": c.dataset_class,
            "path": str(c.path.relative_to(REPO)),
            "frames": frames_total,
            "fps": media.get("fps"),
            "max_stride_for_49_frames": (frames_total - 1) // 48 if frames_total else 0,
            "alignment_ncc": round(pp.ncc(region(med, align_box).mean(axis=2), align_ref), 5),
            "region_activity": round(float(region(dev[..., None], panel_box).mean()), 4),
            "frame_activity": round(float(dev.mean()), 4),
        })
        if i % 25 == 0 or i == len(order):
            log(f"  medians {i}/{len(order)} in {time.time() - t0:.0f}s")

    aligned = [r for r in rows if "error" not in r and r["alignment_ncc"] >= min_align]
    log(f"\nclips aligned to this camera: {len(aligned)}/{len(rows)}")

    # Pass 2: class means of the panel region, then score every aligned clip against both.
    means = {}
    for cls in PANEL_CLASSES:
        ids = [r["clip_id"] for r in aligned if r["dataset_class"] == cls]
        if not ids:
            log(f"no aligned clips in class {cls}; cannot build a mean")
            return 3
        means[cls] = np.mean([region(meds[i], panel_box).astype(np.float32) for i in ids],
                             axis=0)
        log(f"class {cls} panel mean from {len(ids)} clips")
    open_ref = means[2].mean(axis=2)
    closed_ref = means[6].mean(axis=2)
    log(f"panel region {means[2].shape}, class mean separation ncc "
        f"{pp.ncc(open_ref, closed_ref):.4f}")

    for r in aligned:
        panel = region(meds[r["clip_id"]], panel_box).mean(axis=2)
        r["open_ncc"] = round(pp.ncc(panel, open_ref), 5)
        r["closed_ncc"] = round(pp.ncc(panel, closed_ref), 5)
        r["panel_separation"] = round(r["closed_ncc"] - r["open_ncc"], 5)

    def rank_sources(rs):
        # An open panel correlates better with the class-2 mean than the class-6 mean, and
        # the region must hold still for a donor patch to be pasted into it.
        return sorted([r for r in rs if r["dataset_class"] == 2],
                      key=lambda r: (r["panel_separation"], r["region_activity"]))

    def rank_donors(rs):
        return sorted([r for r in rs if r["dataset_class"] == 6],
                      key=lambda r: (-r["panel_separation"], r["region_activity"]))

    src_rank = rank_sources(aligned)
    don_rank = rank_donors(aligned)
    head = (f"{'clip':10s} {'open':>8s} {'closed':>8s} {'sep':>8s} {'act':>7s} "
            f"{'frames':>6s} {'stride':>6s} {'align':>7s}")

    def show(r):
        log(f"{r['clip_id']:10s} {r['open_ncc']:8.4f} {r['closed_ncc']:8.4f} "
            f"{r['panel_separation']:8.4f} {r['region_activity']:7.3f} "
            f"{r['frames']:6d} {r['max_stride_for_49_frames']:6d} {r['alignment_ncc']:7.4f}")

    log("\ntop 20 class-2 source candidates (most open first, then quietest region)")
    log(head)
    for r in src_rank[:20]:
        show(r)
    log("\nall class-6 donor candidates (most closed first)")
    log(head)
    for r in don_rank:
        show(r)
    log("\nclass-2 clips whose panel reads closed (separation > 0), i.e. label without "
        f"this door open: {sum(1 for r in src_rank if r['panel_separation'] > 0)}"
        f"/{len(src_rank)}")

    # Contact sheets, so the ranking is checked by eye before anything is frozen.
    sheets = REPO / args.sheets
    pad = 24
    px, py, pw, ph = panel_box
    wide = [max(0, px - pad), max(0, py - pad), pw + 2 * pad, ph + 2 * pad]
    sheet_log = {}
    for name, rs in (("class2_sources", src_rank), ("class6_donors", don_rank)):
        crops = [region(meds[r["clip_id"]], wide) for r in rs]
        save_png(sheets / f"{name}_panel.png", contact_sheet(crops, scale=2))
        sheet_log[name] = [r["clip_id"] for r in rs]
    top = src_rank[:SHEET_COLS] + don_rank[:SHEET_COLS]
    save_png(sheets / "context.png",
             contact_sheet([region(meds[r["clip_id"]], CONTEXT_BOX) for r in top], cols=4))
    sheet_log["context"] = [r["clip_id"] for r in top]
    save_png(sheets / "mean_panel_class2.png", means[2])
    save_png(sheets / "mean_panel_class6.png", means[6])
    log(f"\nsheets: {sheets} (row-major clip order is in selection.json under sheet_order)")

    report = {
        "_what": ("Per-clip measurements of the identified panel region against class means. "
                  "Nothing is selected here; the ranked lists are evidence for a plan that "
                  "is frozen by hand afterwards."),
        "generated": mf.utc_now(),
        "region_config": cfg,
        "sampling": {"stride": args.stride, "cap": args.cap,
                     "_note": "Medians are built from sampled frames, not every frame."},
        "class_mean_clip_counts": {str(c): sum(1 for r in aligned
                                               if r["dataset_class"] == c)
                                   for c in PANEL_CLASSES},
        "class_mean_separation_ncc": round(pp.ncc(open_ref, closed_ref), 5),
        "clips": rows,
        "aligned_clip_count": len(aligned),
        "source_ranking": [r["clip_id"] for r in src_rank],
        "donor_ranking": [r["clip_id"] for r in don_rank],
        "sheet_order": sheet_log,
        "sheet_boxes": {"panel_with_margin_full_res": wide, "context_full_res": CONTEXT_BOX},
        "_caveats": [
            "A correlation against a class mean is not a panel-state detector. Every clip "
            "that reaches the plan is looked at in the contact sheets first.",
            "Each clip contributes to the mean it is scored against, so the separation is "
            "optimistic. With 100 class-2 and 12 class-6 aligned clips the self-contribution "
            "is about 1% and 8% of the mean respectively.",
            "region_activity cannot tell a worker in front of the cabinet from the door "
            "itself swinging; both disqualify a source.",
            "Clips below the alignment threshold are measured but excluded from the means "
            "and the rankings: the region box is in one camera's coordinates.",
        ],
        "environment": mf.environment_snapshot(),
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1))
    log(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
