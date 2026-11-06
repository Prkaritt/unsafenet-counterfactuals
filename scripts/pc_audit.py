#!/usr/bin/env python
"""Audit the class-2 and class-6 training folders before anything is selected.

The panel-cover pair is structurally unlike the walkway pair this project handled before.
There the label was a property of a person's position; here it is a property of an object's
state, so the questions this audit has to answer are different:

1. **Which camera view is each clip from?** The dataset is two fixed cameras. A closed-panel
   donor is only usable if it shows the same scene as the open-panel source, so the views
   have to be separated before any donor is proposed.
2. **Where is the panel?** No annotation exists. Rather than invent geometry, the candidate
   region is *measured*: the per-pixel difference between a class-2 clip's own median
   background and a class-6 clip's median background from the same view is large exactly
   where the scene differs, and the panel is the thing that differs by construction of the
   labels. The audit reports those regions as candidates for inspection; it does not decide
   that any of them is a panel.
3. **Which clips share a recording?** Fixed-camera footage of one factory means clips from
   one camera look alike. Recording identity is decided on frame overlap, using the measured
   two-tier mean-absolute-difference test calibrated in `docs/GROUPING_CALIBRATION.md`, not
   on scene similarity.
4. **Is the panel region temporally stable?** A region that is occluded by a passing worker,
   or that moves, cannot be replaced by a fixed donor patch. The audit measures it instead
   of assuming it.

Nothing here selects a source, a donor or an interval. That happens in `pc_select.py`, after
a human has looked at the contact sheets this writes.

Decoding 148 clips is substantial media processing, so this runs on a compute node.
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

from unsafenet_cf import clipset as cs      # noqa: E402
from unsafenet_cf import grouping as gp     # noqa: E402
from unsafenet_cf import manifest as mf     # noqa: E402
from unsafenet_cf import manualpath as mp   # noqa: E402

PANEL_CLASSES = (2, 6)
CLASS_NAMES = {2: "Opened Panel Cover", 6: "Closed Panel Cover"}

# Median-background size for view clustering and panel differencing. Big enough that a
# panel of a few hundred source pixels survives, small enough for an all-pairs comparison.
BG_H, BG_W = 270, 480

# Contact-sheet frame size and how many frames per clip.
SHEET_H, SHEET_W = 180, 320
SHEET_N = 6

# Difference-map smoothing window, in BG_W pixels. A box mean, because scipy is absent.
SMOOTH = 9
# A difference blob smaller than this many BG-scale pixels is noise, not a panel.
MIN_BLOB = 60


def log(m: str) -> None:
    print(m, flush=True)


# --- small numpy helpers ---------------------------------------------------------


def box_mean(plane: np.ndarray, k: int) -> np.ndarray:
    """Separable box mean with edge padding. k is forced odd."""
    k = max(1, int(k) | 1)
    r = k // 2
    pad = np.pad(plane.astype(np.float32), r, mode="edge")
    acc = np.cumsum(pad, axis=0)
    acc = np.vstack([np.zeros((1, acc.shape[1]), np.float32), acc])
    plane = (acc[k:, :] - acc[:-k, :]) / k
    acc = np.cumsum(plane, axis=1)
    acc = np.hstack([np.zeros((acc.shape[0], 1), np.float32), acc])
    return (acc[:, k:] - acc[:, :-k]) / k


def resize_nn(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    ys = (np.arange(out_h) * (img.shape[0] / out_h)).astype(int).clip(0, img.shape[0] - 1)
    xs = (np.arange(out_w) * (img.shape[1] / out_w)).astype(int).clip(0, img.shape[1] - 1)
    return img[ys][:, xs]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float32).ravel() - a.mean()
    y = b.astype(np.float32).ravel() - b.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 1e-6 else 0.0


def save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8))).save(path)


def label_strip(width: int, text: str, height: int = 16) -> np.ndarray:
    """A black caption strip with white text, drawn with PIL's default font."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), (0, 0, 0))
    ImageDraw.Draw(img).text((3, 2), text[: max(1, width // 6)], fill=(255, 255, 255))
    return np.array(img)


# --- per-clip decoding -----------------------------------------------------------


def read_clip(path: Path) -> dict:
    """Decode one clip once and keep only the reductions the audit needs."""
    full = []
    for frame in cs.iter_frames(path, width=BG_W, height=BG_H):
        full.append(frame.copy())
    if not full:
        return {}
    stack = np.stack(full)                        # (T, BG_H, BG_W, 3) uint8
    grey = stack.astype(np.float32).mean(axis=3)
    med = np.median(stack, axis=0)                # colour median background
    med_grey = np.median(grey, axis=0)

    # Dense greyscale sample for the duplicate scan, at the calibrated size and stride.
    idx = list(range(0, len(stack), gp.DENSE_STRIDE))[: gp.DENSE_CAP]
    dense = np.stack([resize_nn(grey[i], gp.DENSE_H, gp.DENSE_W) for i in idx]
                     ).astype(np.uint8)

    sheet_idx = np.linspace(0, len(stack) - 1, min(SHEET_N, len(stack))).astype(int)
    return {
        "n_frames_decoded": int(len(stack)),
        "median_bgr": med.astype(np.uint8),
        "median_grey": med_grey.astype(np.float32),
        "dense": dense,
        "sheet": np.stack([resize_nn(stack[i], SHEET_H, SHEET_W) for i in sheet_idx]),
        "sheet_idx": [int(i) for i in sheet_idx],
        # Temporal activity: how much each pixel moves relative to its own median.
        "activity": np.abs(grey - med_grey).mean(axis=0).astype(np.float32),
    }


# --- panel candidate regions -----------------------------------------------------


def difference_boxes(med_a: np.ndarray, med_b: np.ndarray, thresh: float) -> list[dict]:
    """Candidate changed regions between two median backgrounds, largest first."""
    diff = box_mean(np.abs(med_a - med_b), SMOOTH)
    mask = diff >= thresh
    comps = mp.connected_components(mask, min_area=MIN_BLOB)
    out = []
    for c in comps:
        ys, xs = c["pixels"][:, 0], c["pixels"][:, 1]
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        out.append({
            "area_px": int(c["area"]),
            "box_bg_scale": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
            "mean_diff_in_box": round(float(diff[y0:y1 + 1, x0:x1 + 1].mean()), 2),
            "max_diff_in_box": round(float(diff[y0:y1 + 1, x0:x1 + 1].max()), 2),
            "centroid_bg_scale": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        })
    return sorted(out, key=lambda d: -d["area_px"])


def heatmap(diff: np.ndarray) -> np.ndarray:
    """Grey base with the difference painted red, for quick visual inspection."""
    d = diff / max(1e-6, float(diff.max()))
    out = np.zeros(diff.shape + (3,), np.float32)
    out[..., 0] = 255 * d
    out[..., 1] = 60 * d
    out[..., 2] = 60 * d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["data/raw/class2_train", "data/raw/class6_train"])
    ap.add_argument("--out", default="outputs/panel_cover/audit")
    ap.add_argument("--diff-threshold", type=float, default=18.0,
                    help="grey levels of smoothed median difference counted as changed; "
                         "reported alongside the measured distribution so it can be "
                         "revised from evidence rather than kept because it was the default")
    ap.add_argument("--view-threshold", type=float, default=0.95,
                    help="median-background correlation above which two clips are taken "
                         "to be the same camera view; the full matrix is written out")
    ap.add_argument("--limit", type=int, default=0, help="debug: audit only N clips")
    args = ap.parse_args()

    out = REPO / args.out
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    (out / "diffs").mkdir(parents=True, exist_ok=True)

    roots = [REPO / r for r in args.roots]
    missing = [str(r) for r in roots if not r.exists()]
    if missing:
        log(f"missing input folders: {missing}")
        log("see bridges2/DATASET_ACCESS.md: the class-2 and class-6 training folders "
            "need one file link each from the Mendeley web UI")
        return 2

    clips = cs.discover(roots, keep_classes=PANEL_CLASSES)
    if args.limit:
        clips = clips[: args.limit]
    by_class: dict[int, list[str]] = {2: [], 6: []}
    for c in clips:
        by_class[c.dataset_class].append(c.clip_id)
    log(f"clips: {len(clips)} "
        f"(class 2: {len(by_class[2])}, class 6: {len(by_class[6])})")
    if not clips:
        log("no clips matched the publisher filename convention; nothing audited")
        return 2

    t0 = time.time()
    data: dict[str, dict] = {}
    records: dict[str, dict] = {}
    for i, c in enumerate(clips, 1):
        media = cs.ffprobe_media(c.path)
        d = read_clip(c.path)
        if not d:
            log(f"  {c.clip_id}: decoded 0 frames; skipped")
            continue
        data[c.clip_id] = d
        records[c.clip_id] = {
            "clip_id": c.clip_id,
            "dataset_class": c.dataset_class,
            "class_name": CLASS_NAMES[c.dataset_class],
            "official_split": c.official_split,
            "path": str(c.path.relative_to(REPO)),
            "sha256": mf.sha256_file(c.path),
            "media": media,
            "frames_decoded": d["n_frames_decoded"],
        }
        sheet = np.concatenate(
            [np.vstack([label_strip(SHEET_W, f"{c.clip_id} f{j}"), f_]).astype(np.uint8)
             for f_, j in zip(d["sheet"], d["sheet_idx"])], axis=1)
        save_png(sheet, out / "sheets" / f"{c.clip_id}.png")
        if i % 10 == 0 or i == len(clips):
            log(f"  decoded {i}/{len(clips)} in {time.time() - t0:.0f}s")

    ids = sorted(data)

    # --- camera views ------------------------------------------------------------
    log("clustering camera views from median backgrounds")
    view_sim = {}
    edges_at = {t: [] for t in (0.80, 0.90, 0.95, 0.98)}
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            s = corr(data[a]["median_grey"], data[b]["median_grey"])
            view_sim[f"{a}|{b}"] = round(s, 4)
            for t in edges_at:
                if s >= t:
                    edges_at[t].append((a, b))
    views = gp.connected_components(ids, edges_at[args.view_threshold])
    view_counts: dict[str, int] = {}
    for v in views.values():
        view_counts[v] = view_counts.get(v, 0) + 1
    log(f"views at correlation >= {args.view_threshold}: {len(view_counts)} "
        f"({sorted(view_counts.values(), reverse=True)})")
    for t, e in edges_at.items():
        g = gp.connected_components(ids, e)
        log(f"  threshold {t:.2f}: {len(set(g.values()))} views, {len(e)} edges")

    # --- recording groups --------------------------------------------------------
    log("scanning for shared frames")
    certain, probable = gp.pairwise_duplicate_scan({k: data[k]["dense"] for k in ids})
    log(f"certain shared-frame pairs (min MAD <= {gp.MAD_CERTAIN}): {len(certain)}")
    log(f"probable same-recording pairs (<= {gp.MAD_PROBABLE}): {len(probable)}")
    rec = gp.connected_components(
        ids, [(a, b) for a, b, _ in certain] + [(a, b) for a, b, _ in probable])
    rec_members: dict[str, list[str]] = {}
    for cid, g in rec.items():
        rec_members.setdefault(g, []).append(cid)

    # --- cross-class panel candidates -------------------------------------------
    log("measuring changed regions between class-2 and class-6 medians, within a view")
    diff_values: list[float] = []
    pair_reports = []
    for a in [c for c in ids if records[c]["dataset_class"] == 2]:
        for b in [c for c in ids if records[c]["dataset_class"] == 6]:
            if views[a] != views[b]:
                continue
            d = box_mean(np.abs(data[a]["median_grey"] - data[b]["median_grey"]), SMOOTH)
            diff_values.append(float(np.percentile(d, 99)))
            boxes = difference_boxes(data[a]["median_grey"], data[b]["median_grey"],
                                     args.diff_threshold)
            if not boxes:
                continue
            top = boxes[0]
            x, y, w, h = top["box_bg_scale"]
            act2 = float(data[a]["activity"][y:y + h, x:x + w].mean())
            act6 = float(data[b]["activity"][y:y + h, x:x + w].mean())
            pair_reports.append({
                "class2_clip": a, "class6_clip": b, "view": views[a],
                "median_diff_p99": round(diff_values[-1], 2),
                "n_candidate_regions": len(boxes),
                "largest_region": top,
                "activity_in_region": {
                    "class2_mean_abs_dev_from_own_median": round(act2, 3),
                    "class6_mean_abs_dev_from_own_median": round(act6, 3),
                    "_note": ("Low values mean the region holds still for the whole clip, "
                              "which a fixed donor patch needs. High values mean something "
                              "moves through it."),
                },
                "same_recording": rec[a] == rec[b],
            })
            stack = np.vstack([
                label_strip(BG_W, f"{a} median (class 2, open)"), data[a]["median_bgr"],
                label_strip(BG_W, f"{b} median (class 6, closed)"), data[b]["median_bgr"],
                label_strip(BG_W, f"|difference| x{len(boxes)} regions, top box {top['box_bg_scale']}"),
                heatmap(d),
            ])
            save_png(stack, out / "diffs" / f"{a}__{b}.png")
    log(f"cross-class same-view pairs measured: {len(pair_reports)}")

    report = {
        "_what": ("Audit of the class-2 and class-6 training folders. Measurements only: "
                  "no source, donor or interval is selected here."),
        "_class_definitions": "docs/PANEL_COVER_DEFINITIONS.md",
        "generated": mf.utc_now(),
        "clip_count": {"class_2": len(by_class[2]), "class_6": len(by_class[6])},
        "clips": records,
        "camera_views": {
            "threshold": args.view_threshold,
            "assignment": views,
            "group_sizes": view_counts,
            "components_by_threshold": {
                str(t): len(set(gp.connected_components(ids, e).values()))
                for t, e in edges_at.items()},
            "similarity": view_sim,
            "_method": ("Pearson correlation between per-clip median backgrounds at "
                        f"{BG_W}x{BG_H} greyscale. This answers 'same room?', which is "
                        "exactly what a camera view is, and is NOT used for recording "
                        "identity."),
        },
        "recording_groups": {
            "assignment": rec,
            "members": rec_members,
            "certain_pairs": [{"a": a, "b": b, "min_mad": round(d, 4)}
                              for a, b, d in certain],
            "probable_pairs": [{"a": a, "b": b, "min_mad": round(d, 4)}
                               for a, b, d in probable],
            "thresholds": {"mad_certain": gp.MAD_CERTAIN,
                           "mad_probable": gp.MAD_PROBABLE},
            "_method": "docs/GROUPING_CALIBRATION.md, unchanged from the class-0/4 audit.",
        },
        "panel_candidates": {
            "diff_threshold": args.diff_threshold,
            "smoothing_box_px": SMOOTH,
            "min_blob_px": MIN_BLOB,
            "bg_scale": [BG_W, BG_H],
            "p99_distribution": {
                "n": len(diff_values),
                "min": round(min(diff_values), 2) if diff_values else None,
                "median": round(float(np.median(diff_values)), 2) if diff_values else None,
                "max": round(max(diff_values), 2) if diff_values else None,
            },
            "pairs": pair_reports,
            "_warning": ("A changed region between two clips is evidence that something "
                         "differs, not that the thing is a panel cover. Every region here "
                         "is a candidate for human inspection of the written PNGs."),
        },
        "environment": mf.environment_snapshot(),
    }
    (out / "audit.json").write_text(json.dumps(report, indent=1))
    log(f"\nwrote {out / 'audit.json'}")
    log(f"contact sheets: {out / 'sheets'}")
    log(f"difference maps: {out / 'diffs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
