#!/usr/bin/env python
"""Build the closed-panel composite for each frozen case.

What this produces, stated so no reader has to infer it: a clip identical to the
preprocessed source everywhere except inside one annotated region, where the pixels come
from a donor clip that shows the same machine with its panel cover **closed**. The cover is
closed in every frame of the output. **No closing motion is generated.** Nothing here models
a hinge, a hand, or the interior of the cabinet beyond what the donor itself shows.

The donor patch is taken from the donor clip's temporal median by default. A median is the
right default for this intervention because the panel is supposed to be static and a median
removes anyone who walked past the donor's panel while it was being filmed. ``per_frame``
is available and records itself in the manifest, but it carries the donor's own lighting
flicker into the source clip and is not the default for that reason.

Four things are measured and written into the per-case record, rather than assumed:

* the per-landmark reprojection error of the registration fit,
* whether any pixel outside the blended region changed,
* the seam step across the region boundary, against a control boundary in untouched scene,
* how much the source's own pixels moved inside the region over the clip, because a region
  that something walks through cannot be replaced by a still patch.

This is **mask-restricted compositing**, not mask-conditioned generation. No model is
involved at this stage at all.
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

from unsafenet_cf import clipset as cs        # noqa: E402
from unsafenet_cf import manifest as mf       # noqa: E402
from unsafenet_cf import manualpath as mp     # noqa: E402
from unsafenet_cf import panelpatch as pp     # noqa: E402
from unsafenet_cf import preprocess as prep   # noqa: E402

OUT = REPO / "outputs" / "panel_cover"


def log(m: str) -> None:
    print(m, flush=True)


def save_video(frames: list[np.ndarray], path: Path) -> None:
    """Write uint8 frames with export_to_video, via PIL.

    Passing numpy arrays straight in is a trap this project already paid for: diffusers
    rescales a numpy frame by 255 and uint8 wraps modulo 256, producing an exact colour
    negative. See DECISIONS.md D7.
    """
    from PIL import Image
    from diffusers.utils import export_to_video
    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video([Image.fromarray(np.ascontiguousarray(f)) for f in frames],
                    str(path), fps=prep.TARGET_FPS)


def decode(path: Path) -> np.ndarray:
    return np.stack(list(cs.iter_frames(path)))


def spec_from(case: dict, key: str, source_path: Path) -> prep.PreprocessSpec:
    seg = case[key]
    crop = case["crop"]
    return prep.PreprocessSpec(
        source=str(source_path),
        crop=prep.CropBox(crop["width"], crop["height"], crop["x"], crop["y"]),
        start_frame=seg["start"], frame_step=seg["step"],
        num_frames=seg.get("count", prep.SAMPLE_FRAMES))


def find_clip(clip_id: str, roots: list[Path]) -> Path:
    for r in roots:
        p = r / f"{clip_id}.mp4"
        if p.exists():
            return p
    raise FileNotFoundError(f"{clip_id}.mp4 not found under {[str(r) for r in roots]}")


def outline(mask: np.ndarray) -> np.ndarray:
    return pp.ring(mask, 1, outside=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/panel_cover_plan.json")
    ap.add_argument("--roots", nargs="+",
                    default=["data/raw/class2_train", "data/raw/class6_train"])
    ap.add_argument("--cases", default="", help="comma-separated subset of case ids")
    ap.add_argument("--out", default="outputs/panel_cover")
    args = ap.parse_args()

    plan = json.loads((REPO / args.plan).read_text())
    roots = [REPO / r for r in args.roots]
    only = {c for c in args.cases.split(",") if c}
    out_root = REPO / args.out
    results: dict[str, dict] = {}

    for case in plan["cases"]:
        cid = case["case_id"]
        if only and cid not in only:
            continue
        t0 = time.time()
        log(f"\n=== {cid}: source {case['source_clip']}, donor {case['donor_clip']}")
        case_dir = out_root / cid
        (case_dir / "videos").mkdir(parents=True, exist_ok=True)
        (case_dir / "overlays").mkdir(parents=True, exist_ok=True)

        # --- preprocess both arms through the identical path ----------------------
        src_path = find_clip(case["source_clip"], roots)
        don_path = find_clip(case["donor_clip"], roots)
        src_rec = prep.preprocess_clip(spec_from(case, "source_segment", src_path),
                                       case_dir / "videos" / f"{cid}_source.mp4")
        don_rec = prep.preprocess_clip(spec_from(case, "donor_segment", don_path),
                                       case_dir / "videos" / f"{cid}_donor.mp4")
        src = decode(case_dir / "videos" / f"{cid}_source.mp4")
        don = decode(case_dir / "videos" / f"{cid}_donor.mp4")
        log(f"  source {src.shape}, donor {don.shape}")

        # --- the region, from the frozen plan ------------------------------------
        poly = np.asarray(case["panel_polygon"], dtype=float)
        mask = mp.polygon_mask(poly, src.shape[1:3])
        feather_px = int(case.get("feather_px", 2))
        alpha = pp.feather(mask, feather_px)
        blended = alpha > 0.0
        log(f"  region {int(mask.sum())} px, feather {feather_px} px, "
            f"blended {int(blended.sum())} px")

        # --- registration from landmark patches ----------------------------------
        src_med = src.astype(np.float64).mean(axis=0).mean(axis=2)
        don_med = don.astype(np.float64).mean(axis=0).mean(axis=2)
        patches = [tuple(p) for p in case["landmark_patches"]]
        for (x, y, w, h) in patches:
            box = np.zeros_like(mask)
            box[y:y + h, x:x + w] = True
            if (box & blended).any():
                raise SystemExit(
                    f"{cid}: landmark patch {(x, y, w, h)} overlaps the blended region; "
                    "a landmark must sit on scene structure that is NOT replaced")
        offs = pp.patch_offsets(src_med, don_med, patches,
                               search=int(case.get("search_px", 12)))
        good = [o for o in offs
                if o.get("ok") and o["ncc"] >= case.get("min_ncc", 0.70)]
        log(f"  landmarks: {len(good)}/{len(offs)} above ncc "
            f"{case.get('min_ncc', 0.70)}")
        for o in offs:
            log(f"    {o['patch']}: " + (f"offset {o['offset_xy']} ncc {o['ncc']}"
                                         if o.get("ok") else o.get("reason", "failed")))

        kind = case.get("transform", "translation")
        need = {"translation": 1, "similarity": 2, "affine": 3, "homography": 4}[kind]
        if len(good) < need:
            results[cid] = {"error": f"{kind} needs {need} landmarks, {len(good)} usable",
                            "landmarks": offs}
            log(f"  FAILED: {results[cid]['error']}")
            continue
        # The donor is the thing being moved, so the fit maps donor to source. The offsets
        # were measured as source -> donor, hence the direction is reversed here.
        dst_pts = [o["centre_src_xy"] for o in good]
        src_pts = [o["centre_dst_xy"] for o in good]
        fit = {"translation": pp.estimate_translation, "similarity": pp.estimate_similarity,
               "affine": pp.estimate_affine, "homography": pp.estimate_homography}[kind](
                   src_pts, dst_pts)
        log(f"  {kind} fit: residual rms {fit['residual_px']['rms']} px, "
            f"max {fit['residual_px']['max']} px, dof {fit['degrees_of_freedom']}")

        # --- donor patch ----------------------------------------------------------
        policy = case.get("donor_frame_policy", "median")
        if policy == "median":
            donor_frames = [np.median(don, axis=0).astype(np.uint8)] * len(src)
        elif policy == "per_frame":
            if len(don) < len(src):
                results[cid] = {"error": f"per_frame needs {len(src)} donor frames, "
                                         f"donor has {len(don)}"}
                log(f"  FAILED: {results[cid]['error']}")
                continue
            donor_frames = [don[i] for i in range(len(src))]
        else:
            raise SystemExit(f"{cid}: unknown donor_frame_policy {policy!r}")

        warped0, valid = pp.warp(donor_frames[0], fit["_matrix"], src.shape[1:3])
        covered = float((valid & blended).sum()) / max(1, int(blended.sum()))
        log(f"  donor coverage of the blended region: {covered:.4f}")
        if covered < 0.999:
            results[cid] = {"error": "warped donor does not cover the whole region "
                                     f"({covered:.4f}); registration or region is wrong",
                            "transform": {k: v for k, v in fit.items()
                                          if not k.startswith('_')}}
            log(f"  FAILED: {results[cid]['error']}")
            continue

        # --- composite ------------------------------------------------------------
        comp = []
        for i, frame in enumerate(src):
            w, _ = pp.warp(donor_frames[i], fit["_matrix"], src.shape[1:3])
            comp.append(pp.composite(frame, w, alpha))
        comp_arr = np.stack(comp)
        save_video(comp, case_dir / "videos" / f"{cid}_composite.mp4")

        clean = pp.unchanged_outside_mask(src[0], comp_arr[0], blended)
        seam_src = pp.seam_contrast(src[0], mask)
        seam_comp = pp.seam_contrast(comp_arr[0], mask)
        stab_src = pp.region_stability(src, mask)
        stab_comp = pp.region_stability(comp_arr, mask)
        log(f"  outside region untouched: {clean['clean']} "
            f"(changed {clean['pixels_changed_outside_mask']} px)")
        log(f"  seam step: source {seam_src.get('boundary_step')}, "
            f"composite {seam_comp.get('boundary_step')}, "
            f"control {seam_comp.get('control_step')}")
        log(f"  region movement (mean abs dev from median): source "
            f"{stab_src['mad_in_mask']}, composite {stab_comp['mad_in_mask']}")

        # --- overlays for review --------------------------------------------------
        edge = outline(mask)
        for name, arr in (("source", src), ("composite", comp_arr)):
            shot = arr[len(arr) // 2].copy()
            shot[edge] = [255, 255, 0]
            from PIL import Image
            Image.fromarray(shot).save(case_dir / "overlays" / f"{cid}_{name}_region.png")
        side = np.concatenate([src[len(src) // 2], comp_arr[len(comp_arr) // 2]], axis=1)
        from PIL import Image
        Image.fromarray(side).save(case_dir / "overlays" / f"{cid}_before_after.png")

        rec = {
            "case_id": cid,
            "source_clip": case["source_clip"],
            "donor_clip": case["donor_clip"],
            "donor_provenance": {
                "clip": case["donor_clip"],
                "dataset_class": 6,
                "class_name": "Closed Panel Cover",
                "official_split": "tr",
                "path": str(don_path.relative_to(REPO)),
                "sha256": mf.sha256_file(don_path),
                "frame_policy": policy,
                "_note": ("Donor pixels come from this training-split clip only. The "
                          "official test split was not read."),
            },
            "source_provenance": {
                "path": str(src_path.relative_to(REPO)),
                "sha256": mf.sha256_file(src_path),
                "dataset_class": 2,
                "class_name": "Opened Panel Cover",
            },
            "preprocess_source": src_rec,
            "preprocess_donor": don_rec,
            "region": {
                "polygon": case["panel_polygon"],
                "mask_px": int(mask.sum()),
                "feather_px": feather_px,
                "blended_px": int(blended.sum()),
                "fraction_of_frame": round(float(mask.mean()), 5),
            },
            "registration": {
                "transform": kind,
                "landmarks": offs,
                "landmarks_used": len(good),
                "fit": {k: v for k, v in fit.items() if not k.startswith("_")},
                "donor_coverage_of_region": round(covered, 5),
            },
            "checks": {
                "unchanged_outside_blended_region": clean,
                "seam_source": seam_src,
                "seam_composite": seam_comp,
                "region_stability_source": stab_src,
                "region_stability_composite": stab_comp,
            },
            "intervention_description": (
                "The panel region of the source was replaced with registered donor pixels "
                "showing the same panel closed. The cover is closed in all "
                f"{len(comp)} frames. No closing motion was generated and no model was "
                "used at this stage."),
            "seconds": round(time.time() - t0, 1),
        }
        (case_dir / "composite.json").write_text(json.dumps(rec, indent=1))
        results[cid] = rec

    manifest_path = out_root / "composite_manifest.json"
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text()).get("cases", {})
        except (ValueError, OSError):
            prior = {}
        for k, v in prior.items():
            results.setdefault(k, v)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "_what": "Mask-restricted closed-panel composites. No generative model involved.",
        "_not": ["not mask-conditioned generation",
                 "not a generated door-closing action",
                 "not evidence that the resulting scene is safe"],
        "plan": args.plan,
        "generated": mf.utc_now(),
        "cases": results,
    }, indent=1))
    log(f"\nwrote {manifest_path}")
    return 0 if all("error" not in r for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
