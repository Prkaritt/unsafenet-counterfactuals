#!/usr/bin/env python
"""Diagnostics for the class-2/class-6 classifier: VAE sensitivity and region masking.

Two separate questions, deliberately answered in one job so the model loads once.

**1. Does the classifier survive the latent round trip?** Every generated arm in this
experiment has passed through the CogVideoX VAE. If the classifier's prediction changes on a
clip that was only encoded and decoded, with no edit at all, then no later prediction change
can be attributed to an edit. This is the control that the walkway stage already failed, and
it is measured again here for this classifier rather than assumed to behave the same way.

**2. Does the classifier look at the panel?** The panel region of each clip is overwritten
with flat grey, which destroys the panel's state while leaving the rest of the scene intact,
and the prediction change is recorded. The same is done with **control regions** of
identical area placed elsewhere in the frame. If masking the panel moves predictions far
more than masking a control region of the same size, that is evidence the decision depends
on the panel region.

It is evidence and not proof, and the reason is worth stating: a region correlated with the
panel, such as the machine it is attached to, would produce the same signature. The
diagnostic bounds where the model is looking; it does not establish what it is looking at.
Both classes are tested, because a model that only reacts when the panel is open is using a
different cue from one that reacts in both directions.
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

OUT = REPO / "outputs" / "panel_cover"
BINARY_NAMES = {0: "opened_panel_cover", 1: "closed_panel_cover"}


def log(m: str) -> None:
    print(m, flush=True)


def decode(path: Path, frames: int, height: int, width: int) -> np.ndarray:
    got = [f.copy() for f in cs.iter_frames(path, width=width, height=height)]
    arr = np.stack(got)
    idx = np.linspace(0, len(arr) - 1, frames).round().astype(int)
    return arr[idx]


def box_mask(box, shape) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    m = np.zeros(shape, bool)
    m[max(0, y):y + h, max(0, x):x + w] = True
    return m


def scale_box(box, src_wh, dst_wh):
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    x, y, w, h = box
    return [int(round(x * sx)), int(round(y * sy)),
            max(1, int(round(w * sx))), max(1, int(round(h * sy)))]


def control_boxes(box, width: int, height: int) -> dict[str, list[int]]:
    """Equal-area control regions elsewhere in the frame.

    Three of them, because one control can land somewhere uninformative by luck: the
    horizontal mirror of the panel box, the vertical mirror, and the frame centre.
    """
    x, y, w, h = [int(v) for v in box]
    return {
        "mirror_x": [max(0, min(width - w, width - x - w)), y, w, h],
        "mirror_y": [x, max(0, min(height - h, height - y - h)), w, h],
        "centre": [max(0, (width - w) // 2), max(0, (height - h) // 2), w, h],
    }


def main() -> int:
    import torch
    from unsafenet_cf import classifier as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="outputs/panel_cover/classifier/"
                                        "training_report.json")
    ap.add_argument("--checkpoint", default="outputs/panel_cover/classifier/"
                                            "panel_classifier.pt")
    ap.add_argument("--regions", default="configs/panel_cover_regions.json",
                    help="the frozen panel box, in source 1920x1080 coordinates")
    ap.add_argument("--audit", default="outputs/panel_cover/audit/audit.json")
    ap.add_argument("--selection", default="outputs/panel_cover/selection.json",
                    help="supplies each clip's measured alignment to the box's camera")
    ap.add_argument("--vae-subset", type=int, default=8,
                    help="clips sent through the VAE round trip; each costs about 43 s")
    ap.add_argument("--out", default="outputs/panel_cover/diagnostics.json")
    ap.add_argument("--checkpoint-dir", default=None)
    args = ap.parse_args()

    for p in (args.report, args.checkpoint, args.audit, args.regions, args.selection):
        if not (REPO / p).exists():
            log(f"missing required input: {p}")
            return 2
    report = json.loads((REPO / args.report).read_text())
    audit = json.loads((REPO / args.audit).read_text())
    regions = json.loads((REPO / args.regions).read_text())
    selection = json.loads((REPO / args.selection).read_text())

    blob = torch.load(REPO / args.checkpoint, map_location="cpu", weights_only=False)
    pre = blob.get("preprocess", {})
    frames = pre.get("frames", C.DEFAULT_FRAMES)
    height = pre.get("height", C.DEFAULT_HEIGHT)
    width = pre.get("width", C.DEFAULT_WIDTH)
    model = C.build_model(num_classes=2, pretrained=False)
    model.load_state_dict(blob["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    log(f"classifier from epoch {blob.get('epoch')}, preprocessing "
        f"{frames}x{height}x{width}")
    log(f"label map: {blob.get('label_map', {}).get('dataset_class_to_binary_index')}")

    val_ids = report["split"]["val_clips"]
    clips = {cid: audit["clips"][cid] for cid in val_ids if cid in audit["clips"]}
    log(f"validation clips available: {len(clips)}")

    def score(arr: np.ndarray) -> tuple[int, float]:
        with torch.no_grad():
            x = torch.from_numpy(arr).unsqueeze(0).to(device).float() / 255.0
            p = torch.softmax(model(C.prepare(x, frames, height, width)).float(), -1)[0]
        return int(p.argmax()), float(p[1])

    # --- region masking ----------------------------------------------------------
    # The box is frozen once, in source 1920x1080 coordinates, and the classifier sees the
    # whole frame scaled to width x height with no crop, so the box scales by the same
    # factors. An earlier version of this looked the box up by the audit's camera-view
    # label; that lookup never matched, and the view clustering it relied on was already
    # rejected in DECISIONS P5. A clip is now included on its measured alignment to the
    # camera the box belongs to, which is the test that actually governs the box's meaning.
    view_cfg = regions["views"]["press_front"]
    panel_box_full = view_cfg["panel_box_full_res"]
    min_align = float(regions.get("alignment_min_ncc", 0.95))
    align = {c["clip_id"]: c.get("alignment_ncc") for c in selection["clips"]
             if "error" not in c}
    panel_box = scale_box(panel_box_full, (1920, 1080), (width, height))
    masking = {"per_clip": [], "_regions_config": args.regions,
               "_panel_box_full_res": panel_box_full,
               "_panel_box_classifier_frame": panel_box,
               "_panel_box_area_fraction": round(panel_box[2] * panel_box[3]
                                                 / float(width * height), 5),
               "_alignment_min_ncc": min_align}
    log("\nregion masking, panel against three equal-area controls")
    log(f"  panel box {panel_box_full} full res -> {panel_box} in the classifier's "
        f"{height}x{width} frame, "
        f"{panel_box[2] * panel_box[3] / float(width * height) * 100:.2f}% of it")
    for cid, rec in sorted(clips.items()):
        a = align.get(cid)
        if a is None or a < min_align:
            masking["per_clip"].append({
                "clip_id": cid, "alignment_ncc": a,
                "skipped": f"alignment {a} below {min_align}; the box is in one camera's "
                           "coordinates and means nothing for a clip shot elsewhere"})
            continue
        arr = decode(REPO / rec["path"], frames, height, width)
        base_pred, base_p = score(arr)
        boxes = {"panel": panel_box} | control_boxes(panel_box, width, height)
        row = {"clip_id": cid, "dataset_class": rec["dataset_class"],
               "alignment_ncc": a,
               "true_binary": {2: 0, 6: 1}[rec["dataset_class"]],
               "baseline": {"pred": base_pred, "prob_closed": round(base_p, 5)},
               "masked": {}}
        for name, box in boxes.items():
            m = box_mask(box, (height, width))
            masked = arr.copy()
            masked[:, m] = 128
            pred, p = score(masked)
            row["masked"][name] = {
                "box_xywh": box,
                "area_px": int(m.sum()),
                "pred": pred,
                "prob_closed": round(p, 5),
                "delta_prob_closed": round(p - base_p, 5),
                "prediction_changed": pred != base_pred,
            }
        masking["per_clip"].append(row)
        d = row["masked"]
        log(f"  {cid:10s} class {rec['dataset_class']} base P(closed) {base_p:.4f} "
            f"panel {d['panel']['delta_prob_closed']:+.4f} "
            f"mirror_x {d['mirror_x']['delta_prob_closed']:+.4f} "
            f"mirror_y {d['mirror_y']['delta_prob_closed']:+.4f} "
            f"centre {d['centre']['delta_prob_closed']:+.4f}")

    # Summary by class, so a one-sided effect cannot hide in the mean.
    summary = {}
    for cls in (2, 6):
        rows = [r for r in masking["per_clip"]
                if r.get("dataset_class") == cls and "masked" in r]
        if not rows:
            continue
        summary[f"class_{cls}"] = {
            "n": len(rows),
            "mean_abs_delta_prob_closed": {
                name: round(float(np.mean([abs(r["masked"][name]["delta_prob_closed"])
                                           for r in rows])), 5)
                for name in ("panel", "mirror_x", "mirror_y", "centre")},
            "predictions_changed": {
                name: int(sum(r["masked"][name]["prediction_changed"] for r in rows))
                for name in ("panel", "mirror_x", "mirror_y", "centre")},
        }
    masking["summary_by_class"] = summary
    masking["_interpretation"] = (
        "Compare the panel column against the three control columns within each class. A "
        "panel effect much larger than every control is evidence that the decision depends "
        "on the panel region. It is NOT proof of causal dependence on panel state: any "
        "region correlated with the panel, such as the machine it sits on, produces the "
        "same signature. Masking also moves an input off the training distribution, which "
        "changes predictions on its own; that is exactly why controls of identical area "
        "are measured rather than a single masked condition.")

    # --- VAE round trip ----------------------------------------------------------
    log("\nVAE round trip on a balanced subset")
    vae_rows = []
    try:
        from diffusers import AutoencoderKLCogVideoX
        ckpt = args.checkpoint_dir or \
            "/ocean/projects/cis260288p/puprety/checkpoints/CogVideoX-2b"
        vae = AutoencoderKLCogVideoX.from_pretrained(
            ckpt, subfolder="vae", torch_dtype=torch.float16).to(device)
        vae.eval()
        vae.enable_tiling()
        vae.enable_slicing()
        from unsafenet_cf import preprocess as prep
        per_class = max(1, args.vae_subset // 2)
        chosen = []
        for cls in (2, 6):
            ids = [c for c in sorted(clips) if clips[c]["dataset_class"] == cls]
            chosen.extend(ids[:per_class])
        log(f"  subset: {chosen}")
        for cid in chosen:
            rec = clips[cid]
            # The round trip needs the VAE's own geometry, not the classifier's.
            full = decode(REPO / rec["path"], prep.SAMPLE_FRAMES,
                          prep.TARGET_HEIGHT, prep.TARGET_WIDTH)
            t = time.time()
            with torch.no_grad():
                x = torch.from_numpy(full).to(device).float() / 127.5 - 1.0
                x = x.permute(3, 0, 1, 2).unsqueeze(0).half()
                lat = vae.encode(x).latent_dist.mode()
                scaled = lat * vae.config.scaling_factor
                dec = vae.decode(scaled / vae.config.scaling_factor).sample
            secs = time.time() - t
            rec_arr = np.clip((dec[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1.0)
                              * 127.5, 0, 255).astype(np.uint8)
            from PIL import Image
            idx = np.linspace(0, len(full) - 1, frames).round().astype(int)
            small_orig = np.stack([np.asarray(Image.fromarray(full[i])
                                              .resize((width, height))) for i in idx])
            small_rec = np.stack([np.asarray(Image.fromarray(rec_arr[i])
                                             .resize((width, height))) for i in idx])
            p0, q0 = score(small_orig)
            p1, q1 = score(small_rec)
            vae_rows.append({
                "clip_id": cid, "dataset_class": rec["dataset_class"],
                "true_binary": {2: 0, 6: 1}[rec["dataset_class"]],
                "original": {"pred": p0, "prob_closed": round(q0, 5)},
                "reconstructed": {"pred": p1, "prob_closed": round(q1, 5)},
                "prediction_flipped": p0 != p1,
                "delta_prob_closed": round(q1 - q0, 5),
                "roundtrip_seconds": round(secs, 1),
            })
            log(f"  {cid:10s} class {rec['dataset_class']}: pred {p0} -> {p1}, "
                f"P(closed) {q0:.4f} -> {q1:.4f} ({secs:.1f}s)")
        del vae
        mf.free_gpu()
    except Exception as exc:
        vae_rows = [{"error": f"{type(exc).__name__}: {exc}"}]
        log(f"  VAE round trip failed: {exc}")

    flips = [r for r in vae_rows if r.get("prediction_flipped")]
    out = {
        "_what": "VAE sensitivity and region-masking diagnostics for the panel classifier.",
        "generated": mf.utc_now(),
        "classifier": {
            "checkpoint": args.checkpoint,
            "epoch": blob.get("epoch"),
            "label_map": blob.get("label_map"),
            "validation_metrics": blob.get("val_metrics"),
            "preprocess": {"frames": frames, "height": height, "width": width,
                           "function": "unsafenet_cf.classifier.prepare"},
        },
        "vae_round_trip": {
            "clips": vae_rows,
            "n_flipped": len(flips),
            "_control_meaning": ("A flip here means the latent round trip alone changes "
                                 "the decision. Every generated arm passes through this "
                                 "round trip, so any flip measured on an edited clip is "
                                 "uninterpretable while this number is non-zero."),
        },
        "region_masking": masking,
        "environment": mf.environment_snapshot(),
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    log(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
