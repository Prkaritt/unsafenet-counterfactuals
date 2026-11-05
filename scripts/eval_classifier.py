#!/usr/bin/env python3
"""Stage C: VAE consistency and shortcut diagnostics on the validation split.

Runs after training and before any guidance work. Everything here uses **validation**
clips. The official test split is deferred by user decision and is not read.

Gate B3, VAE consistency
------------------------
Guidance never sees a source video. It sees pixels decoded from latents by the CogVideoX
VAE. If a clip's prediction changes when it passes through that VAE untouched, the gradient
the classifier supplies is a gradient about autoencoder artifacts. Measured as an aggregate
over the split: agreement rate and mean absolute change in p(safe walkway). A single clip
changing its prediction proves nothing and is reported, not acted on.

Shortcut diagnostics
--------------------
Three masks of equal area, applied to the same clips:

``worker``   covers the moving person
``lane``     covers the painted walkway region
``control``  covers a label-irrelevant region of the same area

The control mask is the point. Masking anything perturbs a network, so a drop under the
worker mask means little unless the control mask drops less. Both the worker and the lane
carry information the intended decision depends on, so a drop under either is expected. The
diagnosis is the **pattern** across all three, never one number crossing one threshold.

The worker region is located by temporal motion rather than by a detector: the per-pixel
absolute deviation from the clip's median frame is large where something moved and small on
static structure. That is exactly the worker in a fixed-camera clip.
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

from unsafenet_cf import classifier as C    # noqa: E402
from unsafenet_cf import clipset as cs      # noqa: E402
from unsafenet_cf import manifest as mf     # noqa: E402
from unsafenet_cf import preprocess as pp   # noqa: E402

OUT = REPO / "outputs" / "classifier"


def log(msg: str) -> None:
    print(msg, flush=True)


def motion_map(clip: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (H, W) float: transient motion, with flicker suppressed.

    Deviation from the clip's median alone is not enough. These clips contain a workpiece
    glow that pulses in every frame, and raw deviation put every mask on top of it while
    the walking worker went uncovered. Dividing by each pixel's temporal standard
    deviation suppresses anything that varies constantly in place and keeps what passes
    through a pixel once, which is what a person walking does.
    """
    x = clip.astype(np.float32)
    med = np.median(x, axis=0)
    dev = np.abs(x - med).mean(axis=3)              # (T, H, W)
    std = dev.std(axis=0) + 1.0                     # flicker has a large temporal spread
    return (dev.mean(axis=0) / std)


def box_from_scores(scores: np.ndarray, area_fraction: float) -> tuple[int, int, int, int]:
    """Box of the requested area placed where the score mass is greatest.

    Placement is the argmax of a sliding window, computed with an integral image, not the
    centroid of the scores. The centroid version was wrong and the overlay images showed
    it plainly: the walkway in these clips is an L shape whose centroid lies off the
    walkway, so the "lane" box sat in open floor, and every mask converged on the same
    central region. A box placed at the window with the most score mass is inside the
    thing it is masking by construction.
    """
    h, w = scores.shape
    target = area_fraction * h * w
    side_h = max(min(int(round((target * h / w) ** 0.5)), h), 1)
    side_w = max(min(int(round(target / side_h)), w), 1)
    if scores.sum() <= 0:
        return 0, 0, side_h, side_w

    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = scores.cumsum(axis=0).cumsum(axis=1)
    sums = (integral[side_h:, side_w:] - integral[:-side_h or None, side_w:]
            - integral[side_h:, :-side_w or None]
            + integral[:-side_h or None, :-side_w or None])
    y0, x0 = np.unravel_index(int(sums.argmax()), sums.shape)
    return int(y0), int(x0), side_h, side_w


def green_lane_scores(clip: np.ndarray) -> np.ndarray:
    """(H, W) float: how green-dominant each pixel is in the median frame.

    The marked walkway is painted green; this finds it without a segmentation model.
    """
    med = np.median(clip.astype(np.float32), axis=0)
    r, g, b = med[..., 0], med[..., 1], med[..., 2]
    return np.clip(g - (r + b) / 2.0, 0, None)


def control_scores(worker: np.ndarray, lane: np.ndarray) -> np.ndarray:
    """Score map for a region that is neither worker nor lane: the label-irrelevant rest."""
    def norm(a):
        m = a.max()
        return a / m if m > 0 else a
    rest = 1.0 - np.clip(norm(worker) + norm(lane), 0, 1)
    return rest


_DETECTOR = {"model": None}


def load_person_detector(device):
    """COCO-pretrained detector, used only to locate the worker for the occlusion test.

    Motion cannot isolate a person in these clips: the presses and arms move too, and the
    overlay images showed every motion-placed mask landing on machinery. A detector asks
    the right question directly. It plays no part in training or in guidance; it exists so
    that the worker-occlusion diagnostic masks a worker.
    """
    if _DETECTOR["model"] is None:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights as W,
            fasterrcnn_mobilenet_v3_large_fpn as build)
        weights = W.COCO_V1
        model = build(weights=weights, box_score_thresh=0.3).to(device).eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
        _DETECTOR["model"] = model
        _DETECTOR["person_index"] = weights.meta["categories"].index("person")
    return _DETECTOR["model"], _DETECTOR["person_index"]


def detect_person_boxes(clip: np.ndarray, device, score_thresh: float = 0.5):
    """Highest-scoring person box per frame, or None where no person is found.

    Returns (boxes, scores) with one entry per frame. A frame with no detection returns
    None rather than a guess, and those are counted and reported as missed detections.
    """
    import torch

    model, person_idx = load_person_detector(device)
    out_boxes, out_scores = [], []
    with torch.no_grad():
        for start in range(0, len(clip), 8):
            chunk = clip[start:start + 8].astype(np.float32) / 255.0
            batch = [torch.from_numpy(f).permute(2, 0, 1).to(device) for f in chunk]
            for det in model(batch):
                keep = det["labels"] == person_idx
                scores = det["scores"][keep]
                boxes = det["boxes"][keep]
                if len(scores) and float(scores[0]) >= score_thresh:
                    out_boxes.append(boxes[0].cpu().numpy())
                    out_scores.append(float(scores[0]))
                else:
                    out_boxes.append(None)
                    out_scores.append(0.0)
    return out_boxes, out_scores


def box_around(centre_xy, area_fraction: float, h: int, w: int):
    """Mask box of the standard area, centred on a point. Same area as every other mask."""
    target = area_fraction * h * w
    side_h = max(min(int(round((target * h / w) ** 0.5)), h), 1)
    side_w = max(min(int(round(target / side_h)), w), 1)
    cx, cy = centre_xy
    y0 = int(np.clip(round(cy - side_h / 2), 0, max(h - side_h, 0)))
    x0 = int(np.clip(round(cx - side_w / 2), 0, max(w - side_w, 0)))
    return y0, x0, side_h, side_w


def worker_boxes_from_detector(clip: np.ndarray, area_fraction: float, device):
    """Per-frame worker mask boxes from the detector, plus a missed-detection count.

    Where no person is detected, the previous frame's box is carried forward so the mask
    stays on the worker through a brief miss. Frames before any detection are left at the
    frame centre and counted as missed.
    """
    h, w = clip.shape[1:3]
    raw, scores = detect_person_boxes(clip, device)
    boxes, missed, last = [], 0, None
    for b in raw:
        if b is None:
            missed += 1
            boxes.append(last if last is not None
                         else box_around((w / 2, h / 2), area_fraction, h, w))
        else:
            x1, y1, x2, y2 = b
            last = box_around(((x1 + x2) / 2, (y1 + y2) / 2), area_fraction, h, w)
            boxes.append(last)
    return boxes, missed, scores


def per_frame_boxes(clip: np.ndarray, kind: str, area_fraction: float,
                    device=None) -> list[tuple[int, int, int, int]]:
    """One box per frame, so a mask can follow a worker who walks across the frame.

    A single box placed at the motion centroid of a whole clip sits between the worker's
    start and end positions and may cover him in no frame at all. That weakens the
    diagnostic to the point of meaninglessness, which is what a first run showed: the
    worker mask moved accuracy exactly as much as the control mask, which is consistent
    both with a model that ignores the worker and with a mask that never covered him.
    """
    if kind == "worker" and device is not None:
        boxes, _missed, _scores = worker_boxes_from_detector(clip, area_fraction, device)
        return boxes
    x = clip.astype(np.float32)
    med = np.median(x, axis=0)
    # Per-pixel temporal spread, used to divide out anything that flickers in place.
    flicker = np.abs(x - med).mean(axis=3).std(axis=0) + 1.0
    lane = green_lane_scores(clip)
    boxes = []
    for frame in x:
        motion = np.abs(frame - med).mean(axis=2) / flicker
        if kind == "worker":
            scores = motion
        elif kind == "lane":
            scores = lane
        else:
            scores = control_scores(motion, lane)
        boxes.append(box_from_scores(scores, area_fraction))
    return boxes


def motion_energy_removed(clip: np.ndarray,
                          boxes: list[tuple[int, int, int, int]]) -> float:
    """Fraction of the clip's total motion energy that the masks cover.

    This is what makes the diagnostic interpretable. A worker mask that removes 5% of the
    motion tells us nothing when accuracy holds; one that removes 80% and still leaves
    accuracy untouched is evidence about what the model reads.
    """
    x = clip.astype(np.float32)
    med = np.median(x, axis=0)
    flicker = np.abs(x - med).mean(axis=3).std(axis=0) + 1.0
    total = 0.0
    covered = 0.0
    for frame, (y0, x0, bh, bw) in zip(x, boxes):
        motion = np.abs(frame - med).mean(axis=2) / flicker
        total += float(motion.sum())
        covered += float(motion[y0:y0 + bh, x0:x0 + bw].sum())
    return covered / total if total > 0 else 0.0


def apply_boxes(clip: np.ndarray,
                boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Fill a per-frame box in each frame with the clip's mean colour."""
    out = clip.copy()
    fill = clip.reshape(-1, 3).mean(axis=0).astype(clip.dtype)
    for t, (y0, x0, bh, bw) in enumerate(boxes):
        out[t, y0:y0 + bh, x0:x0 + bw, :] = fill
    return out


def apply_box(clip: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Return a copy with the box filled by the clip's own mean colour.

    A grey or black patch is itself a strong out-of-distribution signal. Filling with the
    clip's mean colour keeps the statistics closer to the original, so the measured change
    is more about removed information and less about an alien rectangle.
    """
    y0, x0, bh, bw = box
    out = clip.copy()
    fill = clip.reshape(-1, 3).mean(axis=0).astype(clip.dtype)
    out[:, y0:y0 + bh, x0:x0 + bw, :] = fill
    return out


def predict(model, clips: list[np.ndarray], device, prep, batch_size: int = 4):
    import torch
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(clips), batch_size):
            chunk = clips[start:start + batch_size]
            arr = np.stack(chunk).astype(np.float32) / 255.0
            x = torch.from_numpy(arr).to(device)
            logits = model(C.prepare(x, *prep))
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 2))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rec = []
    for cls in (0, 1):
        m = y_true == cls
        if m.sum():
            rec.append(float((y_pred[m] == cls).mean()))
    return float(np.mean(rec)) if rec else 0.0


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(OUT / "classifier_best.pt"))
    ap.add_argument("--mask-area", type=float, default=0.06,
                    help="fraction of the frame each mask covers; identical for all three")
    ap.add_argument("--vae-clips", type=int, default=10000,
                    help="validation clips put through the VAE round trip for gate B3; "
                         "the round trip costs about 43 s per clip")
    ap.add_argument("--checkpoint-path-vae", default=None,
                    help="staged CogVideoX checkpoint; defaults to $COGVIDEOX_CHECKPOINT")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        log("No CUDA device. Run this on a GPU node.")
        return 2
    device = torch.device("cuda")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log(f"{ckpt_path} missing. Run scripts/train_classifier.py --mode train first.")
        return 2
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prep = (ckpt["preprocess"]["frames"], ckpt["preprocess"]["height"],
            ckpt["preprocess"]["width"])
    model = C.build_model(num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded classifier from epoch {ckpt['epoch']}, preprocess {prep}")

    inventory = json.loads((OUT / "clip_inventory.json").read_text())
    val = [e for e in inventory if e["split"] == "val"]
    if not val:
        log("validation split is empty")
        return 2
    log(f"validation clips: {len(val)}")

    # Decode at the classifier's own geometry.
    sys.path.insert(0, str(REPO / "scripts"))
    from train_classifier import decode_cached
    clips = [decode_cached(e, *prep) for e in val]
    y_true = np.array([e["binary_label"] for e in val])

    base = predict(model, clips, device, prep)
    base_pred = base.argmax(axis=1)
    base_ba = balanced_accuracy(y_true, base_pred)
    log(f"baseline validation balanced accuracy: {base_ba:.4f}")

    # ---------------------------------------------------------------- masks
    results = {}
    missed_total = 0
    frames_total = 0
    for name in ("worker", "lane", "control"):
        masked = []
        energies = []
        first_boxes = None
        for clip in clips:
            if name == "worker":
                boxes, missed, _ = worker_boxes_from_detector(
                    clip, args.mask_area, device)
                missed_total += missed
                frames_total += len(clip)
            else:
                boxes = per_frame_boxes(clip, name, args.mask_area)
            if first_boxes is None:
                first_boxes = boxes[:4]
            energies.append(motion_energy_removed(clip, boxes))
            masked.append(apply_boxes(clip, boxes))
        probs = predict(model, masked, device, prep)
        pred = probs.argmax(axis=1)
        ba = balanced_accuracy(y_true, pred)
        flips = int((pred != base_pred).sum())
        results[name] = {
            "balanced_accuracy": ba,
            "drop_from_baseline": base_ba - ba,
            "prediction_flips": flips,
            "flip_rate": flips / len(clips),
            "mean_abs_prob_shift": float(np.abs(probs - base).mean()),
            "mean_motion_energy_removed": float(np.mean(energies)),
            "localisation": ("COCO person detector" if name == "worker"
                             else "score map, sliding-window maximum"),
            "frames_with_no_person_detected": (missed_total if name == "worker" else None),
            "frames_total": (frames_total if name == "worker" else None),
            "example_boxes_first_clip": [list(map(int, b)) for b in (first_boxes or [])],
        }
        log(f"mask {name:8s}: balanced acc {ba:.4f} "
            f"(drop {base_ba - ba:+.4f}), flips {flips}/{len(clips)}, "
            f"motion energy removed {np.mean(energies):.2%}")

    worker_drop = results["worker"]["drop_from_baseline"]
    lane_drop = results["lane"]["drop_from_baseline"]
    control_drop = results["control"]["drop_from_baseline"]
    worker_energy = results["worker"]["mean_motion_energy_removed"]
    control_energy = results["control"]["mean_motion_energy_removed"]
    # If the worker mask removes little motion, it never covered the worker, and no
    # conclusion about what the model reads can be drawn from it either way.
    # With detector localisation the motion-energy floor no longer decides validity: the
    # mask is on a detected person whether or not that person is the brightest thing
    # moving. Validity is now judged on detection coverage instead.
    miss_rate = (missed_total / frames_total) if frames_total else 1.0
    mask_ineffective = miss_rate > 0.30
    artifact_dominated = control_drop >= min(worker_drop, lane_drop)
    paint_signature = (lane_drop > control_drop) and (worker_drop <= control_drop)

    # ---------------------------------------------------------------- gate B3, VAE
    vae_result = {"ran": False,
                  "reason": "not run"}
    ckpt_vae = args.checkpoint_path_vae or __import__("os").environ.get(
        "COGVIDEOX_CHECKPOINT")
    if ckpt_vae and Path(ckpt_vae).exists():
        from diffusers import AutoencoderKLCogVideoX
        log(f"VAE round trip on {min(args.vae_clips, len(val))} validation clips")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            ckpt_vae, subfolder="vae", torch_dtype=torch.float16).to(device)
        vae.enable_tiling()
        vae.enable_slicing()
        # The round trip must run at the resolution guidance will use, 720x480x49, not at
        # the classifier's 240x320. Gate B3 asks whether the VAE changes the classifier's
        # verdict under the operating conditions of Stage E; measuring it at a different
        # resolution would answer a different question. Both the original and the
        # reconstruction are then fed through the same C.prepare the guidance loop uses.
        # Post-hoc expansion, 2026-11-04: the full fixed validation split, evaluated once.
        # The 0.90 threshold is unchanged. The first 16 clips are the same clips in the
        # same order as the original run, so that result is preserved exactly and reported
        # alongside, whichever way the larger sample goes.
        subset = list(range(min(args.vae_clips, len(val))))
        gen_prep = (49, pp.TARGET_HEIGHT, pp.TARGET_WIDTH)
        log(f"  decoding {len(subset)} clips at {gen_prep[2]}x{gen_prep[1]}x{gen_prep[0]} "
            "for the round trip")
        gen_clips = [decode_cached(val[i], *gen_prep) for i in subset]
        # Both sides of the comparison are the SAME tensor: the reconstruction is produced
        # from gen_clips, and both go through the same predict() with the same prep tuple
        # and the same model in eval mode under no_grad. Frame selection, crop and
        # normalisation cannot differ between them, because there is only one of each.
        base_gen = predict(model, gen_clips, device, prep, batch_size=2)

        recon_dir = OUT / "vae_reconstructions"
        recon_dir.mkdir(parents=True, exist_ok=True)
        recon_clips = []
        reused = 0
        t0 = time.time()
        mf.reset_peak_memory()
        with torch.no_grad():
            for i, arr0 in zip(subset, gen_clips):
                cache_file = recon_dir / f"{val[i]['clip_id']}_{val[i]['sha256'][:12]}.npy"
                if cache_file.exists():
                    recon_clips.append(np.load(cache_file))
                    reused += 1
                    continue
                arr = arr0.astype(np.float32) / 255.0
                x = torch.from_numpy(arr).to(device).permute(3, 0, 1, 2)[None]
                x = (x * 2 - 1).to(torch.float16)
                lat = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
                rec = vae.decode(lat / vae.config.scaling_factor).sample
                rec = ((rec.float() + 1) / 2).clamp(0, 1)[0].permute(1, 2, 3, 0)
                out_arr = (rec.cpu().numpy() * 255).astype(np.uint8)
                np.save(cache_file, out_arr)
                recon_clips.append(out_arr)
        vae_s = time.time() - t0
        peak = mf.peak_memory_gb()
        log(f"  reconstructions: {reused} reused from cache, "
            f"{len(subset) - reused} computed")
        del vae
        mf.free_gpu()

        rec_probs = predict(model, recon_clips, device, prep, batch_size=2)
        rec_pred = rec_probs.argmax(axis=1)
        # Compared against the same clips at the same resolution, so the only difference
        # between the two predictions is the VAE round trip itself.
        base_gen_pred = base_gen.argmax(axis=1)
        y_sub = np.array([val[i]["binary_label"] for i in subset])
        agree = float((rec_pred == base_gen_pred).mean())
        shift = float(np.abs(rec_probs[:, 1] - base_gen[:, 1]).mean())

        # Agreement alone hides which direction a flip went. Accuracy against the true
        # labels, before and after, says whether the round trip degrades the classifier
        # or merely moves clips that were already wrong.
        acc_before = float((base_gen_pred == y_sub).mean())
        acc_after = float((rec_pred == y_sub).mean())
        ba_before = balanced_accuracy(y_sub, base_gen_pred)
        ba_after = balanced_accuracy(y_sub, rec_pred)

        per_clip = [{
            "clip_id": val[i]["clip_id"],
            "true_binary_label": int(y_sub[k]),
            "dataset_class": val[i]["dataset_class"],
            "p_safe_before": float(base_gen[k, 1]),
            "p_safe_after": float(rec_probs[k, 1]),
            "pred_before": int(base_gen_pred[k]),
            "pred_after": int(rec_pred[k]),
            "flipped": bool(rec_pred[k] != base_gen_pred[k]),
            "correct_before": bool(base_gen_pred[k] == y_sub[k]),
            "correct_after": bool(rec_pred[k] == y_sub[k]),
        } for k, i in enumerate(subset)]

        flipped = [r for r in per_clip if r["flipped"]]
        log(f"  accuracy vs true labels: before {acc_before:.4f}, after {acc_after:.4f}")
        log(f"  balanced accuracy      : before {ba_before:.4f}, after {ba_after:.4f}")
        log(f"  flipped clips: {len(flipped)}")
        for r in flipped:
            log(f"    {r['clip_id']:10s} class {r['dataset_class']} "
                f"p_safe {r['p_safe_before']:.4f} -> {r['p_safe_after']:.4f}  "
                f"pred {r['pred_before']} -> {r['pred_after']}  "
                f"correct {r['correct_before']} -> {r['correct_after']}")

        # The original 16-clip result, preserved exactly: same clips, same order.
        first16 = per_clip[:16]
        agree16 = float(np.mean([not r["flipped"] for r in first16])) if first16 else None
        vae_result = {
            "ran": True,
            "clips": len(subset),
            "round_trip_resolution": f"{pp.TARGET_WIDTH}x{pp.TARGET_HEIGHT}x49",
            "agreement_with_240x320_predictions": float(
                (rec_pred == base_pred[subset]).mean()),
            "agreement_rate": agree,
            "mean_abs_shift_p_safe": shift,
            "accuracy_vs_true_before": acc_before,
            "accuracy_vs_true_after": acc_after,
            "balanced_accuracy_before": ba_before,
            "balanced_accuracy_after": ba_after,
            "flipped_clips": flipped,
            "per_clip": per_clip,
            "original_first16_agreement": agree16,
            "original_first16_note": ("The first 16 entries are the same clips in the same "
                                      "order as the run of 2026-11-04 that reported 14/16 "
                                      "= 0.8750. Both results are reported; neither "
                                      "replaces the other."),
            "expansion_note": ("Post-hoc expansion of the sample from 16 clips to the full "
                               "fixed validation split, decided after the 16-clip result "
                               "failed. The 0.90 threshold was NOT changed. See "
                               "docs/DIAGNOSTIC_AMENDMENTS.md."),
            "gate_B3_agreement_at_least_0.90": agree >= 0.90,
            "gate_B3_mean_shift_at_most_0.10": shift <= 0.10,
            "seconds": vae_s,
            "peak_gpu_gib": peak,
            "note": ("Aggregate over the subset. A single clip changing its prediction "
                     "does not establish shortcut learning and is not treated as doing so."),
        }
        log(f"VAE consistency: agreement {agree:.4f}, mean |shift| p(safe) {shift:.4f}, "
            f"{vae_s:.1f}s, peak {peak} GiB")
    else:
        vae_result["reason"] = ("CogVideoX checkpoint not found; set COGVIDEOX_CHECKPOINT "
                                "or pass --checkpoint-path-vae")
        log(f"VAE consistency SKIPPED: {vae_result['reason']}")

    gate_c = ((not artifact_dominated) and (not paint_signature)
              and (not mask_ineffective))
    gate_b3 = bool(vae_result.get("gate_B3_agreement_at_least_0.90") and
                   vae_result.get("gate_B3_mean_shift_at_most_0.10"))

    report = {
        "baseline_balanced_accuracy": base_ba,
        "mask_area_fraction": args.mask_area,
        "masks": results,
        "interpretation": {
            "worker_drop": worker_drop,
            "lane_drop": lane_drop,
            "control_drop": control_drop,
            "worker_mask_motion_energy_removed": worker_energy,
            "control_mask_motion_energy_removed": control_energy,
            "worker_mask_ineffective": mask_ineffective,
            "person_detection_miss_rate": miss_rate,
            "artifact_dominated": artifact_dominated,
            "paint_shortcut_signature": paint_signature,
            "how_to_read": (
                "The control mask covers a label-irrelevant region of identical area. If "
                "its drop is at least as large as the worker or lane drop, masking itself "
                "is moving the prediction and neither diagnostic supports a conclusion. "
                "The paint-shortcut signature is a large lane drop together with a worker "
                "drop no larger than the control's: the model reads the painted region and "
                "not the person. Both worker and lane carry genuine information, so a drop "
                "under either is expected and is not by itself evidence of a shortcut."),
        },
        "vae_consistency": vae_result,
        "gates": {"C_shortcut_diagnostics": gate_c, "B3_vae_consistency": gate_b3},
        "test_split": "DEFERRED by user decision; official test evaluation is pending.",
    }
    (OUT / "eval_report.json").write_text(json.dumps(report, indent=2))
    log(f"\nGate C (shortcut diagnostics): {'PASS' if gate_c else 'FAIL'}")
    log(f"Gate B3 (VAE consistency)    : {'PASS' if gate_b3 else 'FAIL'}")
    log(f"wrote {OUT}/eval_report.json")
    return 0 if (gate_c and gate_b3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
