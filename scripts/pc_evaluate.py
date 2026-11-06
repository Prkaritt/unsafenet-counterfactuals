#!/usr/bin/env python
"""Measure every arm of every panel-cover case against the questions that matter.

The questions, and how each one is turned into a number:

**Is the same physical panel visibly closed?** There is no panel-state detector, so the
measurement is a comparison, not a classification. For each arm the panel region is
correlated against two references taken from the case's own footage: the source's open
panel, and the registered donor's closed panel. An arm whose region matches the closed
reference better than the open one reads as closed. This is a **proxy** and is labelled one
everywhere it appears; the authoritative answer comes from a human looking at the overlays.

**Was the door removed, or covered with a flat patch?** A flat patch has almost no gradient
energy. Mean absolute gradient inside the region is reported for every arm next to the
source and donor values, so a region that was smoothed into a blank rectangle shows up as a
number far below both.

**Is everything else preserved?** PSNR and mean absolute difference against the source are
computed **outside** the panel region only, where any change is by definition unwanted. The
whole-frame figure is also given, and is the weaker number of the two.

**Are the seams plausible?** The step across the region boundary, against a control boundary
of the same shape shifted into untouched scene.

**Temporal behaviour.** Mean absolute deviation from each arm's own temporal median inside
the region: a closed panel should sit still, and flicker is visible here as a number.

**Classifier scores, if a classifier exists.** Reported last and kept apart from everything
above, because a prediction flip is not evidence about the panel. The VAE-only arm is the
control: if the round trip alone flips the label, no later flip can be attributed to the
edit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsafenet_cf import compare as cmp       # noqa: E402
from unsafenet_cf import manifest as mf       # noqa: E402
from unsafenet_cf import manualpath as mp     # noqa: E402
from unsafenet_cf import panelpatch as pp     # noqa: E402

OUT = REPO / "outputs" / "panel_cover"
ARM_FILES = ("source", "donor", "composite", "vae_source", "vae_composite",
             "base_edit", "refined", "refined_restricted",
             "unguided_control", "guided")


def log(m: str) -> None:
    print(m, flush=True)


def decode(path: Path) -> np.ndarray:
    from diffusers.utils import load_video
    return cmp.to_array(load_video(str(path)))


def grad_energy(frame: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute spatial gradient inside the mask, in grey levels per pixel."""
    g = frame.astype(np.float64)
    g = g.mean(axis=2) if g.ndim == 3 else g
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1]))
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    m = mask.astype(bool)
    return round(float(((gx + gy) / 2.0)[m].mean()), 4) if m.any() else 0.0


def region_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    x = x.mean(axis=2) if x.ndim == 3 else x
    y = y.mean(axis=2) if y.ndim == 3 else y
    return round(pp.ncc(x[m], y[m]), 5)


def outside_metrics(src: np.ndarray, arm: np.ndarray, mask: np.ndarray) -> dict:
    m = ~mask.astype(bool)
    a = src[:, m, :].astype(np.float64)
    b = arm[:, m, :].astype(np.float64)
    mad = float(np.abs(a - b).mean())
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    return {"psnr_outside_region_db": round(psnr, 3) if np.isfinite(psnr) else None,
            "mean_abs_diff_outside_region": round(mad, 4),
            "psnr_whole_frame_db": cmp.psnr(src, arm),
            "mean_abs_diff_whole_frame": cmp.mean_abs_diff(src, arm)}


def classifier_scores(clips: dict[str, np.ndarray], ckpt_path: Path) -> dict:
    import torch
    from unsafenet_cf import classifier as C
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pre = blob.get("preprocess", {})
    frames = pre.get("frames", C.DEFAULT_FRAMES)
    height = pre.get("height", C.DEFAULT_HEIGHT)
    width = pre.get("width", C.DEFAULT_WIDTH)
    model = C.build_model(num_classes=2, pretrained=False)
    model.load_state_dict(blob["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    out = {"_label_map": blob.get("label_map"),
           "_validation_at_checkpoint": blob.get("val_metrics"),
           "_warning": ("A prediction flip is not evidence that a panel closed. Read the "
                        "vae_source row first: if the round trip alone flips the label, "
                        "every later flip is uninterpretable."),
           "arms": {}}
    with torch.no_grad():
        for name, arr in clips.items():
            x = torch.from_numpy(arr).unsqueeze(0).to(device).float() / 255.0
            p = torch.softmax(model(C.prepare(x, frames, height, width)).float(), -1)[0]
            out["arms"][name] = {
                "prob_opened_panel_cover": round(float(p[0]), 5),
                "prob_closed_panel_cover": round(float(p[1]), 5),
                "predicted_binary_index": int(p.argmax()),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/panel_cover_plan.json")
    ap.add_argument("--cases", default="")
    ap.add_argument("--classifier", default="outputs/panel_cover/classifier/"
                                            "panel_classifier.pt")
    ap.add_argument("--out", default="outputs/panel_cover/evaluation.json")
    args = ap.parse_args()

    plan = json.loads((REPO / args.plan).read_text())
    only = {c for c in args.cases.split(",") if c}
    results: dict[str, dict] = {}

    for case in plan["cases"]:
        cid = case["case_id"]
        if only and cid not in only:
            continue
        case_dir = OUT / cid
        comp_rec_path = case_dir / "composite.json"
        if not comp_rec_path.exists():
            log(f"{cid}: no composite.json; skipped")
            results[cid] = {"error": "composite.json missing"}
            continue
        comp_rec = json.loads(comp_rec_path.read_text())
        poly = np.asarray(comp_rec["region"]["polygon"], dtype=float)

        arms: dict[str, np.ndarray] = {}
        for name in ARM_FILES:
            # The guidance arms are written to <case>/guided/, every other arm to
            # <case>/videos/. Both are scored here on identical terms: a guided output
            # that flips the classifier still has to meet the same panel-state proxy.
            p = case_dir / "videos" / f"{cid}_{name}.mp4"
            if not p.exists():
                p = case_dir / "guided" / f"{cid}_{name}.mp4"
            if p.exists():
                arms[name] = decode(p)
        if "source" not in arms:
            results[cid] = {"error": "source video missing"}
            continue
        src = arms["source"]
        mask = mp.polygon_mask(poly, src.shape[1:3])
        log(f"\n=== {cid}: arms {sorted(arms)}, region {int(mask.sum())} px")

        # Closed reference: the composite's own region, which is registered donor pixels.
        closed_ref = arms.get("composite", src)[0]
        open_ref = src[0]
        mid = len(src) // 2

        rows = {}
        for name, arr in arms.items():
            if name == "donor":
                continue
            n = min(len(arr), len(src))
            rows[name] = {
                "frames": int(len(arr)),
                "panel_state_proxy": {
                    "ncc_vs_open_reference": region_ncc(arr[mid], open_ref, mask),
                    "ncc_vs_closed_reference": region_ncc(arr[mid], closed_ref, mask),
                    "_reference_note": ("Open reference is the source's own first frame. "
                                        "Closed reference is the composite's first frame, "
                                        "whose region holds registered donor pixels."),
                    "_proxy_warning": ("Correlation of a region, not a panel-state "
                                       "detector. Higher correlation with the closed "
                                       "reference is evidence, not proof, and the human "
                                       "review form decides."),
                },
                "gradient_energy_in_region": grad_energy(arr[mid], mask),
                "preservation": outside_metrics(src[:n], arr[:n], mask),
                "seam": pp.seam_contrast(arr[mid], mask),
                "region_stability": pp.region_stability(arr, mask),
            }
            r = rows[name]["panel_state_proxy"]
            log(f"  {name:20s} ncc open {r['ncc_vs_open_reference']:+.4f} "
                f"closed {r['ncc_vs_closed_reference']:+.4f}  "
                f"grad {rows[name]['gradient_energy_in_region']:7.3f}  "
                f"outside PSNR {rows[name]['preservation']['psnr_outside_region_db']}")

        rows_src_region = {
            "source_region_gradient": grad_energy(src[mid], mask),
            "donor_region_gradient": (grad_energy(arms["donor"][0], mask)
                                      if "donor" in arms else None),
            "_note": ("Donor gradient is measured in the SOURCE's region coordinates, so "
                      "it is only comparable where registration placed donor content "
                      "there. It is a scale for the composite's value, not a target."),
        }

        entry = {
            "case_id": cid,
            "source_clip": case["source_clip"],
            "donor_clip": case["donor_clip"],
            "region_px": int(mask.sum()),
            "region_fraction_of_frame": round(float(mask.mean()), 5),
            "arms": rows,
            "region_texture_references": rows_src_region,
            "registration_residual_px": comp_rec.get("registration", {})
            .get("fit", {}).get("residual_px"),
        }
        ckpt = REPO / args.classifier
        if ckpt.exists():
            try:
                entry["classifier"] = classifier_scores(
                    {k: v for k, v in arms.items() if k != "donor"}, ckpt)
                for name, s in entry["classifier"]["arms"].items():
                    log(f"  {name:20s} P(closed) {s['prob_closed_panel_cover']:.4f}")
            except Exception as exc:
                entry["classifier"] = {"error": f"{type(exc).__name__}: {exc}"}
                log(f"  classifier scoring failed: {exc}")
        else:
            entry["classifier"] = {"skipped": f"no checkpoint at {args.classifier}"}
        results[cid] = entry

    report = {
        "_what": "Measurements of every panel-cover arm against the source and the donor.",
        "_judgement_basis": ("Every number here is computed from decoded frames. None of "
                             "it replaces playback review, and the panel-state figure is "
                             "a correlation proxy rather than a detector."),
        "generated": mf.utc_now(),
        "cases": results,
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1))
    log(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
