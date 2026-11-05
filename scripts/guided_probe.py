#!/usr/bin/env python3
"""Stage D: one guided-step probe. Measures before anything is generated.

Checks, each one a gate from docs/ACCEPTANCE_CRITERIA.md:

D1  the gradient reaching the latent is finite
D2  it is non-zero, and the frames receiving it change with the sampling offset
D3  one guidance step moves the classifier the intended way
D4  peak GPU memory is measured, not assumed

Nothing is generated. The VAE decoder and the classifier are the only modules in the
backward graph; the transformer is never loaded here, because guidance on the predicted
clean latent does not need it. That keeps this probe an honest lower bound on the memory a
guided generation needs, and it is reported as a lower bound rather than as the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsafenet_cf import classifier as C    # noqa: E402
from unsafenet_cf import guidance as G      # noqa: E402
from unsafenet_cf import manifest as mf     # noqa: E402
from unsafenet_cf import preprocess as pp   # noqa: E402

OUT = REPO / "outputs" / "guided"


def log(msg: str) -> None:
    print(msg, flush=True)


def main(argv=None) -> int:
    import torch
    from diffusers import AutoencoderKLCogVideoX
    from diffusers.utils import load_video

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", default=str(REPO / "outputs" / "classifier" /
                                                "classifier_best.pt"))
    ap.add_argument("--checkpoint", default=os.environ.get("COGVIDEOX_CHECKPOINT"))
    ap.add_argument("--target-index", type=int, default=1,
                    help="binary index to guide toward; 1 is safe_walkway")
    ap.add_argument("--step-size", type=float, default=50.0,
                    help="latent units per unit gradient for the D3 direction check")
    ap.add_argument("--latent-window", type=int, default=G.DEFAULT_LATENT_WINDOW,
                    help="latent frames decoded per guided step; a full decode does not "
                         "fit in memory with autograd retained")
    ap.add_argument("--vae-tiling", action="store_true", default=True)
    ap.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        log("No CUDA device. Run this on a GPU node.")
        return 2
    if not args.checkpoint or not Path(args.checkpoint).exists():
        log("CogVideoX checkpoint not found; set COGVIDEOX_CHECKPOINT.")
        return 2
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    # --- classifier, frozen -------------------------------------------------------
    ck = torch.load(args.classifier, map_location="cpu", weights_only=False)
    prep = (ck["preprocess"]["frames"], ck["preprocess"]["height"],
            ck["preprocess"]["width"])
    label_map = ck.get("label_map", {})
    model = C.build_model(num_classes=2, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    # Frozen weights. Note that this does NOT stop gradients reaching the input, which is
    # the whole point: requires_grad on parameters and graph connectivity are different
    # things, and gate D2 is what distinguishes them.
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"classifier loaded, epoch {ck['epoch']}, preprocess {prep}")
    log(f"label map: {label_map.get('binary_index_to_name')}")

    # --- VAE ----------------------------------------------------------------------
    t0 = time.time()
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.checkpoint, subfolder="vae", torch_dtype=torch.float16).to(device)
    if args.vae_tiling:
        vae.enable_tiling()
        vae.enable_slicing()
    # Recompute decoder activations in the backward pass instead of storing them. The
    # decoder is the memory bottleneck here, and this is the difference between fitting
    # and not fitting.
    vae.enable_gradient_checkpointing()
    for p in vae.parameters():
        p.requires_grad_(False)
    log(f"VAE loaded in {time.time() - t0:.1f}s, tiling={args.vae_tiling}")

    # --- a real latent from the real source clip ----------------------------------
    pre_path = REPO / "outputs" / "preprocessed" / "0_tr143_720x480x49.mp4"
    frames = load_video(str(pre_path))
    arr = np.stack([np.asarray(f) for f in frames]).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).to(device).permute(3, 0, 1, 2)[None]
    x = (x * 2 - 1).to(torch.float16)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
    log(f"latent from source clip: {tuple(latent.shape)}")

    # --- D1, D2 --------------------------------------------------------------------
    mf.reset_peak_memory()
    t0 = time.time()
    grad0, score0, scale0, win0 = G.latent_gradient(
        vae, model, latent, args.target_index, prep, offset=0, window=args.latent_window)
    step_s = time.time() - t0
    peak = mf.peak_memory_gb()
    diag0 = G.guidance_diagnostics(grad0)
    log(f"gradient: finite={diag0['finite']} norm={diag0['norm']:.6g} "
        f"nonzero_fraction={diag0['fraction_nonzero']:.4f}")
    log(f"one guided step: {step_s:.2f}s, peak GPU {peak} GiB")

    # Frame coverage must change with the offset, or guidance only ever edits 16 frames.
    grad1, _, _, win1 = G.latent_gradient(
        vae, model, latent, args.target_index, prep,
        offset=max(prep[0] // 2, 1), window=args.latent_window)
    cov0 = (grad0.abs().sum(dim=(0, 1, 3, 4)) > 0)
    cov1 = (grad1.abs().sum(dim=(0, 1, 3, 4)) > 0)
    coverage_changes = bool((cov0 != cov1).any())
    log(f"latent frames receiving gradient: offset 0 -> {int(cov0.sum())}, "
        f"offset {max(prep[0] // 2, 1)} -> {int(cov1.sum())}, "
        f"changes={coverage_changes}")

    d1 = diag0["finite"]
    d2 = diag0["nonzero"] and coverage_changes

    # --- D3 direction ---------------------------------------------------------------
    with torch.no_grad():
        before = G.classifier_score(model, G.decode_latent_to_video(vae, latent),
                                    args.target_index, prep, 0)
        stepped = latent + args.step_size * grad0.to(latent.dtype)
        after = G.classifier_score(model, G.decode_latent_to_video(vae, stepped),
                                   args.target_index, prep, 0)
    delta = float(after - before)
    d3 = delta > 0
    log(f"D3 direction: log p(target) {float(before):.6f} -> {float(after):.6f} "
        f"(delta {delta:+.6f}) with step_size {args.step_size}")

    gate_d = bool(d1 and d2 and d3)
    result = {
        "gates": {
            "D1_gradient_finite": bool(d1),
            "D2_gradient_nonzero_and_coverage_rotates": bool(d2),
            "D3_update_direction_correct": bool(d3),
            "D4_peak_gpu_gib": peak,
        },
        "gradient_diagnostics": diag0,
        "grad_scale_used": scale0,
        "latent_window": {"frames": args.latent_window,
                          "offset0_window": list(win0), "offset_half_window": list(win1),
                          "total_latent_frames": int(latent.shape[2])},
        "vae_gradient_checkpointing": True,
        "recommended_classifier_scale": 0.02,
        "recommended_scale_rationale": (
            "A guided step moves the latent by this fraction of its own norm, along the "
            "normalised gradient. 0.02 over roughly 30 executed steps is a conservative "
            "starting point: it is small enough that no single step dominates, and the "
            "normalisation makes the step size mean the same thing at every timestep "
            "regardless of how steep the classifier happens to be there."),
        "frame_coverage": {"offset_0": int(cov0.sum()),
                           "offset_half": int(cov1.sum()),
                           "latent_frames": int(cov0.numel()),
                           "changes_with_offset": coverage_changes},
        "direction_check": {"log_p_before": float(before), "log_p_after": float(after),
                            "delta": delta, "step_size": args.step_size},
        "timing": {"s_per_guided_step": step_s},
        "memory_note": ("Peak measured with the VAE decoder and classifier in the backward "
                        "graph and no transformer loaded. A full guided generation also "
                        "holds the transformer and its activations, so this is a lower "
                        "bound on that run's peak, not a prediction of it."),
        "target": {"index": args.target_index,
                   "name": (label_map.get("binary_index_to_name", {})
                            .get(str(args.target_index)))},
    }
    (OUT / "probe_guidance.json").write_text(json.dumps(result, indent=2))

    mf.write_manifest(
        mf.build_run_manifest(
            stage="guided_step_probe",
            environment=mf.environment_snapshot(),
            settings={"classifier_checkpoint": args.classifier,
                      "cogvideox_checkpoint": args.checkpoint,
                      "preprocess": list(prep), "target_index": args.target_index,
                      "step_size": args.step_size, "vae_tiling": args.vae_tiling,
                      "transformer_loaded": False,
                      "backward_graph": "VAE decoder and classifier only"},
            inputs={"source_latent_from": str(pre_path),
                    "latent_shape": list(latent.shape)},
            outputs=result["gates"],
            timing=result["timing"],
            memory={"peak_gpu_gib": peak},
            notes=("Stage D probe. One guided step, nothing generated. CogVideoX weights "
                   "are frozen and the transformer is not loaded."),
        ),
        OUT / "manifest_probe.json")

    log(f"\nGate D: {'PASS' if gate_d else 'FAIL'}")
    log(f"wrote {OUT}/probe_guidance.json")
    return 0 if gate_d else 1


if __name__ == "__main__":
    raise SystemExit(main())
