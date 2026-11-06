#!/usr/bin/env python
"""Classifier-guided generation for the panel-cover task, with its own control.

This stage only runs if its prerequisites hold, and the prerequisites are checked here
rather than assumed:

* the panel classifier reaches usable validation performance,
* its diagnostics show it responds to the panel region more than to control regions,
* a gradient taken at the predicted clean latent is finite and non-zero,
* and the backward pass through the VAE decoder fits on the card.

``--mode probe`` answers the last two for a single step and writes the numbers. ``--mode
generate`` then produces a **matched pair**: one guided run and one unguided control, with
the same source, prompt, seed, scheduler, strength, step count and initial noise. The only
difference between them is the classifier term.

Two things this stage cannot do, stated up front:

* **A label flip is not a panel closing.** The guided output's classifier probability and
  its actual panel state are reported as separate quantities, and the panel state is judged
  by `pc_evaluate.py` and by eye, never by the classifier that was optimised against.
* **A flip that the controls also show is not attributable to guidance.** The unguided run
  and the VAE round trip are the comparison. If they flip too, the guided flip means
  nothing, and that is reported as the finding rather than buried.

CogVideoX stays frozen throughout: no LoRA, no optimiser, no parameter updated. The
denoising loop itself is `scripts/guided_generate.py:run_loop`, reused unchanged so the two
experiments share one verified implementation.
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
sys.path.insert(0, str(REPO / "scripts"))

from unsafenet_cf import classifier as C      # noqa: E402
from unsafenet_cf import compare as cmp       # noqa: E402
from unsafenet_cf import guidance as G        # noqa: E402
from unsafenet_cf import manifest as mf       # noqa: E402
from unsafenet_cf import preprocess as prep   # noqa: E402

OUT = REPO / "outputs" / "panel_cover"


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    import torch
    from diffusers import CogVideoXVideoToVideoPipeline
    from diffusers.utils import load_video

    import guided_generate as gg  # reuse the verified denoising loop

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "generate"], required=True)
    ap.add_argument("--case", required=True, help="case id from the plan")
    ap.add_argument("--input-arm", default="source",
                    choices=["source", "composite"],
                    help="which clip guidance starts from; 'source' is the honest test "
                         "of whether guidance alone can close a panel")
    ap.add_argument("--settings", default="configs/panel_cover_guided.json")
    ap.add_argument("--classifier", default="outputs/panel_cover/classifier/"
                                            "panel_classifier.pt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--classifier-scale", type=float, default=0.02)
    ap.add_argument("--guidance-every", type=int, default=1)
    ap.add_argument("--latent-window", type=int, default=G.DEFAULT_LATENT_WINDOW)
    ap.add_argument("--step-fractions", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                    help="probe only: latent steps along the gradient, each a fraction of "
                         "the latent's own norm, scored one by one")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log("no CUDA device; this stage must run on a GPU node")
        return 2
    ckpt = args.checkpoint or os.environ.get(
        "COGVIDEOX_CHECKPOINT",
        "/ocean/projects/cis260288p/puprety/checkpoints/CogVideoX-2b")
    cls_path = REPO / args.classifier
    if not cls_path.exists():
        log(f"no panel classifier at {cls_path}; stage 5 has no prerequisite and is "
            "blocked. This is a blocker to document, not a reason to substitute another "
            "classifier.")
        return 2
    cfg = json.loads((REPO / args.settings).read_text())
    st = cfg["settings"]
    device = torch.device("cuda")

    in_path = OUT / args.case / "videos" / f"{args.case}_{args.input_arm}.mp4"
    if not in_path.exists():
        log(f"missing input {in_path}")
        return 2

    blob = torch.load(cls_path, map_location="cpu", weights_only=False)
    pre = blob["preprocess"]
    prep_tuple = (pre["frames"], pre["height"], pre["width"])
    label_map = blob.get("label_map", {})
    target_index = int(label_map.get("target_binary_index", 1))
    model = C.build_model(num_classes=2, pretrained=False)
    model.load_state_dict(blob["state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"classifier epoch {blob.get('epoch')}, target index {target_index} "
        f"({label_map.get('binary_index_to_name', {}).get(str(target_index))})")
    log(f"validation at checkpoint: {blob.get('val_metrics', {}).get('balanced_accuracy')} "
        "balanced accuracy")

    t0 = time.time()
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(ckpt, torch_dtype=torch.float16
                                                         ).to(device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.vae.enable_gradient_checkpointing()
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
        for p in module.parameters():
            p.requires_grad_(False)
    log(f"pipeline loaded in {time.time() - t0:.1f}s; CogVideoX frozen, no LoRA")

    prompt = cfg["prompts"][args.case]
    with torch.no_grad():
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt, negative_prompt=st["negative_prompt"],
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            max_sequence_length=226, device=device, dtype=pipe.transformer.dtype)
    before = torch.cuda.memory_allocated() / 1024 ** 3
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    if hasattr(pipe, "register_modules"):
        try:
            pipe.register_modules(text_encoder=None)
        except Exception:
            pass
    mf.free_gpu()
    after = torch.cuda.memory_allocated() / 1024 ** 3
    log(f"text encoder released: {before:.2f} -> {after:.2f} GiB "
        f"(freed {before - after:.2f} GiB)")

    source_frames = load_video(str(in_path))
    src_arr = cmp.to_array(source_frames)
    out_dir = OUT / args.case / "guided"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "probe":
        mf.reset_peak_memory()
        video_tensor = pipe.video_processor.preprocess_video(
            source_frames, height=prep.TARGET_HEIGHT, width=prep.TARGET_WIDTH
        ).to(device=device, dtype=pipe.transformer.dtype)
        with torch.no_grad():
            lat = pipe.vae.encode(video_tensor).latent_dist.mode()
            lat = lat * pipe.vae.config.scaling_factor
        log(f"latent {tuple(lat.shape)}")
        t = time.time()
        grad, score, scale_used, win = G.latent_gradient(
            pipe.vae, model, lat, target_index, prep_tuple, offset=0,
            window=args.latent_window)
        secs = time.time() - t
        diag = G.guidance_diagnostics(grad)
        # Steps along the gradient, to show whether the objective actually moves and at
        # what magnitude the decode stops being valid. A single step cannot show either:
        # the first version of this probe took one step of 50 times the latent norm and
        # reported NaN, which says nothing about the gradient and everything about the
        # step. Each fraction below is a multiple of the latent's own norm.
        moved = None
        sweep = []
        if diag["nonzero"]:
            unit = grad / grad.norm() * lat.float().norm()
            for frac in args.step_fractions:
                with torch.no_grad():
                    lat2 = (lat.float() + frac * unit).to(lat.dtype)
                    video = G.decode_latent_to_video(pipe.vae, lat2)
                    s = float(G.classifier_score(model, video, target_index, prep_tuple))
                sweep.append({"step_fraction_of_latent_norm": frac,
                              "log_p_target": s if s == s else None,
                              "finite": s == s})
                log(f"  step {frac:6.3f} x ||latent||: log p(target) "
                    + (f"{s:+.4f}" if s == s else "NaN"))
            finite = [r for r in sweep if r["finite"]]
            moved = finite[0]["log_p_target"] if finite else None
        rec = {
            "mode": "probe",
            "case": args.case,
            "input": str(in_path.relative_to(REPO)),
            "target_index": target_index,
            "latent_shape": list(lat.shape),
            "gradient": diag,
            "grad_scale_used": float(scale_used),
            "latent_window": list(win),
            "log_p_target_before": float(score),
            "log_p_target_after_smallest_step": moved,
            "step_sweep": sweep,
            "_step_sweep_meaning": ("Each row moves the latent along the gradient by that "
                                    "fraction of the latent's own norm, decodes it and "
                                    "scores it. A rising log p(target) means the gradient "
                                    "points somewhere the classifier likes; it does not "
                                    "mean the panel closed. A NaN row means the decode "
                                    "overflowed at that magnitude, which bounds the step "
                                    "a guided loop may take, and is a property of the step "
                                    "size rather than of the gradient."),
            "step_fractions": list(args.step_fractions),
            "seconds": round(secs, 1),
            "peak_gpu_gib": mf.peak_memory_gb(),
            "gate": {
                "finite": diag["finite"],
                "nonzero": diag["nonzero"],
                "fits_in_memory": True,
                "_meaning": ("All three must hold before a guided generation is worth "
                             "its allocation. A finite, non-zero gradient says the path "
                             "from latent to classifier carries a derivative; it says "
                             "nothing about whether following it closes a panel."),
            },
        }
        (out_dir / "probe.json").write_text(json.dumps(rec, indent=1))
        log(f"gradient finite={diag['finite']} nonzero={diag['nonzero']} "
            f"norm={diag['norm']:.4g} peak {rec['peak_gpu_gib']:.2f} GiB")
        log(f"log p(target): {float(score):.5f} -> {moved}")
        log(f"wrote {out_dir / 'probe.json'}")
        return 0 if diag["nonzero"] else 1

    # --- matched guided and unguided pair ----------------------------------------
    loop_cfg = {"prompt": prompt, "negative_prompt": st["negative_prompt"],
                "seed": st["seed"], "num_inference_steps": st["num_inference_steps"],
                "guidance_scale": st["guidance_scale"]}
    runs = {}
    for name, scale in (("unguided_control", 0.0), ("guided", args.classifier_scale)):
        log(f"\n--- {name} (classifier_scale={scale})")
        mf.reset_peak_memory()
        out = gg.run_loop(pipe, pipe.vae, model, prep_tuple, source_frames, loop_cfg,
                          embeds_pair=(prompt_embeds, negative_embeds),
                          seed=st["seed"], steps=st["num_inference_steps"],
                          strength=st["strength"],
                          guidance_scale=st["guidance_scale"],
                          classifier_scale=scale,
                          target_index=target_index,
                          guidance_every=args.guidance_every,
                          latent_window=args.latent_window, device=device)
        path = out_dir / f"{args.case}_{name}.mp4"
        from PIL import Image
        from diffusers.utils import export_to_video
        export_to_video([f if hasattr(f, "save") else Image.fromarray(np.asarray(f))
                         for f in out["frames"]], str(path), fps=prep.TARGET_FPS)
        arr = cmp.to_array(out["frames"])
        probs = gg.classify_clip(model, arr, prep_tuple, device)
        runs[name] = {
            "classifier_scale": scale,
            "output": str(path.relative_to(REPO)),
            "executed_steps": out["executed_steps"],
            "generation_s": round(out["generation_s"], 1),
            "peak_gpu_gib": mf.peak_memory_gb(),
            "psnr_vs_input_db": cmp.psnr(src_arr, arr),
            "mean_abs_diff_vs_input": cmp.mean_abs_diff(src_arr, arr),
            "classifier_probs": {"opened_panel_cover": round(probs[0], 5),
                                 "closed_panel_cover": round(probs[1], 5)},
            "predicted_binary_index": int(np.argmax(probs)),
            "grad_log": out["grad_log"],
        }
        log(f"  {name}: {out['generation_s']:.1f}s, P(closed) {probs[1]:.4f}, "
            f"peak {runs[name]['peak_gpu_gib']:.2f} GiB")

    base_probs = gg.classify_clip(model, src_arr, prep_tuple, device)
    rec = {
        "mode": "generate",
        "case": args.case,
        "input": str(in_path.relative_to(REPO)),
        "input_arm": args.input_arm,
        "matched_settings": {
            "seed": st["seed"], "strength": st["strength"],
            "num_inference_steps": st["num_inference_steps"],
            "guidance_scale": st["guidance_scale"],
            "scheduler": type(pipe.scheduler).__name__,
            "prompt": prompt, "negative_prompt": st["negative_prompt"],
            "_matched": ("Both runs share every setting above and the same generator "
                         "seed, so they start from identical initial noise. The only "
                         "difference is the classifier term."),
        },
        "input_classifier_probs": {"opened_panel_cover": round(base_probs[0], 5),
                                   "closed_panel_cover": round(base_probs[1], 5)},
        "runs": runs,
        "label_flip": {
            "input_prediction": int(np.argmax(base_probs)),
            "unguided_prediction": runs["unguided_control"]["predicted_binary_index"],
            "guided_prediction": runs["guided"]["predicted_binary_index"],
            "_attribution_rule": ("A guided flip counts as evidence about guidance only "
                                  "if the unguided control did NOT flip. Panel state is "
                                  "a separate question measured in "
                                  "outputs/panel_cover/evaluation.json and by eye."),
        },
        "environment": mf.environment_snapshot(),
    }
    (out_dir / "guided.json").write_text(json.dumps(rec, indent=1))
    log(f"\nwrote {out_dir / 'guided.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
