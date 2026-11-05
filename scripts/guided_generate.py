#!/usr/bin/env python3
"""Stage E: one classifier-guided candidate for 0_tr143, plus a matched unguided control.

Both clips come out of the **same custom denoising loop**, with the classifier term scaled
to zero for the control. That is a stricter control than reusing an earlier pipeline
output: source, seed, checkpoint, prompt, negative prompt, scheduler, dtype, strength,
guidance scale, steps, resolution and frame count are identical by construction, and so is
the loop itself. The only difference is the classifier term.

Why a custom loop at all: ``CogVideoXVideoToVideoPipeline.__call__`` runs under
``torch.no_grad()``, so no gradient of any kind can be taken inside it. The loop below
reproduces its mathematics and opens a gradient path only where guidance needs one.

Where the gradient is taken
---------------------------
At each step the scheduler returns ``pred_original_sample``, the model's estimate of the
clean latent. That estimate is detached, decoded through the VAE, scored by the classifier,
and differentiated. The resulting gradient nudges the next latent. The transformer is never
part of a backward graph, so its activations are not retained.

CogVideoX weights are frozen throughout. No LoRA training. The output is a candidate
counterfactual pending human review, and a classifier prediction flip is not proof of
corrected behaviour.
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
from unsafenet_cf import compare as cmp     # noqa: E402
from unsafenet_cf import guidance as G      # noqa: E402
from unsafenet_cf import manifest as mf     # noqa: E402
from unsafenet_cf import preprocess as pp   # noqa: E402

OUT = REPO / "outputs" / "guided"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_configs() -> dict:
    return json.loads((REPO / "configs" / "experiment.json").read_text())


def to_vae_layout(latents):
    """Pipeline layout (B, F, C, h, w) -> VAE layout (B, C, F, h, w)."""
    return latents.permute(0, 2, 1, 3, 4)


def to_pipe_layout(latents):
    """VAE layout (B, C, F, h, w) -> pipeline layout (B, F, C, h, w)."""
    return latents.permute(0, 2, 1, 3, 4)


def run_loop(pipe, vae, model, prep, source_frames, cfg, *, embeds_pair, seed: int,
             steps: int, strength: float, guidance_scale: float, classifier_scale: float,
             target_index: int, guidance_every: int, latent_window: int, device) -> dict:
    """One denoising run. ``classifier_scale`` of 0.0 gives the unguided control."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)

    prompt_embeds, negative_embeds = embeds_pair
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps, num_steps = pipe.get_timesteps(steps, pipe.scheduler.timesteps,
                                              strength, device)
    latent_timestep = timesteps[:1].repeat(1)
    log(f"  requested {steps} steps, executing {num_steps}")

    video_tensor = pipe.video_processor.preprocess_video(source_frames,
                                                         height=pp.TARGET_HEIGHT,
                                                         width=pp.TARGET_WIDTH)
    video_tensor = video_tensor.to(device=device, dtype=prompt_embeds.dtype)

    with torch.no_grad():
        latents = pipe.prepare_latents(
            video=video_tensor, batch_size=1, num_channels_latents=16,
            height=pp.TARGET_HEIGHT, width=pp.TARGET_WIDTH,
            dtype=prompt_embeds.dtype, device=device, generator=generator,
            latents=None, timestep=latent_timestep)

    embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
    image_rotary_emb = None
    if pipe.transformer.config.use_rotary_positional_embeddings:
        image_rotary_emb = pipe._prepare_rotary_positional_embeddings(
            pp.TARGET_HEIGHT, pp.TARGET_WIDTH, latents.size(1), device)

    grad_log = []
    old_pred_original_sample = None
    t0 = time.time()

    for i, t in enumerate(timesteps):
        with torch.no_grad():
            latent_input = torch.cat([latents] * 2)
            latent_input = pipe.scheduler.scale_model_input(latent_input, t)
            timestep = t.expand(latent_input.shape[0])
            noise_pred = pipe.transformer(
                hidden_states=latent_input, encoder_hidden_states=embeds,
                timestep=timestep, image_rotary_emb=image_rotary_emb,
                return_dict=False)[0].float()
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (cond - uncond)

            step_out = pipe.scheduler.step(noise_pred, t, latents.float(),
                                           return_dict=True)
            latents_next = step_out.prev_sample.to(latents.dtype)
            pred_x0 = step_out.pred_original_sample.to(latents.dtype)

        if classifier_scale > 0.0 and (i % guidance_every == 0):
            # Gradient is taken at the predicted clean latent, which is detached first.
            # The transformer produced it, but no graph back to the transformer exists,
            # so it is not needed on the device while the gradient is computed. Parking
            # it on the CPU for that window returns about 3.4 GiB, which is what the
            # decoder's backward pass was short of.
            pipe.transformer.to("cpu")
            torch.cuda.empty_cache()
            try:
                grad_vae, score, grad_scale_used, win = G.latent_gradient(
                    vae, model, to_vae_layout(pred_x0), target_index, prep, offset=i,
                    window=latent_window)
            finally:
                pipe.transformer.to(device)
            grad = to_pipe_layout(grad_vae)
            norm = grad.norm()
            if torch.isfinite(norm) and norm > 0:
                # Normalise, so the step size means the same thing at every timestep
                # regardless of how steep the classifier happens to be there.
                #
                # The update is cast back to the latents' dtype before it is added. The
                # gradient is computed in float32 for numerical headroom, and adding it
                # untouched silently promoted the latents to float32, after which the
                # next step fed float32 into a float16 transformer and died with
                # "mat1 and mat2 must have the same dtype, but got Float and Half".
                update = (classifier_scale * grad / norm * latents_next.float().norm())
                latents_next = latents_next + update.to(latents_next.dtype)
            grad_log.append({"step": i, "timestep": int(t),
                             "grad_norm": float(norm), "log_p_target": float(score),
                             "grad_scale_used": float(grad_scale_used),
                             "latent_window": list(win),
                             "finite": bool(torch.isfinite(grad).all()),
                             "applied": bool(torch.isfinite(norm) and norm > 0)})

        if latents_next.dtype != latents.dtype:
            raise RuntimeError(
                f"latent dtype drifted from {latents.dtype} to {latents_next.dtype} at "
                f"step {i}; the transformer expects {latents.dtype}")
        latents = latents_next
        old_pred_original_sample = pred_x0
        if (i + 1) % 10 == 0 or i + 1 == len(timesteps):
            log(f"  step {i + 1}/{len(timesteps)}")

    gen_s = time.time() - t0

    with torch.no_grad():
        video = pipe.decode_latents(latents)
        frames = pipe.video_processor.postprocess_video(video=video,
                                                        output_type="pil")[0]
    return {"frames": frames, "generation_s": gen_s, "grad_log": grad_log,
            "executed_steps": int(num_steps)}


def classify_clip(model, arr_uint8: np.ndarray, prep, device) -> list[float]:
    import torch
    x = torch.from_numpy(arr_uint8.astype(np.float32) / 255.0)[None].to(device)
    with torch.no_grad():
        logits = model(C.prepare(x, *prep))
        return torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()


def main(argv=None) -> int:
    import torch
    from diffusers import CogVideoXVideoToVideoPipeline
    from diffusers.utils import export_to_video, load_video

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", default=str(REPO / "outputs" / "classifier" /
                                                "classifier_best.pt"))
    ap.add_argument("--checkpoint", default=os.environ.get("COGVIDEOX_CHECKPOINT"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--classifier-scale", type=float, default=0.02,
                    help="fraction of the latent norm moved per guided step")
    ap.add_argument("--guidance-every", type=int, default=1)
    ap.add_argument("--latent-window", type=int, default=G.DEFAULT_LATENT_WINDOW)

    ap.add_argument("--target-index", type=int, default=1)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        log("No CUDA device. Run this on a GPU node.")
        return 2
    if not args.checkpoint or not Path(args.checkpoint).exists():
        log("CogVideoX checkpoint not found; set COGVIDEOX_CHECKPOINT.")
        return 2
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = load_configs()
    seed = args.seed if args.seed is not None else cfg["seed"]
    steps = args.steps if args.steps is not None else cfg["num_inference_steps"]
    guidance_scale = args.guidance if args.guidance is not None else cfg["guidance_scale"]

    ck = torch.load(args.classifier, map_location="cpu", weights_only=False)
    prep = (ck["preprocess"]["frames"], ck["preprocess"]["height"],
            ck["preprocess"]["width"])
    model = C.build_model(num_classes=2, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16).to(device)
    # Default tile size. Shrinking it to 120x180 did cut the probe's peak from 41.852 to
    # 29.018 GiB, but it also changed the encoder's output geometry: the latent came back
    # 62 rows tall where prepare_latents builds 60, and the run died in add_noise with
    # "The size of tensor a (62) must match the size of tensor b (60)". Tile size is not
    # a free memory knob here, so the memory comes from somewhere else instead.
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.vae.enable_gradient_checkpointing()
    log("VAE tiling: default tiles, gradient checkpointing on")
    # CogVideoX stays frozen. No LoRA, no optimiser, no parameter ever updated.
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
        for p in module.parameters():
            p.requires_grad_(False)
    load_s = time.time() - t0
    log(f"pipeline loaded in {load_s:.1f}s; CogVideoX frozen")

    # Encode the prompts once, then release the T5-XXL text encoder. It is roughly 9.5 GiB
    # in float16 and is never used again after this call, but the pipeline holds it for
    # the whole run. A guided step needs about 10 GiB for the decoder backward pass, and
    # on a 44.42 GiB card that is exactly the difference between fitting and an OOM: the
    # first attempt died asking for 10.01 GiB with 7.56 GiB free.
    with torch.no_grad():
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=cfg["prompt"], negative_prompt=cfg["negative_prompt"],
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            max_sequence_length=226, device=device, dtype=pipe.transformer.dtype)
    embeds_pair = (prompt_embeds, negative_embeds)
    before = torch.cuda.memory_allocated() / 1024 ** 3
    # Setting the attribute to None is not enough: a DiffusionPipeline keeps its own
    # reference through register_modules, so the weights stay on the card. Measured:
    # 14.75 -> 14.75 GiB, no change at all. Moving the module to CPU first releases the
    # device memory whatever else still points at the object.
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
    log(f"text encoder released: {before:.2f} -> {after:.2f} GiB allocated "
        f"(freed {before - after:.2f} GiB)")
    if before - after < 1.0:
        log("WARNING: the text encoder did not release device memory; a guided step "
            "needs about 10 GiB and this run may exhaust the card.")

    pre_path = REPO / "outputs" / "preprocessed" / "0_tr143_720x480x49.mp4"
    source_frames = load_video(str(pre_path))
    src_arr = cmp.to_array(source_frames)

    runs = {}
    for name, scale in (("unguided_control", 0.0),
                        ("guided", args.classifier_scale)):
        log(f"\n--- {name} (classifier_scale={scale})")
        mf.reset_peak_memory()
        out = run_loop(pipe, pipe.vae, model, prep, source_frames, cfg,
                       embeds_pair=embeds_pair,
                       seed=seed, steps=steps, strength=args.strength,
                       guidance_scale=guidance_scale, classifier_scale=scale,
                       target_index=args.target_index,
                       guidance_every=args.guidance_every,
                       latent_window=args.latent_window, device=device)
        out["peak_gpu_gib"] = mf.peak_memory_gb()
        path = OUT / f"{name}_seed{seed}_s{args.strength}_cs{scale}.mp4"
        export_to_video(out["frames"], str(path), fps=pp.TARGET_FPS)
        arr = cmp.to_array(out["frames"])
        out["path"] = str(path)
        out["psnr_vs_source_db"] = cmp.psnr(src_arr, arr)
        out["mean_abs_diff_vs_source"] = cmp.mean_abs_diff(src_arr, arr)
        out["array"] = arr
        out.pop("frames", None)   # the clip is on disk; the PIL frames are not needed
        runs[name] = out
        log(f"  {out['generation_s']:.1f}s, peak {out['peak_gpu_gib']} GiB, "
            f"PSNR {out['psnr_vs_source_db']:.2f} dB -> {path.name}")
        mf.free_gpu()

    # --- classifier probabilities on every clip, including the controls -------------
    recon_path = REPO / "outputs" / "stage3_vae" / "reconstruction.mp4"
    probs = {"source": classify_clip(model, src_arr, prep, device)}
    if recon_path.exists():
        probs["vae_reconstruction"] = classify_clip(
            model, cmp.to_array(load_video(str(recon_path))), prep, device)
    for name in runs:
        probs[name] = classify_clip(model, runs[name]["array"], prep, device)
    log("\nclassifier p(safe walkway):")
    for k, v in probs.items():
        log(f"  {k:20s} {v[1]:.4f}")

    rows = [("source", src_arr)]
    if recon_path.exists():
        rows.append(("vae_reconstruction", cmp.to_array(load_video(str(recon_path)))))
    rows.append(("unguided_control", runs["unguided_control"]["array"]))
    rows.append(("guided", runs["guided"]["array"]))
    grid = cmp.comparison_grid(rows, out_path=OUT / "comparison_guided_vs_unguided.png")
    log(f"comparison sheet: {grid['path']}")

    settings = {
        "checkpoint": cfg["checkpoint"], "checkpoint_path": args.checkpoint,
        "classifier_checkpoint": args.classifier,
        "label_map": ck.get("label_map"),
        "seed": seed, "num_inference_steps": steps,
        "executed_steps": runs["guided"]["executed_steps"],
        "strength": args.strength, "guidance_scale": guidance_scale,
        "classifier_scale": args.classifier_scale,
        "guidance_every": args.guidance_every, "target_index": args.target_index,
        "latent_window_frames": args.latent_window,
        "vae_gradient_checkpointing": True,
        "transformer_offloaded_during_guidance": True,
        "prompt": cfg["prompt"], "negative_prompt": cfg["negative_prompt"],
        "dtype": "torch.float16", "preprocess": list(prep),
        "cogvideox_frozen": True, "lora_training": False,
        "identical_between_runs": ["source", "seed", "checkpoint", "prompt",
                                   "negative_prompt", "scheduler", "dtype", "strength",
                                   "guidance_scale", "steps", "resolution", "frames",
                                   "denoising loop"],
    }
    mf.write_manifest(
        mf.build_run_manifest(
            stage="classifier_guided_edit",
            environment=mf.environment_snapshot(),
            settings=settings,
            inputs={"preprocessed_clip": str(pre_path)},
            outputs={
                "guided_clip": runs["guided"]["path"],
                "unguided_control_clip": runs["unguided_control"]["path"],
                "comparison_sheet": str(grid["path"]),
                "classifier_probabilities_p_safe": {k: v[1] for k, v in probs.items()},
                "psnr_vs_source_db": {k: runs[k]["psnr_vs_source_db"] for k in runs},
                "mean_abs_diff_vs_source": {k: runs[k]["mean_abs_diff_vs_source"]
                                            for k in runs},
            },
            timing={k: runs[k]["generation_s"] for k in runs} | {"pipeline_load_s": load_s},
            memory={k: runs[k]["peak_gpu_gib"] for k in runs},
            notes=("Classifier-guided candidate and matched unguided control from one "
                   "custom denoising loop. CogVideoX weights frozen; gradients taken only "
                   "at the predicted clean latent, through the VAE decoder and the "
                   "classifier. A prediction flip is not proof of corrected behaviour. "
                   "Official test-split evaluation is PENDING by user decision."),
        ),
        OUT / "manifest_guided.json")

    (OUT / "guidance_trace.json").write_text(json.dumps(
        {"guided": runs["guided"]["grad_log"]}, indent=2))
    log(f"\nwrote {OUT}/manifest_guided.json")
    log("Both clips are candidate counterfactuals pending human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
