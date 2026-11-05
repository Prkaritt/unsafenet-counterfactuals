#!/usr/bin/env python
"""Matched base-versus-LoRA edits for several source clips in one job.

For every source: one edit with no adapter, one with the existing trained adapter.
Within a pair, everything is held identical -- source clip, seed, scheduler, dtype,
prompt, negative prompt, strength, text guidance, steps, resolution, frame count --
and the initial noise is identical because the generator is re-seeded from the same
value immediately before each call.

**No classifier-gradient guidance anywhere.** This script never imports the guidance
module and never computes a classifier gradient.

**No training.** The adapter is loaded from disk and its sha256 is recorded.

The base pass runs before the adapter is ever loaded, and after the LoRA pass the
adapter state is reported again, so "disabled for base, active for LoRA" is checked
rather than assumed.
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

from unsafenet_cf import compare as cmp  # noqa: E402
from unsafenet_cf import manifest as mf  # noqa: E402
from unsafenet_cf import preprocess as pp  # noqa: E402

OUT = REPO / "outputs" / "multi_walkway"


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    import torch
    from diffusers import CogVideoXVideoToVideoPipeline
    from diffusers.utils import export_to_video, load_video

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="outputs/multi_walkway/generation_plan.json")
    ap.add_argument("--lora-dir", default="outputs/lora/adapter")
    ap.add_argument("--lora-weight-name", default="pytorch_lora_weights.safetensors")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    plan = json.loads((REPO / args.plan).read_text())
    sources = plan["sources"]
    st = plan["settings"]
    OUT.mkdir(parents=True, exist_ok=True)

    import os
    ckpt = args.checkpoint or os.environ.get(
        "COGVIDEOX_CHECKPOINT",
        "/ocean/projects/cis260288p/puprety/checkpoints/CogVideoX-2b")
    revision = (REPO / "bridges2" / "checkpoint_revision.txt").read_text().strip()
    lora_dir = REPO / args.lora_dir
    lora_file = lora_dir / args.lora_weight_name
    if not lora_file.exists():
        raise SystemExit(f"adapter not found: {lora_file}")
    lora_sha = mf.sha256_file(lora_file)
    scale = args.lora_alpha / args.rank

    log(f"checkpoint : {ckpt} (revision {revision})")
    log(f"adapter    : {lora_file}")
    log(f"adapter sha256 : {lora_sha}")
    log(f"adapter scale  : {scale} (lora_alpha {args.lora_alpha} / rank {args.rank})")
    log(f"settings   : {st}")
    log(f"sources    : {[s['clip_id'] for s in sources]}")
    log("classifier-gradient guidance: DISABLED (never imported)")

    t0 = time.time()
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(ckpt, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder):
        for p in m.parameters():
            p.requires_grad_(False)
    log(f"pipeline loaded in {time.time() - t0:.1f}s; scheduler {type(pipe.scheduler).__name__}")

    def n_lora_params() -> int:
        return sum(1 for n, _ in pipe.transformer.named_parameters() if "lora_" in n)

    base_lora_params = n_lora_params()
    log(f"LoRA parameter tensors on the transformer before loading: {base_lora_params}")
    if base_lora_params != 0:
        raise SystemExit("transformer already carries LoRA parameters; base arm would be contaminated")

    def generate(src, arm: str) -> tuple[dict, np.ndarray]:
        frames_in = load_video(str(REPO / src["preprocessed"]))
        mf.reset_peak_memory()
        gen = torch.Generator(device="cpu").manual_seed(st["seed"])
        t = time.time()
        res = pipe(video=frames_in, prompt=src["prompt"],
                   negative_prompt=st["negative_prompt"],
                   strength=st["strength"], num_inference_steps=st["num_inference_steps"],
                   guidance_scale=st["guidance_scale"], num_videos_per_prompt=1,
                   generator=gen)
        secs = time.time() - t
        frames = res.frames[0]
        path = OUT / "videos" / f"{src['clip_id']}_{arm}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(path), fps=pp.TARGET_FPS)
        arr = cmp.to_array(frames)
        srcarr = cmp.to_array(frames_in)
        rec = {"arm": arm, "path": str(path.relative_to(REPO)),
               "generation_s": round(secs, 1),
               "peak_gpu_gib": mf.peak_memory_gb(),
               "psnr_vs_source_db": cmp.psnr(srcarr, arr),
               "mean_abs_diff_vs_source": cmp.mean_abs_diff(srcarr, arr),
               "lora_param_tensors_present": n_lora_params()}
        log(f"  {src['clip_id']} {arm}: {secs:.1f}s peak {rec['peak_gpu_gib']:.2f} GiB "
            f"PSNR {rec['psnr_vs_source_db']:.3f} dB MAD {rec['mean_abs_diff_vs_source']:.4f} "
            f"lora_tensors {rec['lora_param_tensors_present']}")
        return rec, arr

    runs = {}
    log("\n================ BASE ARM (no adapter loaded) ================")
    for src in sources:
        rec, _ = generate(src, "base")
        if rec["lora_param_tensors_present"] != 0:
            raise SystemExit("adapter present during a base generation; aborting")
        runs.setdefault(src["clip_id"], {})["base"] = rec

    log("\n================ loading adapter ================")
    pipe.load_lora_weights(str(lora_dir), weight_name=args.lora_weight_name,
                           adapter_name="cogvideox-lora")
    pipe.set_adapters(["cogvideox-lora"], [scale])
    active = pipe.get_active_adapters() if hasattr(pipe, "get_active_adapters") else None
    after = n_lora_params()
    log(f"active adapters: {active}; LoRA parameter tensors now: {after}")
    if after == 0:
        raise SystemExit("adapter did not attach; refusing to report a LoRA arm")
    adapter_check = {"active_adapters": list(active) if active else None,
                     "lora_param_tensors_before": base_lora_params,
                     "lora_param_tensors_after": after,
                     "scale_applied": scale,
                     "scaling_convention": (
                         "diffusers CogVideoXLoraLoaderMixin: set_adapters(names, weights) "
                         "multiplies the adapter delta by this weight. The trainer used "
                         f"lora_alpha={args.lora_alpha} with rank={args.rank}, and peft "
                         "already folds alpha/rank into the stored layers, so the weight "
                         "passed here is alpha/rank = 1.0, i.e. the adapter at the strength "
                         "it was trained at.")}

    log("\n================ LORA ARM ================")
    for src in sources:
        rec, _ = generate(src, "lora")
        runs[src["clip_id"]]["lora"] = rec

    manifest = {
        "_what_this_is": (
            "Matched base-versus-LoRA prompt-guided video-to-video edits across several "
            "walkway-violation source clips. Not LD-ViCE: no target-model gradients. "
            "No training in this job. Candidate counterfactuals until human review."),
        "classifier_gradient_guidance": "disabled in every run",
        "checkpoint": "THUDM/CogVideoX-2b", "checkpoint_local_dir": str(ckpt),
        "revision": revision,
        "scheduler": type(pipe.scheduler).__name__,
        "adapter": {"dir": str(lora_dir.relative_to(REPO)),
                    "weight_name": args.lora_weight_name, "sha256": lora_sha,
                    "rank": args.rank, "lora_alpha": args.lora_alpha,
                    "trained_by_job": "50230058", **adapter_check},
        "settings": st,
        "sources": sources,
        "runs": runs,
        "environment": mf.environment_snapshot(),
    }
    mf.write_manifest(manifest, OUT / "manifest_multi_walkway.json")
    log(f"\nwrote {OUT}/manifest_multi_walkway.json")
    log("All outputs are CANDIDATE counterfactuals. Human review required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
