#!/usr/bin/env python
"""Refine the rough composite with a frozen CogVideoX-2b. No LoRA, no classifier gradients.

The question this stage answers is narrow and worth stating exactly. The relocation has
already happened, in pixels, in the composite. CogVideoX is not being asked to move anyone.
It is being asked whether a low-strength video-to-video pass makes the rough composite look
more like real footage without undoing the relocation that is already there.

Three things are checked rather than assumed:

* **No adapter.** The transformer is inspected for LoRA parameter tensors before and after
  every generation, and a non-zero count aborts the run.
* **No guidance.** This script never imports the guidance module and never computes a
  classifier gradient.
* **Executed steps.** ``strength`` truncates the schedule, so a run requesting 50 steps at
  strength 0.2 executes 10. The manifest records the requested count and the executed count
  separately, and derives the per-step timing from the executed one. An earlier script in
  this project divided by the requested count and understated every per-step figure.

The pipeline enforces no trajectory and no mask constraint. Whether the refinement keeps the
worker where the composite put him is a measurement taken afterwards, never a guarantee.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsafenet_cf import compare as cmp  # noqa: E402
from unsafenet_cf import manifest as mf  # noqa: E402
from unsafenet_cf import preprocess as pp  # noqa: E402

OUT = REPO / "outputs" / "manual_path"


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    import torch
    from diffusers import CogVideoXVideoToVideoPipeline
    from diffusers.utils import export_to_video, load_video

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="configs/manual_path_sources.json")
    ap.add_argument("--settings", default="configs/manual_path_refine.json")
    ap.add_argument("--clips", default="", help="comma-separated subset")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    specs = json.loads((REPO / args.sources).read_text())["sources"]
    cfg = json.loads((REPO / args.settings).read_text())
    st = cfg["settings"]
    only = {c for c in args.clips.split(",") if c}

    ckpt = args.checkpoint or os.environ.get(
        "COGVIDEOX_CHECKPOINT",
        "/ocean/projects/cis260288p/puprety/checkpoints/CogVideoX-2b")
    revision = (REPO / "bridges2" / "checkpoint_revision.txt").read_text().strip()

    log(f"checkpoint : {ckpt} (revision {revision})")
    log(f"settings   : {st}")
    log("LoRA adapter: NEVER LOADED. classifier-gradient guidance: DISABLED (never imported)")

    t0 = time.time()
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(ckpt, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    for m in (pipe.transformer, pipe.vae, pipe.text_encoder):
        for p in m.parameters():
            p.requires_grad_(False)
    load_s = time.time() - t0
    log(f"pipeline loaded in {load_s:.1f}s; scheduler {type(pipe.scheduler).__name__}; "
        f"dtype {pipe.transformer.dtype}")

    def n_lora() -> int:
        return sum(1 for n, _ in pipe.transformer.named_parameters() if "lora_" in n)

    if n_lora() != 0:
        raise SystemExit("transformer already carries LoRA parameters; aborting")

    results = {}
    for spec in specs:
        cid = spec["clip_id"]
        if only and cid not in only:
            continue
        comp_path = OUT / cid / "videos" / f"{cid}_composite.mp4"
        if not comp_path.exists():
            log(f"{cid}: no composite at {comp_path}; skipping")
            results[cid] = {"error": "composite missing"}
            continue

        frames_in = load_video(str(comp_path))
        prompt = cfg["prompts"][cid]
        mf.reset_peak_memory()
        gen = torch.Generator(device="cpu").manual_seed(st["seed"])
        t = time.time()
        res = pipe(video=frames_in, prompt=prompt,
                   negative_prompt=st["negative_prompt"],
                   strength=st["strength"],
                   num_inference_steps=st["num_inference_steps"],
                   guidance_scale=st["guidance_scale"],
                   num_videos_per_prompt=1, generator=gen)
        secs = time.time() - t
        frames = res.frames[0]

        out_path = OUT / cid / "videos" / f"{cid}_refined.mp4"
        export_to_video(frames, str(out_path), fps=pp.TARGET_FPS)

        arr = cmp.to_array(frames)
        comp_arr = cmp.to_array(frames_in)
        src_arr = cmp.to_array(load_video(str(OUT / cid / "videos" / f"{cid}_source.mp4")))

        executed = min(int(st["num_inference_steps"] * st["strength"]),
                       st["num_inference_steps"])
        rec = {
            "clip_id": cid,
            "prompt": prompt,
            "negative_prompt": st["negative_prompt"],
            "seed": st["seed"],
            "strength": st["strength"],
            "num_inference_steps": st["num_inference_steps"],
            "denoising_steps_executed": executed,
            "_steps_note": ("strength truncates the schedule: executed = int(steps * "
                            "strength). Timing below is per executed step."),
            "guidance_scale": st["guidance_scale"],
            "scheduler": type(pipe.scheduler).__name__,
            "dtype": str(pipe.transformer.dtype),
            "checkpoint": ckpt,
            "checkpoint_revision": revision,
            "input": str(comp_path.relative_to(REPO)),
            "output": str(out_path.relative_to(REPO)),
            "generation_s": round(secs, 1),
            "s_per_denoising_step": round(secs / executed, 3) if executed else None,
            "peak_gpu_gib": mf.peak_memory_gb(),
            "lora_param_tensors_present": n_lora(),
            "psnr_vs_composite_db": cmp.psnr(comp_arr, arr),
            "mean_abs_diff_vs_composite": cmp.mean_abs_diff(comp_arr, arr),
            "psnr_vs_source_db": cmp.psnr(src_arr, arr),
            "mean_abs_diff_vs_source": cmp.mean_abs_diff(src_arr, arr),
            "_psnr_warning": ("Whole-frame PSNR is dominated by the static background and "
                              "is nearly blind to a worker filling a small part of the "
                              "frame. It is not evidence that the subject was preserved."),
        }
        if rec["lora_param_tensors_present"] != 0:
            raise SystemExit("adapter appeared during generation; aborting")
        log(f"  {cid}: {secs:.1f}s ({executed} executed steps, "
            f"{rec['s_per_denoising_step']}s/step) peak {rec['peak_gpu_gib']:.2f} GiB "
            f"PSNR vs composite {rec['psnr_vs_composite_db']:.3f} dB, "
            f"vs source {rec['psnr_vs_source_db']:.3f} dB")
        results[cid] = rec
        (OUT / cid / "refine.json").write_text(json.dumps(rec, indent=1))

    # A run over a subset must not erase the clips an earlier run already recorded. The
    # pilot ran one clip alone, so the manifest is merged rather than replaced.
    manifest_path = OUT / "refine_manifest.json"
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text()).get("clips", {})
        except (ValueError, OSError):
            prior = {}
        for cid, rec in prior.items():
            results.setdefault(cid, rec)

    manifest = {
        "_what": "Frozen CogVideoX-2b refinement of agent-specified relocation composites.",
        "_not": ["no LoRA adapter was loaded",
                 "no classifier-gradient guidance",
                 "no training",
                 "the pipeline enforces no trajectory or mask constraint, so preservation "
                 "of the relocation is measured afterwards, not guaranteed"],
        "pipeline_load_s": round(load_s, 1),
        "environment": mf.environment_snapshot() if hasattr(mf, "environment_snapshot") else None,
        "clips": results,
    }
    (OUT / "refine_manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"\nwrote {OUT / 'refine_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
