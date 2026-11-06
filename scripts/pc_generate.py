#!/usr/bin/env python
"""Generate the GPU comparison arms for every panel-cover case, in one job.

Four arms per case, all from the same preprocessed source and the same frozen settings:

``vae_source``   The preprocessed source through the CogVideoX VAE and back, with no
                 diffusion. This is the ceiling on every later measurement: nothing that
                 passes through the latent space can be sharper than this.
``vae_composite``The composite through the same round trip, so the composite and the
                 refined arm are compared on equal terms rather than across a VAE boundary.
``base_edit``    The source edited by prompt alone, with no composite and no mask. This is
                 the control that shows what asking a model to close a panel actually does.
``refined``      The composite through a low-strength video-to-video pass, which is asked
                 only to make an already-closed panel look like footage.

Optionally a fifth, ``refined_restricted``, re-imposes the panel mask on the refined output
so that pixels outside the region return to the composite exactly. **That is output
compositing, not mask-conditioned generation**: the model never saw a mask and was never
constrained during denoising. The unrestricted output is always kept so the two can be
compared, because the restricted one hides whatever the model did elsewhere.

The walkway LoRA is never loaded. It was trained to move a person onto green paint, which
has nothing to do with a panel cover, and the transformer is inspected for LoRA parameter
tensors before and after every generation.
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

from unsafenet_cf import compare as cmp       # noqa: E402
from unsafenet_cf import manifest as mf       # noqa: E402
from unsafenet_cf import manualpath as mp     # noqa: E402
from unsafenet_cf import panelpatch as pp     # noqa: E402
from unsafenet_cf import preprocess as prep   # noqa: E402

OUT = REPO / "outputs" / "panel_cover"
ARMS = ("vae_source", "vae_composite", "base_edit", "refined", "refined_restricted")


def log(m: str) -> None:
    print(m, flush=True)


def save_video(frames, path: Path) -> None:
    """Write frames as video. PIL images or uint8 arrays both accepted.

    diffusers rescales numpy frames by 255, which wraps uint8 modulo 256 and writes an
    exact colour negative, so arrays are converted to PIL first. See DECISIONS.md D7.
    """
    from PIL import Image
    from diffusers.utils import export_to_video
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [f if hasattr(f, "save") else Image.fromarray(np.ascontiguousarray(f))
           for f in frames]
    export_to_video(out, str(path), fps=prep.TARGET_FPS)


def frames_to_tensor(frames):
    import torch
    arr = np.stack([np.asarray(f) for f in frames]).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)


def tensor_to_frames(x):
    arr = x[0].permute(1, 2, 3, 0).float().cpu().numpy()
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


def main() -> int:
    import torch
    from diffusers import AutoencoderKLCogVideoX, CogVideoXVideoToVideoPipeline
    from diffusers.utils import load_video

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/panel_cover_plan.json")
    ap.add_argument("--settings", default="configs/panel_cover_generate.json")
    ap.add_argument("--cases", default="", help="comma-separated subset of case ids")
    ap.add_argument("--arms", default="vae_source,vae_composite,base_edit,refined,"
                                      "refined_restricted")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    plan = json.loads((REPO / args.plan).read_text())
    cfg = json.loads((REPO / args.settings).read_text())
    only = {c for c in args.cases.split(",") if c}
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; known arms are {list(ARMS)}")

    ckpt = args.checkpoint or os.environ.get(
        "COGVIDEOX_CHECKPOINT",
        "/ocean/projects/cis260288p/puprety/checkpoints/CogVideoX-2b")
    revision = (REPO / "bridges2" / "checkpoint_revision.txt").read_text().strip()
    dtype = torch.float16

    cases = [c for c in plan["cases"] if not only or c["case_id"] in only]
    log(f"checkpoint : {ckpt} (revision {revision})")
    log(f"cases      : {[c['case_id'] for c in cases]}")
    log(f"arms       : {arms}")
    log("LoRA adapter: NEVER LOADED. classifier-gradient guidance: not used in this stage")

    results: dict[str, dict] = {}

    # --- VAE-only arms, with the VAE alone in memory ------------------------------
    if {"vae_source", "vae_composite"} & set(arms):
        t0 = time.time()
        vae = AutoencoderKLCogVideoX.from_pretrained(
            ckpt, subfolder="vae", torch_dtype=dtype).to("cuda")
        vae.eval()
        vae.enable_tiling()
        vae.enable_slicing()
        log(f"VAE loaded in {time.time() - t0:.1f}s, "
            f"scaling_factor={vae.config.scaling_factor}")
        for case in cases:
            cid = case["case_id"]
            rec = results.setdefault(cid, {"case_id": cid, "arms": {}})
            for arm, stem in (("vae_source", "source"), ("vae_composite", "composite")):
                if arm not in arms:
                    continue
                src_path = OUT / cid / "videos" / f"{cid}_{stem}.mp4"
                if not src_path.exists():
                    rec["arms"][arm] = {"error": f"missing input {src_path.name}"}
                    log(f"  {cid}/{arm}: no input at {src_path}; skipped")
                    continue
                frames = load_video(str(src_path))
                mf.reset_peak_memory()
                t = time.time()
                with torch.no_grad():
                    x = frames_to_tensor(frames).to("cuda", dtype=dtype)
                    lat = vae.encode(x).latent_dist.mode()
                    # invert_scale_latents=false for this checkpoint, so the latent is
                    # scaled on the way out and unscaled on the way back in. Written out
                    # rather than cancelled, to match scripts/run_baseline.py exactly.
                    scaled = lat * vae.config.scaling_factor
                    out = vae.decode(scaled / vae.config.scaling_factor).sample
                secs = time.time() - t
                arr = tensor_to_frames(out)
                out_path = OUT / cid / "videos" / f"{cid}_{arm}.mp4"
                save_video(list(arr), out_path)
                ref = cmp.to_array(frames)
                rec["arms"][arm] = {
                    "output": str(out_path.relative_to(REPO)),
                    "input": str(src_path.relative_to(REPO)),
                    "latent_shape": list(lat.shape),
                    "scaling_factor": float(vae.config.scaling_factor),
                    "latent_mode": "latent_dist.mode()",
                    "seconds": round(secs, 1),
                    "peak_gpu_gib": mf.peak_memory_gb(),
                    "psnr_vs_input_db": cmp.psnr(ref, arr),
                    "mean_abs_diff_vs_input": cmp.mean_abs_diff(ref, arr),
                    "_what": "VAE encode and decode only. No diffusion, no prompt.",
                }
                log(f"  {cid}/{arm}: {secs:.1f}s, PSNR "
                    f"{rec['arms'][arm]['psnr_vs_input_db']:.2f} dB, peak "
                    f"{rec['arms'][arm]['peak_gpu_gib']:.2f} GiB")
        del vae
        mf.free_gpu()

    # --- diffusion arms -----------------------------------------------------------
    if {"base_edit", "refined", "refined_restricted"} & set(arms):
        t0 = time.time()
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(ckpt, torch_dtype=dtype)
        pipe.to("cuda")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        for m in (pipe.transformer, pipe.vae, pipe.text_encoder):
            for p in m.parameters():
                p.requires_grad_(False)
        load_s = time.time() - t0
        log(f"pipeline loaded in {load_s:.1f}s; "
            f"scheduler {type(pipe.scheduler).__name__}; dtype {pipe.transformer.dtype}")

        def n_lora() -> int:
            return sum(1 for n, _ in pipe.transformer.named_parameters() if "lora_" in n)

        if n_lora() != 0:
            raise SystemExit("transformer already carries LoRA parameters; aborting")

        for case in cases:
            cid = case["case_id"]
            rec = results.setdefault(cid, {"case_id": cid, "arms": {}})
            for arm in ("base_edit", "refined"):
                if arm not in arms:
                    continue
                st = cfg["settings"][arm]
                stem = "source" if arm == "base_edit" else "composite"
                in_path = OUT / cid / "videos" / f"{cid}_{stem}.mp4"
                if not in_path.exists():
                    rec["arms"][arm] = {"error": f"missing input {in_path.name}"}
                    log(f"  {cid}/{arm}: no input at {in_path}; skipped")
                    continue
                frames_in = load_video(str(in_path))
                prompt = cfg["prompts"][arm][cid]
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
                out_path = OUT / cid / "videos" / f"{cid}_{arm}.mp4"
                save_video(frames, out_path)

                arr = cmp.to_array(frames)
                in_arr = cmp.to_array(frames_in)
                executed = min(int(st["num_inference_steps"] * st["strength"]),
                               st["num_inference_steps"])
                entry = {
                    "output": str(out_path.relative_to(REPO)),
                    "input": str(in_path.relative_to(REPO)),
                    "prompt": prompt,
                    "negative_prompt": st["negative_prompt"],
                    "seed": st["seed"],
                    "strength": st["strength"],
                    "num_inference_steps": st["num_inference_steps"],
                    "denoising_steps_executed": executed,
                    "_steps_note": ("strength truncates the schedule: executed = "
                                    "int(steps * strength). Timing is per executed step."),
                    "guidance_scale": st["guidance_scale"],
                    "scheduler": type(pipe.scheduler).__name__,
                    "dtype": str(pipe.transformer.dtype),
                    "checkpoint": ckpt,
                    "checkpoint_revision": revision,
                    "seconds": round(secs, 1),
                    "s_per_denoising_step": round(secs / executed, 3) if executed else None,
                    "peak_gpu_gib": mf.peak_memory_gb(),
                    "lora_param_tensors_present": n_lora(),
                    "psnr_vs_input_db": cmp.psnr(in_arr, arr),
                    "mean_abs_diff_vs_input": cmp.mean_abs_diff(in_arr, arr),
                    "_psnr_warning": ("Whole-frame PSNR is dominated by static background "
                                      "and is nearly blind to a panel occupying a small "
                                      "part of the frame. It is not evidence about the "
                                      "panel."),
                }
                if entry["lora_param_tensors_present"] != 0:
                    raise SystemExit("adapter appeared during generation; aborting")
                rec["arms"][arm] = entry
                log(f"  {cid}/{arm}: {secs:.1f}s ({executed} executed steps), peak "
                    f"{entry['peak_gpu_gib']:.2f} GiB, PSNR vs input "
                    f"{entry['psnr_vs_input_db']:.2f} dB")

                # Output compositing, kept strictly separate from the arm above.
                if arm == "refined" and "refined_restricted" in arms:
                    comp_rec_path = OUT / cid / "composite.json"
                    if not comp_rec_path.exists():
                        rec["arms"]["refined_restricted"] = {
                            "error": "composite.json missing, so no region polygon"}
                        continue
                    poly = np.asarray(json.loads(comp_rec_path.read_text())
                                      ["region"]["polygon"], dtype=float)
                    base = cmp.to_array(frames_in)
                    mask = mp.polygon_mask(poly, base.shape[1:3])
                    alpha = pp.feather(mask, int(case.get("feather_px", 2)))
                    restricted = np.stack([pp.composite(base[i], arr[i], alpha)
                                           for i in range(len(arr))])
                    rp = OUT / cid / "videos" / f"{cid}_refined_restricted.mp4"
                    save_video(list(restricted), rp)
                    rec["arms"]["refined_restricted"] = {
                        "output": str(rp.relative_to(REPO)),
                        "built_from": ["composite (outside the region)",
                                       "refined (inside the region)"],
                        "region_px": int(mask.sum()),
                        "_what": ("Output compositing applied AFTER generation. The model "
                                  "was not conditioned on the mask and was not constrained "
                                  "during denoising. The unrestricted refined output is "
                                  "kept at "
                                  f"{(OUT / cid / 'videos' / f'{cid}_refined.mp4').name} "
                                  "for comparison."),
                        "psnr_vs_composite_db": cmp.psnr(base, restricted),
                    }
                    log(f"  {cid}/refined_restricted: wrote {rp.name}")

    manifest_path = OUT / "generate_manifest.json"
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text()).get("cases", {})
        except (ValueError, OSError):
            prior = {}
        for cid, old in prior.items():
            if cid not in results:
                results[cid] = old
            else:
                for arm, entry in old.get("arms", {}).items():
                    results[cid]["arms"].setdefault(arm, entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "_what": "CogVideoX arms for the panel-cover cases: VAE round trips, a prompt-only "
                 "control, and a low-strength refinement of the composite.",
        "_not": ["no LoRA adapter was loaded",
                 "no classifier-gradient guidance in this stage",
                 "no training",
                 "no mask conditioning: the pipeline enforces no mask or region constraint"],
        "generated": mf.utc_now(),
        "environment": mf.environment_snapshot(),
        "cases": results,
    }, indent=1))
    log(f"\nwrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
