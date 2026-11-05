#!/usr/bin/env python
"""Matched base-versus-LoRA video-to-video edits of the counterfactual source clip.

The two runs differ in exactly one thing: whether the trained LoRA adapter is
active on the transformer. Source clip, seed, scheduler, dtype, prompt, negative
prompt, strength, text guidance scale, requested steps, resolution and frame
count are identical, and both runs go through the same stock
``CogVideoXVideoToVideoPipeline.__call__``.

**Classifier guidance is disabled in both runs**, by construction: this script
never imports the guidance module and never computes a classifier gradient. The
classifier is loaded only to *report* probabilities afterwards, and those numbers
are informational. Gate C failed -- the classifier showed no measurable
dependence on the worker -- so a probability change here is not evidence about
the worker's position and is not a label flip claim.

Outputs are candidate counterfactuals until a human completes
``templates/REVIEW_TEMPLATE.md``.
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

from unsafenet_cf import classifier as C  # noqa: E402
from unsafenet_cf import compare as cmp  # noqa: E402
from unsafenet_cf import manifest as mf  # noqa: E402
from unsafenet_cf import preprocess as pp  # noqa: E402

OUT = REPO / "outputs" / "lora_edit"


def log(msg: str) -> None:
    print(msg, flush=True)


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

    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", default="outputs/lora/adapter")
    ap.add_argument("--lora-weight-name", default="pytorch_lora_weights.safetensors")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--lora-alpha", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--classifier", default="outputs/classifier/classifier_best.pt")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((REPO / "configs" / "experiment.json").read_text())
    ckpt_dir = args.checkpoint or __import__("os").environ.get(
        "COGVIDEOX_CHECKPOINT", "/ocean/projects/cis260288p/puprety/checkpoints/CogVideoX-2b"
    )
    revision = (REPO / "bridges2" / "checkpoint_revision.txt").read_text().strip()
    device = "cuda"

    lora_dir = (REPO / args.lora_dir) if not Path(args.lora_dir).is_absolute() else Path(args.lora_dir)
    lora_file = lora_dir / args.lora_weight_name
    if not lora_file.exists():
        raise SystemExit(f"LoRA weights not found: {lora_file}")

    log(f"checkpoint : {ckpt_dir} (revision {revision})")
    log(f"lora       : {lora_file}")
    log(f"settings   : seed={args.seed} steps={args.steps} strength={args.strength} "
        f"text_guidance={args.guidance} classifier_guidance=DISABLED")

    t0 = time.time()
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(ckpt_dir, torch_dtype=torch.float16)
    pipe.to(device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
        for p in module.parameters():
            p.requires_grad_(False)
    log(f"pipeline loaded in {time.time() - t0:.1f}s")

    pre_path = REPO / "outputs" / "preprocessed" / "0_tr143_720x480x49.mp4"
    source_frames = load_video(str(pre_path))
    src_arr = cmp.to_array(source_frames)
    log(f"source: {pre_path.name} {src_arr.shape}")

    def generate(tag: str) -> dict:
        mf.reset_peak_memory()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        t = time.time()
        result = pipe(
            video=source_frames,
            prompt=cfg["prompt"],
            negative_prompt=cfg["negative_prompt"],
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            num_videos_per_prompt=1,
            generator=gen,
        )
        secs = time.time() - t
        frames = result.frames[0]
        path = OUT / f"{tag}_seed{args.seed}_s{args.strength}_g{args.guidance}.mp4"
        export_to_video(frames, str(path), fps=pp.TARGET_FPS)
        arr = cmp.to_array(frames)
        rec = {
            "tag": tag,
            "path": str(path),
            "generation_s": secs,
            "peak_gpu_gib": mf.peak_memory_gb(),
            "psnr_vs_source_db": cmp.psnr(src_arr, arr),
            "mean_abs_diff_vs_source": cmp.mean_abs_diff(src_arr, arr),
            "per_frame_psnr_db": cmp.per_frame_psnr(src_arr, arr),
        }
        log(f"  {tag}: {secs:.1f}s, peak {rec['peak_gpu_gib']:.3f} GiB, "
            f"PSNR {rec['psnr_vs_source_db']:.3f} dB, MAD {rec['mean_abs_diff_vs_source']:.4f}")
        return rec, arr

    log("\n--- base (no adapter) ---")
    base_rec, base_arr = generate("base")

    log("\n--- loading LoRA adapter ---")
    pipe.load_lora_weights(str(lora_dir), weight_name=args.lora_weight_name,
                           adapter_name="cogvideox-lora")
    scale = args.lora_alpha / args.rank
    pipe.set_adapters(["cogvideox-lora"], [scale])
    # Check that the adapter really is attached. A silently inactive adapter would make
    # the "lora" run identical to the "base" run and the comparison meaningless, so this
    # is verified rather than assumed. Where the pipeline exposes no way to ask, say so
    # instead of inventing a confirmation.
    if hasattr(pipe, "get_active_adapters"):
        active = pipe.get_active_adapters()
        log(f"adapter active: {active}, scale {scale} "
            f"(= lora_alpha {args.lora_alpha} / rank {args.rank})")
        if not active:
            raise SystemExit("adapter did not activate; refusing to report a LoRA result")
        adapter_check = {"method": "get_active_adapters", "active": list(active)}
    else:
        n_lora = sum(1 for n, _ in pipe.transformer.named_parameters() if "lora_" in n)
        log(f"get_active_adapters unavailable; {n_lora} LoRA parameters found on the "
            f"transformer; scale {scale}")
        if n_lora == 0:
            raise SystemExit("no LoRA parameters on the transformer after loading; "
                             "refusing to report a LoRA result")
        adapter_check = {"method": "named_parameters", "lora_param_tensors": n_lora}

    log("\n--- lora ---")
    lora_rec, lora_arr = generate("lora")

    # Informational only. See the module docstring and Gate C.
    probs = {}
    clf_path = REPO / args.classifier
    if clf_path.exists():
        ck = torch.load(clf_path, map_location="cpu", weights_only=False)
        prep = (ck["preprocess"]["frames"], ck["preprocess"]["height"], ck["preprocess"]["width"])
        model = C.build_model(pretrained=False).to(device).eval()
        model.load_state_dict(ck["state_dict"])
        probs["source"] = classify_clip(model, src_arr, prep, device)
        recon = REPO / "outputs" / "stage3_vae" / "reconstruction.mp4"
        if recon.exists():
            probs["vae_reconstruction"] = classify_clip(
                model, cmp.to_array(load_video(str(recon))), prep, device)
        probs["base"] = classify_clip(model, base_arr, prep, device)
        probs["lora"] = classify_clip(model, lora_arr, prep, device)
        probs["_label_map"] = ck.get("label_map")
        probs["_caveat"] = (
            "Informational only. Gate C FAILED: the classifier showed no measurable "
            "dependence on the worker, so these probabilities are not evidence about the "
            "worker's position and no label flip is claimed. The VAE round trip alone "
            "already moves the source across the decision boundary."
        )
        for k in ("source", "vae_reconstruction", "base", "lora"):
            if k in probs:
                log(f"  p(class4) {k:20s} = {probs[k][1]:.4f}")
    else:
        probs["_missing"] = f"{clf_path} not found; no probabilities reported"

    sheet = cmp.comparison_grid(
        [("source", source_frames),
         ("base", [f for f in base_arr]),
         ("lora", [f for f in lora_arr])],
        indices=cmp.pick_indices(len(source_frames), 6),
        out_path=OUT / "comparison_base_vs_lora.png",
    )
    sheet.pop("image")

    manifest = {
        "_what_this_is": (
            "Matched base-versus-LoRA prompt-guided video-to-video edit. Not LD-ViCE: "
            "LD-ViCE fine-tunes CogVideoX per dataset AND uses target-model gradients; "
            "this run fine-tunes a LoRA adapter and uses NO classifier gradient. "
            "Candidate counterfactuals until human review."
        ),
        "classifier_guidance": "disabled in both runs",
        "checkpoint": cfg["checkpoint"],
        "checkpoint_local_dir": str(ckpt_dir),
        "revision": revision,
        "lora": {
            "dir": str(lora_dir),
            "weight_name": args.lora_weight_name,
            "sha256": mf.sha256_file(lora_file),
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "scale_applied": scale,
            "adapter_verified": adapter_check,
            "training_manifest": "outputs/lora/train_manifest.json",
        },
        "settings": {
            "seed": args.seed,
            "num_inference_steps": args.steps,
            "strength": args.strength,
            "guidance_scale": args.guidance,
            "scheduler": type(pipe.scheduler).__name__,
            "dtype": "torch.float16",
            "width": pp.TARGET_WIDTH,
            "height": pp.TARGET_HEIGHT,
            "num_frames": len(source_frames),
            "fps": pp.TARGET_FPS,
            "prompt": cfg["prompt"],
            "negative_prompt": cfg["negative_prompt"],
        },
        "source": {
            "path": str(pre_path),
            "sha256": mf.sha256_file(pre_path),
        },
        "runs": {"base": base_rec, "lora": lora_rec},
        "classifier_probabilities": probs,
        "comparison_sheet": sheet,
        "intended_change": cfg["intended_change"],
        "spurious_change_to_reject": cfg["spurious_change_to_reject"],
        "context_that_must_not_change": cfg["context_that_must_not_change"],
        "environment": mf.environment_snapshot(),
    }
    mf.write_manifest(manifest, OUT / "manifest_base_vs_lora.json")
    log(f"\nwrote {OUT}/manifest_base_vs_lora.json")
    log("Both clips are CANDIDATE counterfactuals. Human review is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
