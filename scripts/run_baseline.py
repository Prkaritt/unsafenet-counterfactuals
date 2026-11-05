"""Run the feasibility stages non-interactively on a Bridges-2 GPU node.

This is the batch equivalent of ``notebooks/unsafenet_cogvideox_feasibility.ipynb``. The
notebook remains the record of the earlier free-Colab run; it is not runnable under Slurm
because it depends on ``/content`` paths and on ``google.colab``.

The same helpers in ``src/unsafenet_cf`` back both paths, so the preprocessing maths, the
manifest schema and the comparison metrics are shared rather than duplicated.

Stages
------
``probe``       Report the assigned GPU, driver, VRAM, RAM and resolved package versions.
                Allocates one small CUDA tensor and nothing else. No model is loaded.
``preprocess``  ffmpeg only. Crop, subsample and rescale the source clip to the
                checkpoint's 720x480x49 input shape. Runs on CPU.
``vae``         Stage 3. Load only ``AutoencoderKLCogVideoX``, encode and decode the
                preprocessed clip, measure fidelity. No diffusion.
``edit``        Stage 4. Prompt-guided source-video edit with
                ``CogVideoXVideoToVideoPipeline``, initialised from the real source
                latents. Writes the edited clip, comparison sheet and manifest. Repeat
                ``--sweep`` to generate several settings from one pipeline load; each
                one still gets its own clip, sheet and manifest.

Nothing here makes a safety claim. Outputs are candidate counterfactuals until a human
completes ``templates/REVIEW_TEMPLATE.md``.

Differences from the Colab run, all recorded in the manifest
------------------------------------------------------------
The Colab T4 had 15 GB of VRAM and only 12 GB of system RAM, which forced two-phase text
encoding and ``enable_model_cpu_offload()``. A Bridges-2 GPU node has at least 32 GB of
VRAM and 512 GB of RAM, so the whole 13.8 GB checkpoint fits resident on the device. The
offload path is kept behind ``--offload`` because it changes nothing about the output, only
the speed, and because it is the documented fallback if a smaller GPU is assigned.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsafenet_cf import compare as cmp  # noqa: E402
from unsafenet_cf import manifest as mf  # noqa: E402
from unsafenet_cf import preprocess as pp  # noqa: E402


# --------------------------------------------------------------------------- helpers


def log(msg: str) -> None:
    """Print with an immediate flush so a Slurm log stays readable while the job runs."""
    print(msg, flush=True)


def slurm_context() -> dict:
    """The Slurm facts that decide what the run cost and where it ran."""
    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_QOS",
        "CUDA_VISIBLE_DEVICES",
    )
    return {k: os.environ.get(k) for k in keys}


def nvidia_driver_version() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


def load_configs() -> tuple[dict, dict]:
    clip_cfg = json.loads((REPO / "configs" / "clip_0_tr143.json").read_text())
    exp_cfg = json.loads((REPO / "configs" / "experiment.json").read_text())
    return clip_cfg, exp_cfg


def resolve_dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


def frames_to_tensor(frames):
    """(T, H, W, 3) uint8 PIL frames -> (1, 3, T, H, W) float in [-1, 1]."""
    import numpy as np
    import torch

    arr = np.stack([np.asarray(f) for f in frames]).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)


def tensor_to_frames(x):
    """(1, 3, T, H, W) in [-1, 1] -> list of uint8 (H, W, 3) arrays."""
    import torch

    x = ((x.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu()
    return list(x[0].permute(1, 2, 3, 0).numpy())


def checkpoint_revision() -> str | None:
    """The commit SHA recorded by scripts/fetch_weights.py, if staging has finished."""
    path = REPO / "bridges2" / "checkpoint_revision.txt"
    if path.exists():
        return path.read_text().strip() or None
    return None


# The staged checkpoint is a plain directory, not a Hugging Face cache. The hub renamed
# THUDM/CogVideoX-2b to zai-org/CogVideoX-2b and snapshot_download stalled repeatedly on
# this cluster's login nodes, so scripts/fetch_weights.py downloads byte ranges with curl
# and verifies every file against bridges2/checkpoint_manifest.json. Loading by path means
# the job needs no network access and no cache semantics.
STAGED_CHECKPOINT = Path(
    os.environ.get("COGVIDEOX_CHECKPOINT",
                   "/ocean/projects/cis260288p/puprety/checkpoints/CogVideoX-2b")
)


def checkpoint_source(args, exp_cfg) -> tuple[str, str | None, str | None]:
    """Resolve where weights load from.

    Returns ``(name_or_path, revision_arg, recorded_revision)``.

    ``revision_arg`` is what ``from_pretrained`` receives. It is ``None`` for a local
    directory: a path has no revision to resolve, and passing one is meaningless there.
    ``recorded_revision`` is the commit SHA written into the manifest, which stays exact
    even when the weights come off disk.
    """
    manifest_path = REPO / "bridges2" / "checkpoint_manifest.json"
    staged_revision = None
    if manifest_path.exists():
        staged_revision = json.loads(manifest_path.read_text()).get("revision")

    local = Path(args.checkpoint) if args.checkpoint else STAGED_CHECKPOINT
    if (local / "model_index.json").exists():
        recorded = args.revision or checkpoint_revision() or staged_revision
        log(f"checkpoint  : {local} (staged directory)")
        log(f"revision    : {recorded} (recorded; not resolved over the network)")
        return str(local), None, recorded

    if args.checkpoint:
        # An explicit path that holds no checkpoint is a mistake to report, not a reason
        # to fall back to the network.
        log(f"--checkpoint {local} has no model_index.json")
        raise SystemExit(2)

    repo_id = exp_cfg["checkpoint"]
    revision = args.revision or checkpoint_revision()
    log(f"no staged checkpoint at {local}")
    log(f"checkpoint  : {repo_id} (hub id; needs HF_HUB_OFFLINE=0 or a populated cache)")
    return repo_id, revision, revision


# --------------------------------------------------------------------------- stages


def stage_probe(args) -> int:
    """Report the hardware and the environment before any weights are loaded."""
    log("=== slurm context ===")
    for k, v in slurm_context().items():
        log(f"{k:24s} {v}")

    log("\n=== nvidia-smi ===")
    smi = mf.nvidia_smi()
    if smi is None:
        log("nvidia-smi NOT FOUND. No GPU is attached to this allocation.")
        log("Check the --gres/--gpus request in the sbatch script.")
        return 2
    log(smi)
    log(f"\ndriver/name/memory: {nvidia_driver_version()}")

    log("\n=== environment ===")
    env = mf.environment_snapshot()
    log(json.dumps({k: v for k, v in env.items() if k != "nvidia_smi"}, indent=2))

    import torch

    log("\n=== torch build vs device ===")
    log(f"torch              {torch.__version__}")
    log(f"built against CUDA {torch.version.cuda}")
    log(f"compiled arch list {torch.cuda.get_arch_list()}")
    if not torch.cuda.is_available():
        log("torch.cuda.is_available() is False. The wheel and the driver do not match.")
        return 2

    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    log(f"device capability  {arch}")
    if arch not in torch.cuda.get_arch_list():
        log(f"WARNING: {arch} is absent from this wheel's arch list. Kernels will be")
        log("JIT-compiled from PTX at best, or fail outright. Install a matching build.")

    # The smallest possible real kernel launch: proves the driver, the wheel and the
    # device actually work together, without loading a model.
    t0 = time.time()
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = (a @ a).sum().item()
    torch.cuda.synchronize()
    log(f"fp16 matmul smoke test ok in {time.time() - t0:.2f}s (checksum finite: "
        f"{b == b})")
    log(f"bf16 supported     {torch.cuda.is_bf16_supported()}")

    out = REPO / "outputs" / "probe"
    out.mkdir(parents=True, exist_ok=True)
    mf.write_manifest(
        mf.build_run_manifest(
            stage="hardware_probe",
            environment=env,
            settings={"slurm": slurm_context(),
                      "torch_arch_list": list(torch.cuda.get_arch_list()),
                      "device_capability": arch,
                      "driver": nvidia_driver_version()},
            inputs={}, outputs={},
            timing={}, memory={},
            notes="Hardware probe only. No model weights were loaded.",
        ),
        out / f"probe_{os.environ.get('SLURM_JOB_ID', 'local')}.json",
    )
    log(f"\nwrote {out}/probe_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    return 0


def stage_preprocess(args) -> int:
    """ffmpeg crop, subsample and rescale to the checkpoint's input shape."""
    clip_cfg, _ = load_configs()
    raw = REPO / clip_cfg["source"]
    if not raw.exists():
        log(f"Source clip missing: {raw}")
        log("Run scripts/stage_inputs.sh on a login node first.")
        return 2

    digest = mf.sha256_file(raw)
    if digest != clip_cfg["source_sha256"]:
        log(f"SHA-256 mismatch for {raw}")
        log(f"  expected {clip_cfg['source_sha256']}")
        log(f"  got      {digest}")
        return 2
    log(f"{raw.name}: {raw.stat().st_size} bytes, sha256 verified")

    spec = pp.load_spec(REPO / "configs" / "clip_0_tr143.json")
    out_path = REPO / "outputs" / "preprocessed" / "0_tr143_720x480x49.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = pp.preprocess_clip(spec, out_path)

    keys = ("width", "height", "fps", "frames", "duration_s")
    log(f"source: {({k: record['source_probe'][k] for k in keys})}")
    log(f"output: {({k: record['output_probe'][k] for k in keys})}")
    log("latent shape (C,T,H,W): "
        f"{(pp.LATENT_CHANNELS, record['spec']['latent_frames'], record['spec']['latent_height'], record['spec']['latent_width'])}")
    (out_path.parent / "preprocess_record.json").write_text(json.dumps(record, indent=2))
    log(f"wrote {out_path}")
    return 0


def stage_vae(args) -> int:
    """Stage 3: VAE-only round trip. No diffusion, no editing."""
    import numpy as np
    import torch
    from diffusers import AutoencoderKLCogVideoX
    from diffusers.utils import export_to_video, load_video
    from PIL import Image

    _, exp_cfg = load_configs()
    dtype = resolve_dtype(args.dtype)
    pre_path = REPO / "outputs" / "preprocessed" / "0_tr143_720x480x49.mp4"
    if not pre_path.exists():
        log(f"Preprocessed clip missing: {pre_path}. Run --stage preprocess first.")
        return 2

    source_frames = load_video(str(pre_path))
    assert len(source_frames) == pp.SAMPLE_FRAMES, \
        f"expected {pp.SAMPLE_FRAMES} frames, got {len(source_frames)}"
    log(f"loaded {len(source_frames)} frames of size {source_frames[0].size}")

    mf.reset_peak_memory()
    t0 = time.time()
    ckpt, revision_arg, recorded_revision = checkpoint_source(args, exp_cfg)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        ckpt, subfolder="vae", torch_dtype=dtype, revision=revision_arg,
    ).to("cuda")
    vae.eval()
    if args.vae_tiling:
        vae.enable_tiling()
        vae.enable_slicing()
    load_s = time.time() - t0
    log(f"VAE loaded in {load_s:.1f}s; scaling_factor={vae.config.scaling_factor}; "
        f"tiling={args.vae_tiling}")

    t0 = time.time()
    with torch.no_grad():
        x = frames_to_tensor(source_frames).to("cuda", dtype=dtype)
        latent = vae.encode(x).latent_dist.mode()
        scaled = latent * vae.config.scaling_factor   # invert_scale_latents=false
        recon = vae.decode(scaled / vae.config.scaling_factor).sample
    roundtrip_s = time.time() - t0
    peak = mf.peak_memory_gb()

    log(f"input tensor  : {tuple(x.shape)}")
    log(f"latent tensor : {tuple(latent.shape)}  <- expect (1, 16, 13, 60, 90)")
    log(f"round trip    : {roundtrip_s:.1f}s, peak GPU {peak} GiB")

    recon_frames = tensor_to_frames(recon)
    src_arr = cmp.to_array(source_frames)
    rec_arr = np.stack(recon_frames)
    psnr_db = cmp.psnr(src_arr, rec_arr)
    mad = cmp.mean_abs_diff(src_arr, rec_arr)
    log(f"PSNR source vs reconstruction: {psnr_db:.2f} dB")
    log(f"mean |diff|                  : {mad:.2f} / 255")

    out = REPO / "outputs" / "stage3_vae"
    out.mkdir(parents=True, exist_ok=True)
    recon_path = out / "reconstruction.mp4"
    export_to_video([Image.fromarray(f) for f in recon_frames], str(recon_path),
                    fps=pp.TARGET_FPS)
    grid = cmp.comparison_grid(
        [("source", src_arr), ("vae_reconstruction", rec_arr)],
        out_path=out / "source_vs_reconstruction.png",
    )
    log(f"rows: {grid['order']} frames: {grid['frame_indices']}")

    mf.write_manifest(
        mf.build_run_manifest(
            stage="vae_reconstruction",
            environment={**mf.environment_snapshot(),
                         "slurm": slurm_context(),
                         "driver": nvidia_driver_version()},
            settings={"checkpoint": exp_cfg["checkpoint"], "subfolder": "vae",
                      "checkpoint_path": ckpt,
                      "revision": recorded_revision,
                      "dtype": str(dtype), "tiling": args.vae_tiling,
                      "slicing": args.vae_tiling,
                      "latent_mode": "latent_dist.mode()",
                      "scaling_factor": vae.config.scaling_factor},
            inputs={"preprocessed_clip": str(pre_path)},
            outputs={"reconstruction": str(recon_path),
                     "comparison_sheet": str(grid["path"]),
                     "psnr_db": psnr_db,
                     "mean_abs_diff": mad,
                     "per_frame_psnr_db": cmp.per_frame_psnr(src_arr, rec_arr)},
            timing={"vae_load_s": load_s, "roundtrip_s": roundtrip_s},
            memory={"peak_gpu_gib": peak},
            notes="VAE only. No diffusion, no editing, no safety claim.",
        ),
        out / "manifest.json",
    )
    log(f"wrote {out}/manifest.json")

    del vae, x, latent, scaled, recon
    mf.free_gpu()
    return 0


def vae_ceiling_psnr() -> float | None:
    """PSNR of the Stage 3 VAE round trip, if it has been measured.

    No edit can score higher against the source than the VAE round trip does, because
    every edited frame is decoded through the same VAE. Printing the ceiling next to the
    edit PSNR keeps that comparison honest: a 16 dB edit under a 32.6 dB ceiling has lost
    about 17 dB to the edit itself, not to the codec.
    """
    path = REPO / "outputs" / "stage3_vae" / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["outputs"]["psnr_db"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def denoising_steps(num_inference_steps: int, strength: float) -> int:
    """Number of denoising steps the video-to-video pipeline actually runs.

    ``CogVideoXVideoToVideoPipeline`` truncates the timestep schedule by ``strength``:
    it starts at ``num_inference_steps - int(num_inference_steps * strength)`` and runs
    to the end, which leaves ``int(num_inference_steps * strength)`` steps. Reporting a
    per-step time against the requested count instead of this one understates it. The
    earlier T4 progress bar showing 40 steps for a requested 50 at strength 0.8 is the
    same arithmetic, not a configuration mismatch.

    Verified against ``CogVideoXVideoToVideoPipeline.get_timesteps`` in diffusers 0.36.0,
    which computes ``min(int(num_inference_steps * strength), num_inference_steps)``.
    """
    return min(max(int(num_inference_steps * strength), 1), num_inference_steps)


def parse_sweep_spec(spec: str, exp_cfg: dict, args) -> dict:
    """Parse one ``--sweep`` value such as ``s=0.6,g=3.5`` into a settings dict.

    Keys: ``s``/``strength``, ``g``/``guidance``, ``n``/``steps``, ``seed``. Anything
    omitted falls back to the command line, then to configs/experiment.json.
    """
    aliases = {"s": "strength", "strength": "strength",
               "g": "guidance", "guidance": "guidance",
               "n": "steps", "steps": "steps",
               "seed": "seed"}
    settings = {
        "seed": args.seed if args.seed is not None else exp_cfg["seed"],
        "steps": args.steps if args.steps is not None else exp_cfg["num_inference_steps"],
        "strength": (args.strength if args.strength is not None
                     else exp_cfg["strength"]),
        "guidance": (args.guidance if args.guidance is not None
                     else exp_cfg["guidance_scale"]),
    }
    for field in spec.split(","):
        field = field.strip()
        if not field:
            continue
        if "=" not in field:
            raise ValueError(f"sweep field {field!r} is not key=value")
        key, _, value = field.partition("=")
        key = aliases.get(key.strip().lower())
        if key is None:
            raise ValueError(f"unknown sweep key in {field!r}; "
                             "use s/strength, g/guidance, n/steps or seed")
        settings[key] = int(value) if key in ("steps", "seed") else float(value)
    return settings


def run_specs(args, exp_cfg) -> list[dict]:
    """Resolve the list of generation settings for this invocation."""
    if args.sweep:
        return [parse_sweep_spec(spec, exp_cfg, args) for spec in args.sweep]
    return [parse_sweep_spec("", exp_cfg, args)]


def stage_edit(args) -> int:
    """Stage 4: prompt-guided edit initialised from the real source latents.

    The pipeline is loaded once and reused for every setting in ``--sweep``, because the
    load costs about 200 s against about 90 s per generation. Each setting still gets its
    own clip, comparison sheet and manifest, so no run is traceable only through another.
    """
    import numpy as np
    import torch
    from diffusers import CogVideoXVideoToVideoPipeline
    from diffusers.utils import export_to_video, load_video

    clip_cfg, exp_cfg = load_configs()
    dtype = resolve_dtype(args.dtype)
    pre_path = REPO / "outputs" / "preprocessed" / "0_tr143_720x480x49.mp4"
    if not pre_path.exists():
        log(f"Preprocessed clip missing: {pre_path}. Run --stage preprocess first.")
        return 2

    try:
        specs = run_specs(args, exp_cfg)
    except ValueError as exc:
        log(f"bad --sweep value: {exc}")
        return 2

    source_frames = load_video(str(pre_path))
    src_arr = cmp.to_array(source_frames)

    free0, total = torch.cuda.mem_get_info()
    log(f"free VRAM before load: {free0 / 1024**3:.2f} GiB of {total / 1024**3:.2f} GiB")

    mf.reset_peak_memory()
    t0 = time.time()
    ckpt, revision_arg, recorded_revision = checkpoint_source(args, exp_cfg)
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
        ckpt, torch_dtype=dtype, revision=revision_arg,
    )
    # On a 32 GB or larger card the whole 13.8 GB checkpoint is resident, which is both
    # faster and simpler than the offload path the 12 GB-RAM Colab runtime required.
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    if args.vae_tiling:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    load_s = time.time() - t0
    log(f"pipeline loaded in {load_s:.1f}s, offload={args.offload}, "
        f"vae_tiling={args.vae_tiling}")
    log(f"scheduler : {type(pipe.scheduler).__name__}")
    log(f"prediction: {pipe.scheduler.config.prediction_type}")
    log(f"prompt  : {exp_cfg['prompt']}")
    log(f"runs    : {len(specs)}")

    out = REPO / "outputs" / "stage4_edit"
    out.mkdir(parents=True, exist_ok=True)
    recon_mp4 = REPO / "outputs" / "stage3_vae" / "reconstruction.mp4"
    recon_arr = cmp.to_array(load_video(str(recon_mp4))) if recon_mp4.exists() else None
    ceiling = vae_ceiling_psnr()

    summary = []
    for index, spec in enumerate(specs, start=1):
        seed = spec["seed"]
        steps = spec["steps"]
        strength = spec["strength"]
        guidance = spec["guidance"]
        executed = denoising_steps(steps, strength)
        log(f"\n--- run {index}/{len(specs)}: seed={seed} steps={steps} "
            f"strength={strength} guidance={guidance} "
            f"(denoising steps: {executed})")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        mf.reset_peak_memory()
        t0 = time.time()
        result = pipe(
            video=source_frames,
            prompt=exp_cfg["prompt"],
            negative_prompt=exp_cfg["negative_prompt"],
            height=pp.TARGET_HEIGHT,
            width=pp.TARGET_WIDTH,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=guidance,
            generator=generator,
        )
        gen_s = time.time() - t0
        peak = mf.peak_memory_gb()

        edit_frames = result.frames[0]
        log(f"generated {len(edit_frames)} frames in {gen_s:.1f}s "
            f"({gen_s / executed:.2f}s per denoising step), peak GPU {peak} GiB")

        tag = f"seed{seed}_s{strength}_g{guidance}_n{steps}"
        edit_path = out / f"edit_{tag}.mp4"
        export_to_video(edit_frames, str(edit_path), fps=pp.TARGET_FPS)
        log(f"saved {edit_path}")

        edit_arr = cmp.to_array(edit_frames)
        rows = [("source", src_arr)]
        if recon_arr is not None:
            rows.append(("vae_reconstruction", recon_arr))
        rows.append(("edited", edit_arr))
        grid = cmp.comparison_grid(rows, out_path=out / f"comparison_{tag}.png")
        log(f"rows: {grid['order']} frames: {grid['frame_indices']}")

        psnr_db = cmp.psnr(src_arr, edit_arr)
        mad = cmp.mean_abs_diff(src_arr, edit_arr)
        log(f"PSNR source vs edit: {psnr_db:.2f} dB, mean |diff| {mad:.2f} / 255")

        mf.write_manifest(
            mf.build_run_manifest(
                stage="prompt_guided_edit",
                environment={**mf.environment_snapshot(),
                             "slurm": slurm_context(),
                             "driver": nvidia_driver_version()},
                settings={
                    "checkpoint": exp_cfg["checkpoint"],
                    "checkpoint_path": ckpt,
                    "revision": recorded_revision,
                    "dtype": str(dtype),
                    "scheduler": type(pipe.scheduler).__name__,
                    "prediction_type": pipe.scheduler.config.prediction_type,
                    "seed": seed,
                    "generator_device": "cpu",
                    "num_inference_steps": steps,
                    # The pipeline truncates the schedule by strength; see
                    # denoising_steps(). Both counts are recorded so a per-step time
                    # read off this manifest is unambiguous.
                    "denoising_steps_executed": executed,
                    "strength": strength,
                    "guidance_scale": guidance,
                    "prompt": exp_cfg["prompt"],
                    "negative_prompt": exp_cfg["negative_prompt"],
                    "num_frames": len(source_frames),
                    "height": pp.TARGET_HEIGHT,
                    "width": pp.TARGET_WIDTH,
                    "offload": args.offload,
                    "vae_tiling": args.vae_tiling,
                    "max_sequence_length": 226,
                },
                inputs={
                    "preprocessed_clip": str(pre_path),
                    "source_sha256": clip_cfg["source_sha256"],
                    "unsafe_label": clip_cfg["unsafe_label"],
                    "target_safe_label": clip_cfg["target_safe_label"],
                    "context_that_must_not_change":
                        exp_cfg["context_that_must_not_change"],
                    "spurious_change_to_reject":
                        exp_cfg["spurious_change_to_reject"],
                },
                outputs={
                    "edited_clip": str(edit_path),
                    "comparison_sheet": str(grid["path"]),
                    "psnr_source_vs_edit_db": psnr_db,
                    "mean_abs_diff_source_vs_edit": mad,
                    "vae_ceiling_psnr_db": ceiling,
                },
                timing={"pipeline_load_s": load_s, "generation_s": gen_s,
                        "s_per_denoising_step": gen_s / executed},
                memory={"peak_gpu_gib": peak},
                notes=("Prompt-guided editing baseline using a pretrained CogVideoX-2b. "
                       "Not LD-ViCE: no per-dataset fine-tuning, no classifier gradients, "
                       "no SmoothGrad, no artifact refinement. Output is a candidate "
                       "counterfactual pending human review. generation_s covers the whole "
                       "pipeline call, which includes VAE encode, text encode and VAE "
                       "decode as well as the denoising loop."),
            ),
            out / f"manifest_{tag}.json",
        )
        log(f"wrote {out}/manifest_{tag}.json")
        summary.append({"tag": tag, "strength": strength, "guidance": guidance,
                        "steps": steps, "denoising_steps": executed,
                        "psnr_db": psnr_db, "mean_abs_diff": mad,
                        "generation_s": gen_s, "peak_gpu_gib": peak})

        del result, edit_frames, edit_arr
        mf.free_gpu()

    ceiling_note = (f"VAE round-trip ceiling: {ceiling:.2f} dB" if ceiling is not None
                    else "VAE round-trip ceiling: not measured")
    log(f"\n{'=' * 72}\n== sweep summary ({ceiling_note})\n{'=' * 72}")
    log(f"{'strength':>9} {'guidance':>9} {'steps':>6} {'denoise':>8} "
        f"{'PSNR dB':>8} {'mean|d|':>8} {'gen s':>7}")
    for row in summary:
        log(f"{row['strength']:>9.2f} {row['guidance']:>9.2f} {row['steps']:>6d} "
            f"{row['denoising_steps']:>8d} {row['psnr_db']:>8.2f} "
            f"{row['mean_abs_diff']:>8.2f} {row['generation_s']:>7.1f}")
    log("\nHigher PSNR means more of the source survived. It does NOT mean the intended "
        "behaviour changed. Both questions are decided by a human against "
        "templates/REVIEW_TEMPLATE.md, not by this table.")

    summary_path = out / "sweep_summary.json"
    summary_path.write_text(json.dumps(
        {"vae_ceiling_psnr_db": ceiling, "runs": summary}, indent=2))
    log(f"wrote {summary_path}")
    return 0


STAGES = {
    "probe": stage_probe,
    "preprocess": stage_preprocess,
    "vae": stage_vae,
    "edit": stage_edit,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=[*STAGES, "all"],
                    help="'all' runs preprocess, vae and edit in that order")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="CogVideoX-2b is documented with float16 (default)")
    ap.add_argument("--offload", default="none",
                    choices=["none", "model", "sequential"],
                    help="'none' keeps the checkpoint resident; the others are the "
                         "documented fallbacks for a smaller card")
    ap.add_argument("--vae-tiling", dest="vae_tiling", action="store_true", default=True)
    ap.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false")
    ap.add_argument("--checkpoint", default=None,
                    help="path to a staged checkpoint directory; defaults to "
                         "$COGVIDEOX_CHECKPOINT, then the Ocean staging path, then the "
                         "hub id in configs/experiment.json")
    ap.add_argument("--revision", default=None,
                    help="checkpoint commit SHA; defaults to the staged revision")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--guidance", type=float, default=None,
                    help="classifier-free guidance scale; defaults to the value in "
                         "configs/experiment.json")
    ap.add_argument("--sweep", action="append", default=None, metavar="s=0.6,g=3.5",
                    help="one generation setting, repeatable. Keys: s/strength, "
                         "g/guidance, n/steps, seed. Omitted keys fall back to the "
                         "other flags, then to configs/experiment.json. The pipeline "
                         "is loaded once and reused for every --sweep value.")
    args = ap.parse_args(argv)

    if args.revision is None:
        args.revision = checkpoint_revision()

    log(f"repo        : {REPO}")
    log(f"HF_HOME     : {os.environ.get('HF_HOME')}")
    log(f"HF offline  : {os.environ.get('HF_HUB_OFFLINE')}")
    log(f"stage       : {args.stage}")

    order = ["preprocess", "vae", "edit"] if args.stage == "all" else [args.stage]
    for name in order:
        log(f"\n{'=' * 72}\n== stage: {name}\n{'=' * 72}")
        rc = STAGES[name](args)
        if rc != 0:
            log(f"stage {name} failed with code {rc}; stopping.")
            return rc
    log("\nall requested stages completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
