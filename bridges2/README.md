# Running on PSC Bridges-2

This replaces the free-Colab route for the GPU stages. By user decision it **overrides** the
Colab-only requirement in `CLAUDE.md`. Nothing about the experiment's scope changes: one
clip, one prompt-guided editing baseline, no LD-ViCE claim, no safety claim.

Nothing here has executed on a Bridges-2 GPU yet. The hardware probe is the first job.

---

## Allocation

```
Account        cis260288p
GPU balance    223 of 300 SU remaining, expires 2027-09-08
Ocean storage  187 GB remaining of 400 GB, expires 2027-09-08
```

Read it again with `projects`.

## What the cluster actually offers

Read from `sinfo` and `scontrol` on 2026-11-03, not from documentation:

| Partition | GPU types available | Notes |
|---|---|---|
| `GPU` | `v100-16`, `v100-32`, `l40s-48`, `h100-80` | whole nodes, 8 GPUs charged whether used or not |
| `GPU-shared` | `v100-16` (72), `v100-32` (200), `l40s-48` (24) | charged per GPU allocated, max 4 GPUs, `DefMemPerGPU=63000` MB |
| `GPU-small` | `v100-32` | one node, `v001` |

`h100-80` nodes appear in the `GPU-shared` node list but **not** in its GPU TRES counts, so
do not assume an H100 is reachable from `GPU-shared` with this allocation.

## Charge rates

From the Bridges-2 user guide: SUs are charged per GPU-hour, and `GPU-shared` bills only the
GPUs allocated to the job, not the whole node.

| GPU | SU per GPU-hour |
|---|---|
| `v100-16`, `v100-32` | 1 |
| `l40s-48` | 1 |
| `h100-80` | 2 |

Slurm accounts on elapsed time, so the figures below are worst cases at the requested wall
limit; a job that finishes early costs proportionally less. Confirm each job's real charge
afterwards with `sacct -j <jobid>` and `projects`.

---

## Why L40S-48

An `l40s-48` costs the same 1 SU per GPU-hour as a V100 and is the better card for this
work:

- **48 GB of VRAM.** The whole 13.8 GB CogVideoX-2b checkpoint stays resident. The Colab
  run's two-phase text encoding and `enable_model_cpu_offload()` existed only because that
  runtime had 12 GiB of system RAM. Dropping both removes the slowest part of the T4 run.
- **`sm_89`.** Every current PyTorch CUDA wheel targets it. A V100 is `sm_70`, which recent
  CUDA 12.8 and 13.x builds have been progressively dropping, and the exact cut-off is not
  worth betting a run on.
- **Ada-generation fp16 throughput**, several times a T4's.

The trade-off is queue depth: 24 L40S GPUs cluster-wide against 272 V100s. If the L40S queue
is slow, switch both sbatch scripts to `--gpus=v100-32:1`. That is a 32 GB card, still enough
to hold the checkpoint resident, at the same SU rate. Keep `--dtype float16`: a V100 has no
usable bf16.

## Why `torch==2.9.1+cu128`

The CUDA toolkit modules on this cluster are split by GPU generation:
`cuda-legacy/12.9.2` is labelled "for V100 nodes" and `cuda-latest/13.3.1` "for L40S and
H100 nodes". That means the node drivers are not uniform, and the login nodes have no GPU, so
the driver version cannot be read before a job runs.

A CUDA 12.x wheel runs on any driver from 525 upward through CUDA minor-version
compatibility, so `cu128` is safe on every GPU node here regardless of which driver is
installed. A `cu130` wheel would require driver 580 or newer and is a gamble until the probe
job reports the real number. `scripts/setup_env.sh` pins `torch==2.9.1+cu128` and
`torchvision==0.24.1+cu128`, a matched pair.

The probe job prints `torch.cuda.get_arch_list()` beside the device's actual compute
capability and warns if the two do not line up.

---

## One-time setup, on a login node

Both steps only download and unpack files. **No model workload runs on a login node**, and
no CUDA kernel is launched there.

```bash
cd /ocean/projects/cis260288p/puprety/unsafenet-counterfactuals

# 1. Isolated Python 3.12 environment in Ocean, roughly 8 GB.
#    Re-runnable: it resumes after a failed wheel download. CLEAR_ENV=1 starts fresh.
bash scripts/setup_env.sh

# 2. Re-download the source clip (SHA-256 verified) and stage the checkpoint into an
#    Ocean-resident Hugging Face cache, roughly 14 GB.
bash scripts/stage_inputs.sh
```

Everything lands in Ocean, outside the repository:

```
/ocean/projects/cis260288p/puprety/
  envs/cogvx-cu128/     the virtual environment
  hf_cache/             HF_HOME, the staged CogVideoX-2b weights
  uv_cache/             wheel cache, safe to delete
  uv_python/            the uv-managed CPython 3.12
```

Staging the weights means the job sets `HF_HUB_OFFLINE=1`: a silent re-download becomes an
immediate failure, the job spends no allocation on network transfer, and the run does not
depend on outbound network access from the compute node.

---

## The two jobs

### 1. Hardware probe, `slurm/probe_gpu.sbatch`

Loads no model weights. Reports the assigned GPU, driver version, compute capability, VRAM,
RAM and resolved package versions, then launches one small fp16 matmul to prove the wheel and
the driver work together.

| | |
|---|---|
| Requested | 1 × `l40s-48`, 1 node, `GPU-shared`, 5 min wall |
| Worst-case charge | **0.08 SU** |
| Expected charge | under 0.03 SU |

```bash
sbatch slurm/probe_gpu.sbatch
squeue -u "$USER"
cat logs/probe-<jobid>.out
```

### 2. Baseline reproduction, `slurm/baseline.sbatch`

Stages 2, 3 and 4 in one job: ffmpeg preprocessing, the VAE-only reconstruction check, and
the prompt-guided source-video edit. Same clip, prompt, negative prompt, seed, scheduler,
strength, step count and dtype as `configs/experiment.json`.

| | |
|---|---|
| Requested | 1 × `l40s-48`, 1 node, `GPU-shared`, 60 min wall |
| Worst-case charge | **1.00 SU** |
| Expected charge | 0.17 to 0.33 SU, 10 to 20 minutes |
| Allocation after | 222 to 223 SU of 223 remaining |

The time estimate is rough on purpose. The T4 generation took 820.5 s of wall time, and the
step count behind that number is in dispute (see `MISSING_FILES.md`), so it cannot be scaled
cleanly. Sixty minutes is requested to give the first run headroom; it is not a prediction.

```bash
sbatch slurm/baseline.sbatch
squeue -u "$USER"
tail -f logs/baseline-<jobid>.out
sacct -j <jobid> --format=JobID,Elapsed,State,AllocTRES%45   # the real charge
```

Run the probe first. It costs almost nothing and it is the only way to learn the driver
version and confirm the wheel matches the card before a longer job is queued.

---

## Outputs

Written under `outputs/` in the repository, which is on Ocean, so they survive the job:

```
outputs/probe/probe_<jobid>.json
outputs/preprocessed/0_tr143_720x480x49.mp4
outputs/preprocessed/preprocess_record.json
outputs/stage3_vae/{reconstruction.mp4, source_vs_reconstruction.png, manifest.json}
outputs/stage4_edit/{edit_<tag>.mp4, comparison_<tag>.png, manifest_<tag>.json}
logs/{probe,baseline}-<jobid>.{out,err}
```

Each manifest records the Slurm job ID, partition, node, GPU, driver, resolved package
versions, every generation setting, timings and peak GPU memory. `outputs/` is gitignored:
the clips show identifiable workers who consented to the original study, not to
redistribution.

Copy results off Ocean before the allocation expires on 2027-09-08.

---

## After the baseline reproduces

In this order, and not before:

1. **Complete `templates/REVIEW_TEMPLATE.md`** against the new edited clip, all four
   criteria judged separately. Reject the spurious counterfactual explicitly: a repainted,
   widened or extended green walkway underneath the worker changes the environment, not the
   behaviour.
2. **Decide what the failure actually was.** The T4 run produced a clip but not the
   behaviour change. Candidate causes, which the comparison sheet and the strength sweep can
   distinguish: the edit restyled the clip without moving the worker; the prompt cannot
   express a 700-source-pixel lateral relocation; the pretrained checkpoint has no concept of
   this factory's walkway semantics. Those imply different fixes.
3. **Only then consider LoRA training.** It is the right answer to cause 3 and the wrong
   answer to causes 1 and 2, and it is outside this feasibility milestone as `CLAUDE.md`
   defines it. Plan it against a measured cause, with an SU budget, once a reproduced
   baseline exists to compare against.

Stage 5, classifier-guided feasibility, stays blocked: the Unsafe-Net authors published no
code and no weights, so no classifier with the right label space exists. Substituting an
unrelated classifier would invalidate the experiment.
