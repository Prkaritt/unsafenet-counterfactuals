#!/bin/bash
# Single-GPU baseline: Stage 2 preprocessing, Stage 3 VAE reconstruction, Stage 4
# prompt-guided source-video edit. This reproduces the earlier free-Colab T4 run on
# Bridges-2 with the same clip, prompt, seed, scheduler, strength, steps and dtype.
#
# Requested: 1 L40S-48 GPU, 1 node, 1 hour wall, GPU-shared partition.
# Charge:    GPU-shared bills only the GPUs allocated, at 1 SU per L40S GPU-hour.
#            Worst case, if the job runs the full hour, 1.00 SU.
#            Expected 10 to 20 minutes based on the T4 timings scaled to an L40S, so
#            roughly 0.17 to 0.33 SU. Slurm bills elapsed time, not the requested limit.
#
# No model workload runs on a login node. The environment and the checkpoint are built
# beforehand by scripts/setup_env.sh and scripts/fetch_weights.py, which only download and
# verify files.
#SBATCH --job-name=cfbase
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/baseline-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/baseline-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
# The weights load from a staged plain directory, not from the Hugging Face cache:
# scripts/fetch_weights.py downloads them with curl byte ranges and verifies every file
# against bridges2/checkpoint_manifest.json. Offline mode turns a silent re-download into
# an immediate, obvious failure, and removes any dependency on outbound network access
# from the compute node.
export HF_HUB_OFFLINE=1
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
# torch 2.9 renamed PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF and warns on the
# old name. Both are set so the setting applies whichever name the installed torch
# reads. It must be set before CUDA initialises.
export PYTORCH_ALLOC_CONF="expandable_segments:True"
# Keep thread counts sane: GPU-shared gives a fraction of a node's cores.
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "gpus     : ${SLURM_JOB_GPUS:-unset}"
echo "cores    : ${SLURM_CPUS_ON_NODE:-unset}"
echo "started  : $(date -Is)"
nvidia-smi

# --offload none keeps the whole 13.8 GB checkpoint resident on the device. The Colab
# run needed two-phase text encoding and enable_model_cpu_offload() only because that
# runtime had 12 GB of system RAM; a Bridges-2 GPU node has 500 GB or more.
#
# --vae-tiling stays on, matching the measured Colab configuration, so the comparison
# against the T4 numbers is like for like. Turn it off with --no-vae-tiling once the
# baseline is reproduced, if the 48 GB card makes an untiled decode worthwhile.
"${ENV_DIR}/bin/python" "${REPO}/scripts/run_baseline.py" \
  --stage all \
  --dtype float16 \
  --offload none \
  --vae-tiling

echo "finished : $(date -Is)"
echo "=== resource use ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
