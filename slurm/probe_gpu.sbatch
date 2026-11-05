#!/bin/bash
# Hardware probe. Loads no model weights.
#
# Purpose: learn the actual GPU, driver version and compute capability on an assigned
# Bridges-2 node, and prove the installed PyTorch wheel can launch a kernel on it, before
# spending allocation on a generation run.
#
# Requested: 1 L40S-48 GPU, 5 minutes wall, GPU-shared.
# Charge:    GPU-shared bills only the GPUs allocated, at 1 SU per L40S GPU-hour.
#            Worst case, the full 5-minute request = 0.08 SU. Slurm bills elapsed time,
#            so the probe's real cost is under 0.03 SU.
#SBATCH --job-name=cfprobe
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:05:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/probe-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/probe-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
# The probe loads no weights; this is set only so its environment matches the
# baseline job it precedes.
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
# Set before CUDA initialises. Reduces allocator fragmentation.
# torch 2.9 renamed PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF and warns on the
# old name. Both are set so the setting applies whichever name the installed torch
# reads. It must be set before CUDA initialises.
export PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "gpus     : ${SLURM_JOB_GPUS:-unset}"
echo "started  : $(date -Is)"

"${ENV_DIR}/bin/python" "${REPO}/scripts/run_baseline.py" --stage probe

echo "finished : $(date -Is)"
