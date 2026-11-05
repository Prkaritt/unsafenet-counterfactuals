#!/bin/bash
# Stage 4 edit-strength sweep. Four generations from one pipeline load.
#
# Why a sweep: the single baseline run at strength 0.8 scored PSNR 15.89 dB against a
# source whose VAE round-trip ceiling is 32.60 dB, and the contact sheet shows the room
# itself redrawn. Strength is the knob that trades preservation against change, so this
# job measures that trade-off instead of guessing one more point on it.
#
# Grid, seed 20261103 and 50 requested steps throughout:
#   s=0.5 g=6.0   25 denoising steps
#   s=0.6 g=6.0   30
#   s=0.7 g=6.0   35
#   s=0.6 g=3.5   30   <- lower guidance, to test whether high guidance is what drives
#                         the whole-scene overhaul rather than strength alone
#
# Requested: 1 L40S-48 GPU, 1 node, 1 hour wall, GPU-shared partition.
# Charge:    GPU-shared bills only the GPUs allocated, at 1 SU per L40S GPU-hour, and
#            Slurm bills elapsed time rather than the requested limit.
#            Expected 9 to 12 minutes, so roughly 0.15 to 0.20 SU. Worst case 1.00 SU.
#
# Stages preprocess and vae are NOT repeated: their outputs are already in Ocean and the
# sweep reads the same preprocessed clip, so every run is compared against one source.
#
# No model workload runs on a login node.
#SBATCH --job-name=cfsweep
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/sweep-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/sweep-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
# Offline mode turns a silent re-download into an immediate, obvious failure. The weights
# load from the staged plain directory verified by scripts/fetch_weights.py.
export HF_HUB_OFFLINE=1
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
# torch 2.9 renamed PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF and warns on the old
# name. Both are set so the setting applies whichever name the installed torch reads.
# It must be set before CUDA initialises.
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "gpus     : ${SLURM_JOB_GPUS:-unset}"
echo "cores    : ${SLURM_CPUS_ON_NODE:-unset}"
echo "started  : $(date -Is)"
nvidia-smi

"${ENV_DIR}/bin/python" "${REPO}/scripts/run_baseline.py" \
  --stage edit \
  --dtype float16 \
  --offload none \
  --vae-tiling \
  --sweep s=0.5,g=6.0 \
  --sweep s=0.6,g=6.0 \
  --sweep s=0.7,g=6.0 \
  --sweep s=0.6,g=3.5

echo "finished : $(date -Is)"
echo "=== resource use ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
