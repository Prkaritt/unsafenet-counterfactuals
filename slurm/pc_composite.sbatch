#!/bin/bash
# Stage 3a of the panel-cover experiment: mask-restricted closed-panel compositing.
# No model is loaded. Registration is fitted from landmark patches on static scene
# structure, and every check (region coverage, untouched pixels outside the region, seam
# step, region stability) is written into outputs/panel_cover/<case>/composite.json.
#
# Requested: 1 L40S-48 GPU, 1 node, 25 minutes wall, GPU-shared.
# Maximum charge: GPU-shared bills the allocated GPU at 1 SU per L40S GPU-hour of elapsed
#                 time, so 25 minutes is at most 0.417 SU. Expected 4 to 8 minutes, about 0.07 to 0.13 SU: ffmpeg preprocessing of six clips plus array compositing of three 49-frame clips.
#
# The allocation has one resource only, Bridges-2 GPU. There is no CPU-only RM allocation
# on this grant, so even CPU-bound media work is charged here. That is why every stage is
# given the shortest wall time that fits its measured throughput.
#SBATCH --job-name=pccomp
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:25:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pccomp-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pccomp-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "gpus     : ${SLURM_JOB_GPUS:-unset}"
echo "started  : $(date -Is)"
nvidia-smi

cd "${REPO}"
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_composite.py" \
  --plan configs/panel_cover_plan.json \
  --cases "${CASES:-}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
