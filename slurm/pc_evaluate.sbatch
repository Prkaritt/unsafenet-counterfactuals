#!/bin/bash
# Stage 6 of the panel-cover experiment: measure every arm of every case.
# Panel state is reported as a correlation proxy against the case's own open and closed
# references, never as a detector output. Classifier scores are reported last and apart.
#
# Requested: 1 L40S-48 GPU, 1 node, 20 minutes wall, GPU-shared.
# Maximum charge: GPU-shared bills the allocated GPU at 1 SU per L40S GPU-hour of elapsed
#                 time, so 20 minutes is at most 0.333 SU. Expected 2 to 5 minutes, about 0.03 to 0.08 SU: decoding up to eight short clips per case and scoring them.
#
# The allocation has one resource only, Bridges-2 GPU. There is no CPU-only RM allocation
# on this grant, so even CPU-bound media work is charged here. That is why every stage is
# given the shortest wall time that fits its measured throughput.
#SBATCH --job-name=pceval
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:20:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pceval-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pceval-%j.err

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
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_evaluate.py" \
  --plan configs/panel_cover_plan.json \
  --cases "${CASES:-}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
