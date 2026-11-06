#!/bin/bash
# Stage 4 of the panel-cover experiment: fine-tune R3D-18 for class 2 against class 6.
# A new classifier. The walkway classifier is not loaded, reused or relabelled. Splits are
# separated by recording group, and the selected sources and donors are excluded together
# with everything sharing their recording groups.
#
# Requested: 1 L40S-48 GPU, 1 node, 70 minutes wall, GPU-shared.
# Maximum charge: GPU-shared bills the allocated GPU at 1 SU per L40S GPU-hour of elapsed
#                 time, so 70 minutes is at most 1.167 SU. Expected 25 to 45 minutes, about 0.42 to 0.75 SU: decoding about 148 clips once into a cache, then up to 25 epochs over roughly 110 training clips at 16x240x320.
#
# The allocation has one resource only, Bridges-2 GPU. There is no CPU-only RM allocation
# on this grant, so even CPU-bound media work is charged here. That is why every stage is
# given the shortest wall time that fits its measured throughput.
#SBATCH --job-name=pctrain
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:70:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pctrain-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pctrain-%j.err

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
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_train.py" \
  --mode "${MODE:-train}" \
  --epochs "${EPOCHS:-25}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
