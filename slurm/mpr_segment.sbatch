#!/bin/bash
# Stage 2 of the agent-specified relocation experiment: segment the worker with SAM 2.1,
# recover the background, and build the rough relocated composite for all three clips.
#
# Requested: 1 L40S-48 GPU, 1 node, 40 minutes wall, GPU-shared.
# Maximum charge: GPU-shared bills only the GPUs allocated, at 1 SU per L40S GPU-hour,
#                 so 40 minutes is at most 0.667 SU. Expected 10 to 15 minutes, about
#                 0.17 to 0.25 SU, because the work is three 49-frame clips through a
#                 small segmentation model plus array compositing.
#
# Environment: envs/sam2-seg, which reaches envs/cogvx-cu128 site-packages through a .pth
# file and adds only tqdm, hydra-core and iopath. SAM 2 itself is imported from Meta's
# repository on PYTHONPATH and is never installed, so no third-party setup script runs.
#SBATCH --job-name=mprseg
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:40:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/mprseg-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/mprseg-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/sam2-seg"
SAM2_REPO="${PROJECT_ROOT}/third_party_ext/sam2_official/sam2_repo"

export PYTHONPATH="${SAM2_REPO}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
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
"${ENV_DIR}/bin/python" "${REPO}/scripts/mpr_segment_composite.py" \
  --backend sam2 \
  --erase-dilate 7 \
  --scratch "${PROJECT_ROOT}/scratch/mpr"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
