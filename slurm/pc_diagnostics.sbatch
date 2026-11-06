#!/bin/bash
# Stage 4b of the panel-cover experiment: classifier diagnostics.
# Two questions in one job so the model loads once: does the latent round trip alone change
# the prediction, and does masking the panel region move predictions more than masking
# control regions of identical area. Both classes are tested.
#
# Requested: 1 L40S-48 GPU, 1 node, 40 minutes wall, GPU-shared.
# Maximum charge: 1 SU per L40S GPU-hour of elapsed time, so 40 minutes is at most
#                 0.667 SU. Expected 12 to 22 minutes, about 0.20 to 0.37 SU: the VAE round trip costs roughly 43 s per clip and dominates, so the subset size is the cost knob.
#SBATCH --job-name=pcdiag
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:40:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcdiag-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcdiag-%j.err

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
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_diagnostics.py" \
  --vae-subset "${VAE_SUBSET:-8}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
