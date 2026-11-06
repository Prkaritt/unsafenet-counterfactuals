#!/bin/bash
# Stage 3b of the panel-cover experiment: the CogVideoX arms, all in one job.
# One pipeline load serves every case and every arm, which is the whole reason these are
# not separate jobs. The walkway LoRA is never loaded and the transformer is inspected for
# LoRA parameter tensors before and after each generation.
#
# Requested: 1 L40S-48 GPU, 1 node, 60 minutes wall, GPU-shared.
# Maximum charge: GPU-shared bills the allocated GPU at 1 SU per L40S GPU-hour of elapsed
#                 time, so 60 minutes is at most 1.0 SU. Expected 18 to 30 minutes, about 0.30 to 0.50 SU: two VAE round trips per case at roughly 43 s each, plus one prompt-only edit and one refinement per case.
#
# The allocation has one resource only, Bridges-2 GPU. There is no CPU-only RM allocation
# on this grant, so even CPU-bound media work is charged here. That is why every stage is
# given the shortest wall time that fits its measured throughput.
#SBATCH --job-name=pcgen
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:60:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcgen-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcgen-%j.err

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
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_generate.py" \
  --plan configs/panel_cover_plan.json \
  --settings configs/panel_cover_generate.json \
  --arms "${ARMS:-vae_source,vae_composite,base_edit,refined,refined_restricted}" \
  --cases "${CASES:-}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
