#!/bin/bash
# Stage 5b of the panel-cover experiment: matched guided and unguided generation.
# Both runs share source, prompt, seed, scheduler, strength, step count and initial noise.
# The only difference is the classifier term. CogVideoX stays frozen: no LoRA, no optimiser,
# no parameter updated.
#
# Requested: 1 L40S-48 GPU, 1 node, 60 minutes wall, GPU-shared.
# Maximum charge: 1 SU per L40S GPU-hour of elapsed time, so 60 minutes is at most
#                 1.0 SU. Expected 25 to 40 minutes, about 0.42 to 0.67 SU: two full runs at the same settings, one guided and one unguided, plus one pipeline load.
#SBATCH --job-name=pcguide
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:60:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcguide-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcguide-%j.err

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
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_guided.py" \
  --mode generate \
  --case "${CASE:?set CASE to a case id}" \
  --input-arm "${INPUT_ARM:-source}" \
  --classifier-scale "${CLASSIFIER_SCALE:-0.02}" \
  --guidance-every "${GUIDANCE_EVERY:-1}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
