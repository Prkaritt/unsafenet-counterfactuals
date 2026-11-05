#!/bin/bash
# Measure the worker's position in the source, the base edit and the LoRA edit, with the
# same detector and walkway test used on the training intervals. Answers the one question
# the comparison was run to answer, objectively rather than by eye.
#
# Requested: 1 L40S-48 GPU, 15 minutes, GPU-shared. Maximum charge 0.25 SU.
#SBATCH --job-name=cflora-verify
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:15:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/loraverify-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/loraverify-%j.err

set -euo pipefail
PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
cd "${REPO}"
"${ENV_DIR}/bin/python" scripts/verify_edit_outputs.py --device cuda
echo "finished : $(date -Is)"
