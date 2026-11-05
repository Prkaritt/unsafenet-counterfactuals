#!/bin/bash
# Re-evaluate the six existing edits with a tracked worker. No generation, no training.
#
# The first evaluation (job 50249193) took the largest detection in each frame as "the
# worker". These clips contain distant figures as well as the foreground worker, so the
# largest flipped between people and the foot path jumped across the frame. That was
# visible in the first overlays as long straight lines between unrelated people, and it
# makes any distance statistic from that pass unusable. evaluate_multi_walkway.py now
# tracks one person with a jump gate and a height-consistency gate.
#
# Requested: 1 L40S-48 GPU, 15 minutes, GPU-shared. Maximum charge 0.25 SU.
#SBATCH --job-name=cfeval2
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:15:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/cfeval2-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/cfeval2-%j.err

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
cp outputs/multi_walkway/evaluation.json outputs/multi_walkway/evaluation_untracked_50249193.json
"${ENV_DIR}/bin/python" scripts/evaluate_multi_walkway.py \
  --plan outputs/multi_walkway/generation_plan.json --device cuda
echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,State
