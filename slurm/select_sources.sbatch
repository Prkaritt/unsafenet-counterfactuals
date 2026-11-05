#!/bin/bash
# Multi-walkway comparison, step 1 of 2: characterise candidate source clips.
#
# Measures, for 15 eligible class-0 candidates, where the worker's feet are relative to
# that clip's own green lane, at full 1920x1080 resolution, plus a crop window adapted to
# each source. The three sources are then chosen from these numbers, and the reasons are
# written down BEFORE any generation runs.
#
# Candidates were filtered on a login node from the recorded audits: class 0, present on
# disk, not 0_tr143, not quarantined or label-ambiguous, recording group not overlapping
# LoRA training footage or 0_tr143, no uncertain overlap with a LoRA training clip, and
# one clip per recording group, drawn from three different scene clusters.
#
# Requested: 1 L40S-48 GPU, 15 minutes, GPU-shared. Maximum charge 0.25 SU.
# Spent before this job: 3.6072 SU of the approved 10 SU (verified with sacct).
#SBATCH --job-name=cfsel
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:15:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/cfsel-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/cfsel-%j.err

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
"${ENV_DIR}/bin/python" scripts/select_walkway_sources.py \
  --candidates outputs/multi_walkway/candidates.json --device cuda
echo "finished : $(date -Is)"
