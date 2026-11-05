#!/bin/bash
# Stage D (gradient and memory probe) then Stage E (one exploratory guided candidate).
#
# Run under explicit user authorisation given after Gate C failed. Gate C's failure stands
# in the record and is NOT marked passed: the shortcut diagnostic found no measurable
# dependence of the classifier on the worker, so whatever this guidance steers toward is
# not established to be the worker's position. The output is exploratory, not verified
# safety, and not a counterfactual until a human reviews it.
#
# Settings fixed by the user: seed 20261103, strength 0.6, text guidance 6.0, 50 requested
# steps, the pinned checkpoint and the existing prompts. Classifier guidance strength is
# recorded separately from text guidance in the manifest.
#
# CogVideoX and the classifier are frozen throughout. No training of any kind.
#
# Requested: 1 L40S-48 GPU, 60 minutes, GPU-shared. Maximum charge 1.00 SU.
# Spent before this job: 1.016 SU of the approved 10 SU.
#SBATCH --job-name=cfguide
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/guide-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/guide-%j.err

set -euo pipefail
PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
# torch 2.9 renamed PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF, but the allocator's
# own diagnostics still refer to the old name and a run with only the new one set showed
# 10.75 GiB reserved-but-unallocated defeating an 8.90 GiB request. Both are set, as
# slurm/baseline.sbatch already does. Must be set before CUDA initialises.
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "started : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
cd "${REPO}"

echo
echo "============== Stage D: gradient and memory probe ======================"
# Gates: finite, non-zero, coverage rotates with the sampling offset, correct direction,
# and the measured peak fits. A failure here stops the job before any generation.
"${ENV_DIR}/bin/python" scripts/guided_probe.py
echo "Stage D gates passed"

echo
echo "============== Stage E: guided candidate + unguided control ============"
"${ENV_DIR}/bin/python" scripts/guided_generate.py \
  --seed 20261103 \
  --steps 50 \
  --strength 0.6 \
  --guidance 6.0 \
  --classifier-scale 0.02 \
  --guidance-every 1

echo
echo "============== Build review folder ====================================="
"${ENV_DIR}/bin/python" scripts/build_review_page.py

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Elapsed,State,MaxRSS
