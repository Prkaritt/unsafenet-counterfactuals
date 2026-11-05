#!/bin/bash
# Stages A and B: data audit, training memory/time probe, then train and validate.
#
# One job, three steps, each gating the next. Stage A is CPU work but runs here rather than
# on a login node: PSC throttles login-node CPU, and the audit decodes a few hundred clips.
# GPU-shared bills only the GPUs allocated, so the node's 24 cores cost nothing extra and
# the parallel audit is effectively free inside this allocation.
#
# Requested: 1 L40S-48 GPU, 1 node, 1.5 hour wall, GPU-shared.
# Maximum possible charge: 1.50 SU. Budget for the whole experiment is 10 SU.
# No other job of this experiment may be in flight at the same time.
#SBATCH --job-name=cfclf
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/clf-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/clf-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "started  : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd "${REPO}"

echo
echo "===================== Stage A: audit and splits ====================="
# Training folders only. The official test folders are deferred by user decision, so no
# test clips are present and none are placed in any split.
"${ENV_DIR}/bin/python" scripts/audit_splits.py \
  --data-root "${REPO}/data/raw" \
  --workers "${SLURM_CPUS_ON_NODE:-8}"
echo "Stage A passed"

echo
echo "===================== Stage B0: training probe ======================"
# Shape compatibility is not memory fit. This measures a real forward, backward and
# optimiser step at the real batch size and input shape before any training starts.
"${ENV_DIR}/bin/python" scripts/train_classifier.py --mode probe --batch-size 8
echo "Stage B0 passed"

echo
echo "===================== Stage B: train and validate ==================="
"${ENV_DIR}/bin/python" scripts/train_classifier.py --mode train --batch-size 8 --epochs 30
echo "Stage B gates passed"

echo
echo "finished : $(date -Is)"
echo "=== resource use ==="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
