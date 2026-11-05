#!/bin/bash
# Stage B: training memory probe, then train and validate the binary walkway classifier.
#
# Stage A is not repeated. Its signatures are cached, so the splits are recomputed from
# them in under a second rather than by decoding 228 clips again.
#
# Requested: 1 L40S-48 GPU, 45 minutes, GPU-shared. Maximum charge 0.75 SU.
# Spent before this job: 0.07 SU of the approved 10 SU.
#SBATCH --job-name=cftrain
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/train-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/train-%j.err

set -euo pipefail
PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "started : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
cd "${REPO}"

echo
echo "============== Stage A: splits from cached signatures =============="
"${ENV_DIR}/bin/python" scripts/audit_splits.py \
  --data-root "${REPO}/data/raw" --reuse-signatures
echo "Stage A passed"

echo
echo "============== Stage B0: training memory and time probe ============"
"${ENV_DIR}/bin/python" scripts/train_classifier.py --mode probe --batch-size 8
echo "Stage B0 passed"

echo
echo "============== Stage B: train and validate ========================="
"${ENV_DIR}/bin/python" scripts/train_classifier.py --mode train \
  --batch-size 8 --epochs 30 --decode-workers "${SLURM_CPUS_ON_NODE:-8}"
echo "Stage B gates passed"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Elapsed,State,MaxRSS
