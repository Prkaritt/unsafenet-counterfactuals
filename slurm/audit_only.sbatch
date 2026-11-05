#!/bin/bash
# Stage A only: decode every clip once, save motion-residual signatures, write splits.
#
# Separated from training because the first combined run failed at Stage B0 with an empty
# training split: the average-hash grouping over-merged and put every clip in validation.
# The signatures this job writes let a threshold be re-judged without decoding again, so
# that mistake costs one short job rather than one per attempt.
#
# Requested: 1 L40S-48 GPU, 20 minutes, GPU-shared. Maximum charge 0.33 SU.
# A GPU is requested for its 24 cores: GPU-shared bills only the GPU, while RM-shared
# bills per core-hour, so 24 cores are cheaper here than on a CPU partition.
#SBATCH --job-name=cfaudit
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:20:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/audit-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/audit-%j.err

set -euo pipefail
PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"
export OMP_NUM_THREADS=1

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "started : $(date -Is)"
cd "${REPO}"

# A non-zero exit here is expected while the threshold is uncalibrated: the signatures are
# written regardless, which is the point of this job.
"${ENV_DIR}/bin/python" scripts/audit_splits.py \
  --data-root "${REPO}/data/raw" \
  --workers "${SLURM_CPUS_ON_NODE:-8}" || echo "audit exited non-zero; signatures saved"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Elapsed,State
