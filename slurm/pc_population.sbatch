#!/bin/bash
# Stage 2b of the panel-cover experiment: find the labelled object from the populations.
# Builds a median per aligned clip at 960x540, averages those medians within each class,
# differences the two averages, and tests how well each strong region separates the classes
# clip by clip. It selects nothing. See DECISIONS.md P6 for why a per-pair difference failed.
#
# Requested: 1 L40S-48 GPU, 1 node, 30 minutes wall, GPU-shared.
# Maximum charge: 1 SU per L40S GPU-hour of elapsed time, so 30 minutes is at most
#                 0.500 SU. Expected 6 to 12 minutes, about 0.10 to 0.20 SU: the same
#                 clips at full resolution took 10:54, and this reads 20 frames per clip
#                 at a quarter of the pixels.
#SBATCH --job-name=pcpop
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcpop-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcpop-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "started  : $(date -Is)"

cd "${REPO}"
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_population.py" \
  --align-min "${ALIGN_MIN:-0.95}" --top "${TOP:-6}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
