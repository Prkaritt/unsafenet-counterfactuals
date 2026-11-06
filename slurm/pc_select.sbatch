#!/bin/bash
# Stage 2 of the panel-cover experiment: score every clip against the identified panel.
# Decodes 148 clips at full 1920x1080 resolution, sampled, and measures alignment, panel
# correlation against the two class means, and region activity. It also writes contact
# sheets of every ranked clip's panel region. It selects nothing.
#
# Second run of this stage. The first run (job 50282082) scored the box named in the
# superseded region config and disproved it; the box now comes from the population
# difference in job 50282294. See DECISIONS.md P5 and P6.
#
# Requested: 1 L40S-48 GPU, 1 node, 35 minutes wall, GPU-shared.
# Maximum charge: 1 SU per L40S GPU-hour of elapsed time, so 35 minutes is at most
#                 0.583 SU. Expected 10 to 15 minutes, about 0.17 to 0.25 SU: the first
#                 run of this stage took 10 min 54 s and this one adds a second scoring
#                 pass over medians already in memory plus the contact sheets.
#SBATCH --job-name=pcsel
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:35:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcsel-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcsel-%j.err

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
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_select.py" \
  --regions configs/panel_cover_regions.json \
  --stride "${STRIDE:-8}" --cap "${CAP:-24}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
