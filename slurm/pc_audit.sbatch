#!/bin/bash
# Stage 1 of the panel-cover experiment: audit the class-2 and class-6 training folders.
#
# What it does: decodes every clip once at 480x270, clusters camera views from median
# backgrounds, finds recording groups by frame overlap, measures the regions where a
# class-2 median differs from a class-6 median of the same view, and writes a contact
# sheet per clip plus a difference map per cross-class same-view pair. It selects nothing.
#
# Why a GPU job for CPU work: the duplicate scan in src/unsafenet_cf/grouping.py runs its
# correlation pre-filter and its exact cdist on the GPU when one is present. On 148 clips
# the all-pairs part is the expensive half. Decoding is CPU-bound and this is still the
# cheapest place to run it, because login nodes must not carry substantial media work.
#
# Requested: 1 L40S-48 GPU, 1 node, 50 minutes wall, GPU-shared.
# Maximum charge: 1 SU per L40S GPU-hour of elapsed time, so 50 minutes is at most
#                 0.833 SU. Expected 12 to 20 minutes, about 0.20 to 0.33 SU: the
#                 class-0/4 audit decoded 228 clips in roughly that time.
#SBATCH --job-name=pcaudit
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:50:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcaudit-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcaudit-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "gpus     : ${SLURM_JOB_GPUS:-unset}"
echo "started  : $(date -Is)"
nvidia-smi

cd "${REPO}"
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_audit.py" \
  --roots data/raw/class2_train data/raw/class6_train \
  --out outputs/panel_cover/audit

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
