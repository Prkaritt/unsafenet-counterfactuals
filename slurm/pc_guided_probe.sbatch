#!/bin/bash
# Stage 5a of the panel-cover experiment: the gradient and memory probe.
# One guided step only. It answers whether a gradient taken at the predicted clean latent
# is finite and non-zero and whether the decoder's backward pass fits on the card. It does
# not answer whether following that gradient closes a panel.
#
# Requested: 1 L40S-48 GPU, 1 node, 25 minutes wall, GPU-shared.
# Maximum charge: 1 SU per L40S GPU-hour of elapsed time, so 25 minutes is at most
#                 0.417 SU. Expected 3 to 6 minutes, about 0.05 to 0.10 SU: one pipeline load, one latent encode and one backward pass through the VAE decoder.
#SBATCH --job-name=pcprobe
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:25:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcprobe-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/pcprobe-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "gpus     : ${SLURM_JOB_GPUS:-unset}"
echo "started  : $(date -Is)"
nvidia-smi

cd "${REPO}"
"${ENV_DIR}/bin/python" "${REPO}/scripts/pc_guided.py" \
  --mode probe \
  --case "${CASE:?set CASE to a case id}" \
  --input-arm "${INPUT_ARM:-source}" \
  --latent-window "${LATENT_WINDOW:-5}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
