#!/bin/bash
# LoRA stage, part 3 of 3: matched base-versus-LoRA edits of the source clip.
#
# Two video-to-video edits of outputs/preprocessed/0_tr143_720x480x49.mp4 that differ in
# exactly one thing, the trained adapter. Held identical: source clip, seed 20261103,
# strength 0.6, text guidance 6.0, 50 requested steps, scheduler, dtype, prompt, negative
# prompt, resolution and frame count.
#
# CLASSIFIER GUIDANCE IS DISABLED IN BOTH RUNS. scripts/lora_generate.py never imports
# the guidance module and never computes a classifier gradient; the classifier is loaded
# only to report probabilities, which are informational because Gate C failed.
#
# Both outputs are CANDIDATE counterfactuals until a human completes
# templates/REVIEW_TEMPLATE.md against them.
#
# Requested: 1 L40S-48 GPU, 45 minutes, GPU-shared. Maximum charge 0.75 SU.
#SBATCH --job-name=cflora-gen
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/loragen-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/loragen-%j.err

set -euo pipefail
PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "started : $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
cd "${REPO}"

"${ENV_DIR}/bin/python" scripts/lora_generate.py \
  --lora-dir outputs/lora/adapter \
  --rank 64 --lora-alpha 64 \
  --seed 20261103 \
  --steps 50 \
  --strength 0.6 \
  --guidance 6.0

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Elapsed,State,MaxRSS
