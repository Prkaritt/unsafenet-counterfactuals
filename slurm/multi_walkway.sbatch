#!/bin/bash
# Multi-walkway comparison, step 2 of 2: generate six edits, then evaluate them.
#
# Three walkway-violation sources, each edited twice: once with no adapter, once with the
# adapter trained by job 50230058. Within each pair everything is held identical, including
# the initial noise, which is guaranteed by re-seeding the generator from seed 20261103
# immediately before each call. The plan, including every prompt, was frozen in
# outputs/multi_walkway/generation_plan.json before this job was submitted.
#
# NO classifier-gradient guidance anywhere. NO training. NO parameter sweeps.
#
# The base arm runs before the adapter is ever loaded, and the script counts LoRA parameter
# tensors on the transformer before and after loading, aborting if the base arm is
# contaminated or if the adapter fails to attach.
#
# Timing from measurement: the earlier generation job took 82.2 s and 75.1 s for two clips
# at these settings plus 208 s to load the pipeline. Six clips is about 8 minutes of
# generation, plus load, plus the evaluation pass over nine clips with the person detector.
# 45 minutes requested gives roughly 3x headroom.
#
# Requested: 1 L40S-48 GPU, 45 minutes, GPU-shared. Maximum charge 0.75 SU.
# Spent before this job: 3.6529 SU of the approved 10 SU (verified with sacct).
#SBATCH --job-name=cfmulti
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/cfmulti-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/cfmulti-%j.err

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
export PYTHONUNBUFFERED=1

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "started : $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
cd "${REPO}"

echo
echo "=============== generate six edits ==============="
"${ENV_DIR}/bin/python" scripts/multi_walkway_generate.py \
  --plan outputs/multi_walkway/generation_plan.json \
  --lora-dir outputs/lora/adapter --rank 64 --lora-alpha 64

echo
echo "=============== evaluate against each source's own lane ==============="
"${ENV_DIR}/bin/python" scripts/evaluate_multi_walkway.py \
  --plan outputs/multi_walkway/generation_plan.json --device cuda

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Elapsed,State,MaxRSS
