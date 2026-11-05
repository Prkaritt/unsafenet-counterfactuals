#!/bin/bash
# LoRA stage, part 2 of 3: train the adapter.
#
# Trains a LoRA adapter on the CogVideoX-2b transformer over the verified class-4
# "Safe Walkway" intervals. The base checkpoint is not modified; only the adapter
# is learned, and it is saved separately.
#
# Authorised by the user on 2026-11-04 within the remaining original 10 SU budget.
# This supersedes the earlier "CogVideoX weights stay frozen, no LoRA training"
# constraint recorded in docs/ACCEPTANCE_CRITERIA.md, which is left unchanged as the
# preregistered record. Classifier guidance is not used anywhere in this stage.
#
# What this is NOT: it is not LD-ViCE. LD-ViCE fine-tunes CogVideoX per dataset AND
# denoises with target-model gradients. Only the first half is done here, deliberately,
# so that the base-versus-LoRA comparison isolates the adapter.
#
# Training data: data/lora_safe_walkway, built by scripts/build_lora_dataset.py from
# the class-4 train split only. The counterfactual source clip, every clip sharing a
# recording with it, every quarantined label-ambiguous clip, the held-out validation
# groups and the official test split are all excluded. Two further clips were dropped
# after scripts/verify_lora_intervals.py showed their windows do not contain the
# behaviour; see configs/lora_captions.json "_dropped".
#
# Trainer: diffusers examples/cogvideo/train_cogvideox_lora.py at tag v0.36.0, which is
# the tag matching the installed diffusers 0.36.0. The main-branch copy of the same
# script calls check_min_version("0.42.0.dev0") and would refuse to run here.
#
# Schedule, chosen from job 50229034's measurement, not from a guess:
#   measured 4.14 s per optimiser step steady state (5.48 s/it including the save),
#   peak 20.16 GiB of 45.0 GiB usable, 614 s from job start to the first step with
#   2 clips, dominated by loading the 13.8 GB checkpoint from Ocean.
#   1000 steps x 4.0 s + load + 16-clip pre-encode + save = about 81 minutes expected.
#   105 minutes requested, so the worst case is 1.75 SU.
#
# 1000 steps over 16 clips is 62 epochs. The upstream README suggests 1500-2000 steps
# for 50 videos; scaled by clip count that is nearer 500, and scaled by epochs (its
# example runs 30) nearer 480. 1000 sits above both deliberately, paired with the lower
# end of the recommended learning rate (1e-4 rather than 1e-3, which the README offers
# for datasets of 25+ videos and this one is 16), so the adapter gets enough updates
# without the large steps that would collapse it on a small set.
#
# Checkpoints every 250 steps, 2 kept. Job 50229034 exited non-zero AFTER a successful
# save; intermediate checkpoints mean a late failure cannot cost the whole run.
#
# Attempt 2. Attempt 1 (job 50229842, 0.0450 SU) died at step 0 with
#   RuntimeError: Cannot re-initialize CUDA in forked subprocess
# because --dataloader_num_workers 2 forks workers while the dataset already holds the
# pre-encoded latents as CUDA tensors. The probe ran with 0 workers and was fine; the
# training script still had 2. Now 0. Not a tuning choice: with the latents already on
# the GPU the loader does no decoding work, so workers buy nothing here.
#
# Spent before this job: 2.3841 SU of the approved 10 SU (verified with sacct).
# Remaining after the worst case: 5.91 SU, which keeps the generation job (0.75 SU
# worst case) and a retry comfortably inside the budget.
#SBATCH --job-name=cflora-train
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=01:45:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/loratrain-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/loratrain-%j.err

set -euo pipefail
PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export CKPT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

RANK=64
ALPHA=64
LR=1e-4
STEPS=1000
OUT="${REPO}/outputs/lora/adapter"

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "started : $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
cd "${REPO}"
echo "clips   : $(wc -l < data/lora_safe_walkway/videos.txt)"
echo "schedule: rank=${RANK} alpha=${ALPHA} lr=${LR} steps=${STEPS}"

rm -rf "${OUT}"
"${ENV_DIR}/bin/python" third_party/diffusers_v0.36.0/train_cogvideox_lora.py \
  --pretrained_model_name_or_path "${CKPT}" \
  --instance_data_root "${REPO}/data/lora_safe_walkway" \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --output_dir "${OUT}" \
  --height 480 --width 720 --fps 8 --max_num_frames 49 \
  --video_reshape_mode center \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps "${STEPS}" \
  --checkpointing_steps 250 \
  --checkpoints_total_limit 2 \
  --rank "${RANK}" --lora_alpha "${ALPHA}" \
  --learning_rate "${LR}" \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
  --optimizer adamw \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --enable_slicing --enable_tiling \
  --max_grad_norm 1.0 \
  --seed 20261103 \
  --dataloader_num_workers 0 \
  --report_to tensorboard

echo
echo "adapter files:"
ls -la "${OUT}"
echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Elapsed,State,MaxRSS
