#!/bin/bash
# LoRA stage, part 1 of 3: training memory and timing probe.
#
# Attempt 3. What the earlier attempts cost and taught:
#   50226774 (0.0694 SU, FAILED)  interval verification completed and is on disk; the
#                                 trainer then died in AutoTokenizer because protobuf was
#                                 not installed. Fixed: protobuf 7.36.2.
#   50226994 (0.5019 SU, TIMEOUT) hit the 30 minute wall having printed NOTHING, because
#                                 the trainer was piped into `tail`, so everything it wrote
#                                 was still in the pipe when Slurm killed it. Two fixes
#                                 here: no pipe, and a smaller probe dataset.
#
# The probe trains on a TWO-CLIP subset, not the real 16. The script pre-encodes every
# clip through the VAE before the first optimiser step, and that cost is O(number of
# clips) and paid once per job. Measuring seconds-per-step does not need the whole set,
# and with all 16 the job never reached the step loop inside its wall clock.
#
# The adapter this produces is thrown away. Only the measurement is kept.
#
# Requested: 1 L40S-48 GPU, 40 minutes, GPU-shared. Maximum charge 0.67 SU.
# Spent before this job: 2.1505 SU of the approved 10 SU (verified with sacct).
#SBATCH --job-name=cflora-probe
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:40:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/loraprobe-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/loraprobe-%j.err

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

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "started : $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
cd "${REPO}"

# Sample the GPU every 30 s. If the trainer silently fell back to the CPU, utilisation
# stays at 0 and this file says so; without it a slow job is indistinguishable from a
# job running on the wrong device.
SAMPLE="${REPO}/logs/gpu-sample-${SLURM_JOB_ID}.csv"
( while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
               --format=csv,noheader >> "${SAMPLE}" 2>/dev/null || true
    sleep 30
  done ) &
SAMPLER_PID=$!
trap 'kill ${SAMPLER_PID} 2>/dev/null || true' EXIT

echo
echo "=========== training memory and timing probe (2-clip subset) ==========="
echo "phase: launching trainer at $(date -Is)"
PROBE_OUT="${REPO}/outputs/lora/probe_adapter"
rm -rf "${PROBE_OUT}"
# No pipe. Slurm captures stdout and stderr directly, so a job that is killed still
# leaves everything it had written.
/usr/bin/time -v "${ENV_DIR}/bin/python" third_party/diffusers_v0.36.0/train_cogvideox_lora.py \
  --pretrained_model_name_or_path "${CKPT}" \
  --instance_data_root "${REPO}/data/lora_probe_subset" \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --output_dir "${PROBE_OUT}" \
  --height 480 --width 720 --fps 8 --max_num_frames 49 \
  --video_reshape_mode center \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 8 \
  --checkpointing_steps 100000 \
  --rank 64 --lora_alpha 64 \
  --learning_rate 1e-4 \
  --lr_scheduler constant --lr_warmup_steps 0 \
  --optimizer adamw \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --enable_slicing --enable_tiling \
  --seed 20261103 \
  --dataloader_num_workers 0 \
  --report_to tensorboard

echo "phase: trainer returned at $(date -Is)"
echo
echo "GPU utilisation samples:"
tail -20 "${SAMPLE}" || true
echo
echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Elapsed,State,MaxRSS
