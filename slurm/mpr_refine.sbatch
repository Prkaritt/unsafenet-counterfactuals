#!/bin/bash
# Stage 3: frozen CogVideoX-2b refinement of the agent-specified relocation composites.
# No LoRA adapter is loaded. No classifier gradient is computed.
#
# Requested: 1 L40S-48 GPU, 1 node, 35 minutes wall, GPU-shared.
# Maximum charge: 0.583 SU at 1 SU per L40S GPU-hour.
# Expected:  pipeline load 160-210 s, then roughly 25-35 s per clip at strength 0.2,
#            which executes 10 of 50 steps. Three clips, about 5 to 6 minutes total,
#            so roughly 0.10 SU. The pilot variant runs one clip, about 0.05 SU.
#
# Pass CLIPS=<id> to run a subset, which is how the single-clip pilot is run first.
#SBATCH --job-name=mprref
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:35:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/mprref-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/mprref-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "clips    : ${CLIPS:-all}"
echo "started  : $(date -Is)"
nvidia-smi

cd "${REPO}"
"${ENV_DIR}/bin/python" "${REPO}/scripts/mpr_refine.py" --clips "${CLIPS:-}"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,State,MaxRSS
