#!/bin/bash
# Stage C (repaired) and Stage D.
#
# Changes from the previous attempt, both decided after seeing results and recorded in
# docs/DIAGNOSTIC_AMENDMENTS.md:
#   - the worker mask is placed by a COCO person detector, because motion does not isolate
#     a person in a working factory;
#   - gate B3 is evaluated on the full fixed validation split rather than 16 clips. The
#     0.90 threshold is unchanged and the original 16-clip result is preserved and
#     reported alongside.
#
# Detector weights are fetched here rather than on a login node: download.pytorch.org
# stalls at 0 bytes from the login node, while the compute nodes reach it.
#
# Requested: 1 L40S-48 GPU, 75 minutes, GPU-shared. Maximum charge 1.25 SU.
# Spent before this job: 0.428 SU of the approved 10 SU.
#SBATCH --job-name=cfeval
#SBATCH --account=cis260288p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=01:15:00
#SBATCH --output=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/eval-%j.out
#SBATCH --error=/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals/logs/eval-%j.err

set -euo pipefail
PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export COGVIDEOX_CHECKPOINT="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1

echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "started : $(date -Is)"
cd "${REPO}"

echo
echo "============== Fetch COCO person detector weights ======================"
"${ENV_DIR}/bin/python" -c "
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn as F
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights as W
m = F(weights=W.COCO_V1)
print('detector ready, person index', W.COCO_V1.meta['categories'].index('person'))
"

echo
echo "============== Mask overlays for visual validation ====================="
"${ENV_DIR}/bin/python" scripts/mask_overlays.py --per-class 3

echo
echo "============== Stage C: VAE consistency and shortcut diagnostics ======="
"${ENV_DIR}/bin/python" scripts/eval_classifier.py
echo "Stage C gates passed"

echo
echo "============== Stage D: guided-step gradient probe ====================="
"${ENV_DIR}/bin/python" scripts/guided_probe.py
echo "Stage D gates passed"

echo "finished : $(date -Is)"
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Elapsed,State
