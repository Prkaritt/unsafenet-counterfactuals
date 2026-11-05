#!/bin/bash
# Stage every input a GPU job needs, from a login node.
#
# Two jobs, both pure data transfer:
#   1. Re-download the one source clip that was excluded from the transfer, and verify
#      its SHA-256 against configs/clip_0_tr143.json before anything uses it.
#   2. Pre-fetch the CogVideoX-2b weights into an Ocean-resident Hugging Face cache, so
#      the Slurm job does not spend allocation time downloading and does not depend on
#      outbound network access from the compute node.
set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"
export HF_HOME="${PROJECT_ROOT}/hf_cache"

CLIP="${REPO}/data/raw/0_tr143.mp4"
EXPECT_SHA="$(python3 -c "import json;print(json.load(open('${REPO}/configs/clip_0_tr143.json'))['source_sha256'])")"
URL="$(python3 -c "import json;print(json.load(open('${REPO}/configs/clip_0_tr143.json'))['source_download_url'])")"

mkdir -p "$(dirname "${CLIP}")" "${HF_HOME}"

if [[ -f "${CLIP}" ]]; then
  echo "=== clip already present, verifying ==="
else
  echo "=== downloading source clip from Mendeley ==="
  curl -fsSL -o "${CLIP}" "${URL}"
fi

GOT_SHA="$(sha256sum "${CLIP}" | cut -d' ' -f1)"
if [[ "${GOT_SHA}" != "${EXPECT_SHA}" ]]; then
  echo "SHA-256 MISMATCH for ${CLIP}" >&2
  echo "  expected ${EXPECT_SHA}" >&2
  echo "  got      ${GOT_SHA}" >&2
  echo "Do not continue. Delete the file and retry, or the Mendeley record changed." >&2
  exit 1
fi
echo "${CLIP}: $(stat -c%s "${CLIP}") bytes, sha256 verified"

echo "=== pre-fetching THUDM/CogVideoX-2b into ${HF_HOME} ==="
# Retried because the login nodes terminate long-running heavy processes, and the
# T5-XXL shards take long enough to be caught. The download resumes from the cache, so
# a killed attempt loses only the file it was mid-way through.
for attempt in 1 2 3 4 5 6 7 8; do
  if "${ENV_DIR}/bin/python" "${REPO}/scripts/prefetch_checkpoint.py"; then
    break
  fi
  echo "prefetch attempt ${attempt} did not complete; retrying in 15s" >&2
  sleep 15
  if [[ "${attempt}" == "8" ]]; then
    echo "prefetch failed after 8 attempts. Re-run this script; it resumes." >&2
    exit 1
  fi
done

echo "=== done ==="
