#!/bin/bash
# Build the isolated Python environment for the Bridges-2 runs.
#
# Runs on a LOGIN NODE on purpose: this only downloads and unpacks wheels, which is a
# data-transfer task, not a model workload. No CUDA kernel is launched here.
#
# Everything lands in Ocean, not in $HOME, because $HOME is small and is not the
# project allocation.
set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
ENV_DIR="${PROJECT_ROOT}/envs/cogvx-cu128"
export UV_CACHE_DIR="${PROJECT_ROOT}/uv_cache"
export UV_PYTHON_INSTALL_DIR="${PROJECT_ROOT}/uv_python"
# pypi.nvidia.com serves the large CUDA runtime wheels and intermittently stalls from
# the Bridges-2 login nodes. Give each request time and keep the parallelism low.
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=4
UV="/opt/packages/uv/26.05/bin/uv"

# CUDA 12.8 wheels are chosen deliberately. CUDA minor-version compatibility means a
# 12.x wheel runs on any driver from 525 upward, so this build does not depend on the
# exact driver on the assigned GPU node. A cu130 wheel would require driver 580+.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PINS=("torch==2.9.1+cu128" "torchvision==0.24.1+cu128")

# Same application pins that resolved on the earlier T4 run, minus torch itself.
APP_PINS=(
  "diffusers==0.36.0"
  "transformers==4.57.6"
  "accelerate==1.15.0"
  "huggingface_hub>=0.34,<1.0"
  "sentencepiece==0.2.2"
  "imageio==2.37.0"
  "imageio-ffmpeg==0.8.1"
  "numpy==2.3.4"
  "pillow==11.3.0"
  "pytest>=8.0"
)

mkdir -p "${PROJECT_ROOT}/envs" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

echo "=== creating venv (python 3.12) at ${ENV_DIR} ==="
# --allow-existing makes this script safe to re-run after a failed download, which
# matters because the CUDA wheels are gigabytes. Pass CLEAR_ENV=1 to start fresh.
if [[ "${CLEAR_ENV:-0}" == "1" ]]; then
  "${UV}" venv --clear --python 3.12 "${ENV_DIR}"
else
  "${UV}" venv --allow-existing --python 3.12 "${ENV_DIR}"
fi

echo "=== installing torch from ${TORCH_INDEX} ==="
# Retried because a single stalled CUDA wheel download should not throw away the
# gigabytes already fetched. uv pip install is idempotent and reuses its cache.
for attempt in 1 2 3 4 5; do
  if VIRTUAL_ENV="${ENV_DIR}" "${UV}" pip install --index-url "${TORCH_INDEX}" "${TORCH_PINS[@]}"; then
    break
  fi
  echo "torch install attempt ${attempt} failed; retrying in 20s" >&2
  sleep 20
  if [[ "${attempt}" == "5" ]]; then
    echo "torch install failed after 5 attempts" >&2
    exit 1
  fi
done

echo "=== installing application pins from PyPI ==="
VIRTUAL_ENV="${ENV_DIR}" "${UV}" pip install "${APP_PINS[@]}"

echo "=== resolved versions ==="
"${ENV_DIR}/bin/python" - <<'PY'
import importlib, platform, sys
print("python  ", sys.version.split()[0])
print("platform", platform.platform())
for name in ["torch", "torchvision", "diffusers", "transformers", "accelerate",
             "huggingface_hub", "safetensors", "numpy", "PIL", "imageio",
             "imageio_ffmpeg", "sentencepiece"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name:18s} {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name:18s} MISSING ({type(exc).__name__}: {exc})")
import torch
print("torch cuda build  ", torch.version.cuda)
print("torch arch list   ", torch.cuda.get_arch_list())
print("cuda available    ", torch.cuda.is_available(), "(False is expected on a login node)")
PY

echo "=== freezing the exact set ==="
VIRTUAL_ENV="${ENV_DIR}" "${UV}" pip freeze > "${REPO}/bridges2/requirements-bridges2.lock.txt"
echo "wrote ${REPO}/bridges2/requirements-bridges2.lock.txt"
