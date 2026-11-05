"""Run manifests: capture what actually ran, not what was intended.

Every heavy import is deferred so that this module can be imported, and most of it
exercised, on a machine with no torch and no GPU.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

#: Packages whose exact resolved versions decide whether a run reproduces.
TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "numpy",
    "PIL",
    "imageio",
    "imageio_ffmpeg",
    "sentencepiece",
)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict:
    """Resolved version of each tracked package, or None when not installed."""
    import importlib

    out: dict = {}
    for name in names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            out[name] = None
            continue
        out[name] = getattr(mod, "__version__", "unknown")
    return out


def python_snapshot() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def nvidia_smi() -> str | None:
    """Raw ``nvidia-smi`` output, or None when the tool is absent."""
    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


def system_memory_gb() -> float | None:
    """Total system RAM in GiB, read from /proc/meminfo where available."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return round(kb / (1024 ** 2), 2)
    return None


def gpu_snapshot() -> dict:
    """GPU model and memory as torch reports it.

    Returns ``{"available": False, ...}`` rather than raising when torch is
    missing or no GPU is assigned, so a manifest can still be written.
    """
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda.is_available() is False"}

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    return {
        "available": True,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_vram_gb": round(props.total_memory / 1024 ** 3, 2),
        "free_vram_gb": round(free / 1024 ** 3, 2),
        "torch_cuda_version": torch.version.cuda,
        "supports_bf16": bool(torch.cuda.is_bf16_supported()),
    }


def reset_peak_memory() -> None:
    """Reset torch's peak-allocation counter before a measured stage."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float | None:
    """Peak GPU memory reserved since the last reset, in GiB."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)


def free_gpu() -> None:
    """Drop cached blocks after releasing a model. Call after ``del``."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def environment_snapshot() -> dict:
    """Everything about the machine that a reproduction attempt needs."""
    return {
        "captured_at": utc_now(),
        "python": python_snapshot(),
        "packages": package_versions(),
        "gpu": gpu_snapshot(),
        "system_ram_gb": system_memory_gb(),
        "nvidia_smi": nvidia_smi(),
    }


def build_run_manifest(
    stage: str,
    environment: dict,
    settings: dict,
    inputs: dict,
    outputs: dict,
    timing: dict,
    memory: dict,
    notes: str = "",
) -> dict:
    """Assemble one stage's manifest.

    ``stage`` is a short label such as ``"vae_reconstruction"`` or
    ``"prompt_guided_edit"``. Nothing here asserts that a generated clip is safe;
    the manifest records settings and measurements only.
    """
    return {
        "schema": "unsafenet-cf/run-manifest/1",
        "stage": stage,
        "created_at": utc_now(),
        "environment": environment,
        "settings": settings,
        "inputs": inputs,
        "outputs": outputs,
        "timing": timing,
        "memory": memory,
        "notes": notes,
        "claims": {
            "is_ld_vice_reproduction": False,
            "safety_verified": False,
            "classifier_evaluated": False,
            "status": "candidate counterfactual, pending human review",
        },
    }


def write_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    return path
