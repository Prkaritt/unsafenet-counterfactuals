"""Download the CogVideoX-2b weights into the Ocean-resident Hugging Face cache.

Run from a login node. This is pure data transfer: no model is constructed and no CUDA
kernel is launched.

Resumable on purpose. ``snapshot_download`` skips files already present in the cache and
resumes partial transfers, so a killed run loses only the file it was mid-way through. The
Bridges-2 login nodes terminate long-running heavy processes, and the T5-XXL shards take
long enough to be caught, so expect to need more than one attempt.

Exits non-zero if any requested file is still missing, so the caller can retry.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = "THUDM/CogVideoX-2b"

# Only the components the two stages load. The .bin duplicates of the safetensors shards
# are excluded, which keeps the cache near 14 GB instead of doubling it.
ALLOW = [
    "model_index.json",
    "scheduler/*",
    "vae/*",
    "transformer/*",
    "text_encoder/*",
    "tokenizer/*",
]
IGNORE = ["*.bin", "*.msgpack", "*.onnx", "*.md"]

# Every file the pipeline needs. Checked explicitly, because a partial snapshot that
# merely looks big is the failure mode that would blow up inside a Slurm job with
# HF_HUB_OFFLINE=1 set.
REQUIRED = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/spiece.model",
]


def dir_size_gib(path: Path) -> float:
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                continue
            try:
                total += os.path.getsize(full)
            except OSError:
                pass
    return total / 1024 ** 3


def attempt() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            CHECKPOINT,
            allow_patterns=ALLOW,
            ignore_patterns=IGNORE,
            max_workers=2,          # fewer parallel streams survive the login node better
        )
    )


def main() -> int:
    hf_home = os.environ.get("HF_HOME")
    print(f"HF_HOME   : {hf_home}", flush=True)
    if not hf_home:
        print("HF_HOME is not set. Refusing to write into the default $HOME cache, "
              "which is on a small filesystem.", file=sys.stderr)
        return 2

    path = None
    for n in range(1, 11):
        print(f"\n=== snapshot attempt {n} ===", flush=True)
        t0 = time.time()
        try:
            path = attempt()
        except Exception as exc:                      # network, timeout, partial read
            print(f"attempt {n} raised {type(exc).__name__}: {exc}", flush=True)
            time.sleep(10)
            continue
        print(f"attempt {n} returned in {time.time() - t0:.0f}s", flush=True)
        break

    if path is None:
        print("snapshot_download never completed. Re-run this script; it resumes.",
              file=sys.stderr)
        return 1

    missing = [rel for rel in REQUIRED if not (path / rel).exists()]
    print(f"\nsnapshot at {path}")
    print(f"cache size  {dir_size_gib(path):.2f} GiB")
    if missing:
        print(f"MISSING {len(missing)} required file(s):", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
        print("Re-run this script; it resumes from the cache.", file=sys.stderr)
        return 1
    print(f"all {len(REQUIRED)} required files present")

    # Pin the exact commit so the run manifest records which weights were used. The T4
    # run's revision is unknown because its manifest was lost with the Colab outputs.
    from huggingface_hub import HfApi

    sha = HfApi().model_info(CHECKPOINT).sha
    print(f"resolved revision {sha}")
    out = REPO / "bridges2" / "checkpoint_revision.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sha + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
