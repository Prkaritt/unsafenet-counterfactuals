"""Download the CogVideoX-2b checkpoint into a plain Ocean directory, in parallel chunks.

Run from a login node. Pure data transfer: no model is constructed and no CUDA kernel is
launched.

Why this exists rather than ``huggingface_hub.snapshot_download``:

* ``THUDM/CogVideoX-2b`` was renamed and now 307-redirects to ``zai-org/CogVideoX-2b``.
* Two ``snapshot_download`` attempts from a Bridges-2 login node stalled dead on the large
  ``text_encoder`` shards. The ``.incomplete`` blobs did not grow for ten minutes and the
  library reported nothing.
* A single HTTP stream to ``huggingface.co`` from this login node sustains roughly 1 MB/s,
  which is about 3.6 hours for 12.83 GiB.

The Hugging Face CDN answers ``accept-ranges: bytes``, so each large file is fetched as
several byte ranges at once and reassembled. Every chunk is a separate ``curl`` with
``--continue-at -``, so an interrupted run resumes per chunk rather than per file.

Correctness is checked, not assumed. Each finished file must match the exact byte size and,
for LFS files, the SHA-256 recorded in ``bridges2/checkpoint_manifest.json``. The LFS
object id published by the hub *is* the file's SHA-256. A file whose size matches but whose
hash does not is deleted rather than kept.

The result is a plain directory, not a Hugging Face cache, so the Slurm job loads it by
path with no network access and no cache semantics in the way.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "bridges2" / "checkpoint_manifest.json"

# A single stream saturates well below the link, so several ranges run at once. 256 MiB
# keeps the chunk count small enough that reassembly is cheap and a lost chunk is cheap
# to refetch.
CHUNK_BYTES = 256 * 1024 * 1024
# Files at or under this size are fetched whole: splitting them costs more in round trips
# than it saves.
WHOLE_FILE_LIMIT = 64 * 1024 * 1024

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verified(path: Path, entry: dict) -> bool:
    """True when the file on disk is exactly the file the manifest describes."""
    if not path.exists() or path.stat().st_size != entry["bytes"]:
        return False
    if not entry.get("sha256"):
        return True
    return sha256_file(path) == entry["sha256"]


def curl(url: str, target: Path, byte_range: str | None, timeout_s: int) -> int:
    """One resumable curl. Returns curl's exit code.

    ``-s -S`` suppresses the progress meter but keeps real errors: this curl is 7.61 and
    predates ``--no-progress-meter``. ``--speed-limit``/``--speed-time`` abort a socket
    that has gone quiet, which is the exact failure the hub downloader sat through in
    silence; the caller then reconnects.
    """
    cmd = [
        "curl", "-fL", "-s", "-S",
        "--continue-at", "-",
        "-o", str(target),
        "--retry", "2", "--retry-delay", "3",
        "--connect-timeout", "30",
        "--max-time", str(timeout_s),
        "--speed-limit", "50000", "--speed-time", "45",
    ]
    if byte_range:
        cmd += ["--range", byte_range]
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and proc.stderr.strip():
        log(f"    curl rc={proc.returncode}: {proc.stderr.strip().splitlines()[0]}")
    return proc.returncode


def fetch_chunk(url: str, part: Path, start: int, end: int, label: str,
                attempts: int = 8) -> bool:
    """Fetch bytes [start, end] inclusive into ``part``, resuming what is there."""
    want = end - start + 1
    for n in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return True
        if have > want:
            # A previous run wrote past the range end. Start this chunk again.
            part.unlink()
            have = 0
        # --continue-at - resumes the local file, so the remote range must start at the
        # byte after what is already held, not at the chunk's own start.
        rc = curl(url, part, f"{start + have}-{end}", timeout_s=3600)
        now = part.stat().st_size if part.exists() else 0
        if now == want:
            return True
        if now == have and rc != 0:
            log(f"    {label}: no progress on attempt {n} ({now}/{want}); retrying")
            time.sleep(5)
    return (part.stat().st_size if part.exists() else 0) == want


def fetch_file(entry: dict, dest: Path, url_base: str, workers: int) -> bool:
    rel = entry["path"]
    want = entry["bytes"]
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{url_base}/{rel}"

    if verified(target, entry):
        log(f"ok      {rel}  ({want} bytes, already verified)")
        return True

    if want <= WHOLE_FILE_LIMIT:
        for _ in range(6):
            curl(url, target, None, timeout_s=600)
            if verified(target, entry):
                log(f"ok      {rel}  ({want} bytes)")
                return True
            if target.exists() and target.stat().st_size == want:
                log(f"BAD HASH {rel}: size matches, SHA-256 does not. Discarding.")
                target.unlink()
        log(f"FAILED  {rel}")
        return False

    ranges = [(s, min(s + CHUNK_BYTES, want) - 1)
              for s in range(0, want, CHUNK_BYTES)]
    parts = [target.with_suffix(target.suffix + f".part{i:03d}")
             for i in range(len(ranges))]
    held = sum(p.stat().st_size for p in parts if p.exists())
    log(f"fetch   {rel}  {want} bytes in {len(ranges)} chunks, "
        f"{held} already on disk")

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_chunk, url, part, start, end,
                        f"{rel}#{i:03d}"): i
            for i, (part, (start, end)) in enumerate(zip(parts, ranges))
        }
        bad = [futures[f] for f in cf.as_completed(futures) if not f.result()]
    if bad:
        log(f"FAILED  {rel}: chunks {sorted(bad)} incomplete. Re-run to resume.")
        return False

    elapsed = max(time.time() - t0, 1e-6)
    moved = want - held
    log(f"        {rel}: chunks done in {elapsed:.0f}s "
        f"({moved / elapsed / 1e6:.1f} MB/s), reassembling")

    tmp = target.with_suffix(target.suffix + ".joined")
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                while True:
                    block = fh.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
    tmp.replace(target)

    if not verified(target, entry):
        size = target.stat().st_size
        log(f"FAILED  {rel}: reassembled {size} bytes, verification failed. "
            "Discarding the file; chunks are kept for inspection.")
        target.unlink()
        return False

    for part in parts:
        part.unlink()
    log(f"ok      {rel}  ({want} bytes, SHA-256 verified)")
    return True


def seed_from_hub_cache(entries: list[dict], dest: Path, blobs: Path) -> None:
    """Reuse partial blobs an abandoned huggingface_hub attempt left behind.

    The cache names each blob after its SHA-256, which is the same value the manifest
    records, so a partial can be matched to its file. It becomes chunk 000, because the
    hub downloads a file from its start.
    """
    if not blobs.is_dir():
        return
    by_sha = {e["sha256"]: e for e in entries if e.get("sha256")}
    for blob in sorted(blobs.glob("*.incomplete")):
        entry = by_sha.get(blob.name[: -len(".incomplete")])
        if entry is None:
            continue
        have = blob.stat().st_size
        chunk0 = min(CHUNK_BYTES, entry["bytes"])
        if have <= 0 or have > chunk0:
            continue
        target = dest / entry["path"]
        part0 = target.with_suffix(target.suffix + ".part000")
        if target.exists() or (part0.exists() and part0.stat().st_size >= have):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.replace(part0)
        log(f"seeded  {entry['path']} chunk 000 with {have} of {chunk0} bytes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="/ocean/projects/cis260288p/puprety/checkpoints/CogVideoX-2b")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel byte-range requests per file (default 8)")
    args = ap.parse_args(argv)

    man = json.loads(MANIFEST.read_text())
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    url_base = f"https://huggingface.co/{man['repo_id']}/resolve/{man['revision']}"

    log(f"repo     : {man['repo_id']}  (legacy id {man['legacy_repo_id']})")
    log(f"revision : {man['revision']}")
    log(f"dest     : {dest}")
    log(f"total    : {man['total_bytes'] / 1024**3:.2f} GiB in {len(man['files'])} files")
    log(f"workers  : {args.workers} parallel ranges per file\n")

    seed_from_hub_cache(man["files"], dest,
                        Path(args.dest).parents[1] / "hf_cache" / "hub"
                        / "models--THUDM--CogVideoX-2b" / "blobs")

    t0 = time.time()
    failed = [e["path"] for e in man["files"]
              if not fetch_file(e, dest, url_base, args.workers)]

    if failed:
        log(f"\n{len(failed)} file(s) incomplete after this pass:")
        for rel in failed:
            log(f"  {rel}")
        log("Re-run this script; verified files are skipped and chunks resume.")
        return 1

    log(f"\nall {len(man['files'])} files present and SHA-256 verified, "
        f"{man['total_bytes'] / 1024**3:.2f} GiB in {time.time() - t0:.0f}s")
    rev = REPO / "bridges2" / "checkpoint_revision.txt"
    rev.write_text(man["revision"] + "\n")
    log(f"wrote {rev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
