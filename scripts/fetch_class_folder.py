#!/usr/bin/env python3
"""Download one Mendeley class folder of the Unsafe-Net dataset and verify every file.

The Mendeley public API will not list the dataset root, so a folder cannot be discovered
from the dataset id alone. It can be discovered from any single file in it: the file
metadata endpoint returns that file's ``folder_id``, and the folder can then be listed in
full. See bridges2/DATASET_ACCESS.md.

Nothing here runs on a GPU and nothing is written outside the given output directory. Every
file is checked against the sha256 the API publishes, and a file that fails is deleted
rather than left in place to be trained on.

Usage
-----
    python scripts/fetch_class_folder.py --folder-id <uuid> --out data/raw/class0_train
    python scripts/fetch_class_folder.py --from-file-id <uuid> --out data/raw/class4_train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/xjmtb22pff"
UA = "unsafenet-counterfactuals/1.0 (research use; PSC Bridges-2)"


def get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def folder_of_file(file_id: str) -> str:
    meta = get_json(f"{API}/files/{file_id}?version=1")
    return meta["folder_id"]


def list_folder(folder_id: str) -> list[dict]:
    return get_json(f"{API}/files?folder_id={folder_id}&version=1")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(entry: dict, out_dir: Path, retries: int = 3) -> tuple[str, bool]:
    name = entry["filename"]
    want = entry["content_details"]["sha256_hash"]
    dest = out_dir / name

    if dest.exists() and sha256_of(dest) == want:
        return name, True  # already verified; do not refetch

    url = entry["content_details"]["download_url"]
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r, dest.open("wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = sha256_of(dest)
            if got == want:
                return name, True
            print(f"  {name}: sha256 mismatch (want {want[:12]}, got {got[:12]})",
                  flush=True)
            dest.unlink(missing_ok=True)
        except Exception as exc:  # network, timeout, partial read
            print(f"  {name}: attempt {attempt}/{retries} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            dest.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    return name, False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--folder-id", help="Mendeley folder uuid to download in full")
    src.add_argument("--from-file-id",
                     help="any file uuid in the wanted folder; its folder is resolved "
                          "and then downloaded in full")
    ap.add_argument("--out", required=True, help="output directory, created if absent")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds between requests; keep this polite (default 0.5)")
    args = ap.parse_args(argv)

    folder_id = args.folder_id or folder_of_file(args.from_file_id)
    entries = list_folder(folder_id)
    if not entries:
        print(f"folder {folder_id} listed as empty; nothing to do")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(e["content_details"]["size"] for e in entries)
    print(f"folder   : {folder_id}")
    print(f"files    : {len(entries)}, {total_bytes / 1e9:.2f} GB")
    print(f"out      : {out_dir.resolve()}", flush=True)

    ok, failed = 0, []
    for i, entry in enumerate(sorted(entries, key=lambda e: e["filename"]), start=1):
        name, good = download(entry, out_dir)
        ok += good
        if not good:
            failed.append(name)
        if i % 10 == 0 or i == len(entries):
            print(f"[{i}/{len(entries)}] verified {ok}, failed {len(failed)}", flush=True)
        time.sleep(args.delay)

    manifest = {
        "folder_id": folder_id,
        "file_count": len(entries),
        "verified": ok,
        "failed": failed,
        "files": [{"filename": e["filename"],
                   "id": e["id"],
                   "sha256": e["content_details"]["sha256_hash"],
                   "size": e["content_details"]["size"]}
                  for e in sorted(entries, key=lambda e: e["filename"])],
    }
    (out_dir / "folder_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir / 'folder_manifest.json'}")

    if failed:
        print(f"FAILED {len(failed)} files: {failed}")
        return 1
    print("all files downloaded and sha256-verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
