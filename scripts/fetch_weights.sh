#!/bin/bash
# Download the CogVideoX-2b checkpoint into a plain directory in Ocean, with curl.
#
# Run from a login node. Pure data transfer: no model is constructed and no CUDA kernel
# is launched.
#
# Why curl and not huggingface_hub.snapshot_download:
#   THUDM/CogVideoX-2b was renamed and now 307-redirects to zai-org/CogVideoX-2b. Two
#   snapshot_download attempts from this login node stalled dead on the large
#   text_encoder shards, leaving .incomplete blobs that did not grow for ten minutes.
#   curl with byte-range resume is transparent: every attempt either adds bytes or
#   reports why it did not.
#
# Resumable. Each file is checked by exact byte size and, for LFS files, by SHA-256
# against the object id in bridges2/checkpoint_manifest.json. A file that already
# verifies is skipped, so re-running this script is cheap and safe.
#
# The result is a plain directory, not a Hugging Face cache, so the Slurm job loads it by
# path with no network access and no cache semantics in the way.
set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260288p/puprety"
REPO="${PROJECT_ROOT}/unsafenet-counterfactuals"
DEST="${PROJECT_ROOT}/checkpoints/CogVideoX-2b"
MANIFEST="${REPO}/bridges2/checkpoint_manifest.json"
PY="${PROJECT_ROOT}/envs/cogvx-cu128/bin/python"

REPO_ID="$("${PY}" -c "import json;print(json.load(open('${MANIFEST}'))['repo_id'])")"
REVISION="$("${PY}" -c "import json;print(json.load(open('${MANIFEST}'))['revision'])")"

echo "repo     : ${REPO_ID}"
echo "revision : ${REVISION}"
echo "dest     : ${DEST}"
mkdir -p "${DEST}"

# Seed the destination from any partial blobs an earlier huggingface_hub attempt left
# behind. The cache names each blob after its SHA-256, which is the same value the
# manifest records, so a partial can be matched to its file and resumed rather than
# re-downloaded from zero.
BLOBS="${PROJECT_ROOT}/hf_cache/hub/models--THUDM--CogVideoX-2b/blobs"
if [[ -d "${BLOBS}" ]]; then
  echo "=== seeding from abandoned hub partials in ${BLOBS} ==="
  "${PY}" - "${MANIFEST}" "${BLOBS}" "${DEST}" <<'PY'
import json, os, shutil, sys

manifest, blobs, dest = sys.argv[1], sys.argv[2], sys.argv[3]
files = json.load(open(manifest))["files"]
by_sha = {f["sha256"]: f for f in files if f.get("sha256")}

for name in sorted(os.listdir(blobs)):
    if not name.endswith(".incomplete"):
        continue
    sha = name[: -len(".incomplete")]
    entry = by_sha.get(sha)
    if entry is None:
        continue
    src = os.path.join(blobs, name)
    tgt = os.path.join(dest, entry["path"])
    have = os.path.getsize(src)
    if have <= 0 or have >= entry["bytes"]:
        continue
    if os.path.exists(tgt) and os.path.getsize(tgt) >= have:
        continue
    os.makedirs(os.path.dirname(tgt), exist_ok=True)
    shutil.move(src, tgt)
    print(f"  seeded {entry['path']} with {have} of {entry['bytes']} bytes")
PY
fi

echo "=== downloading ==="
# One curl per file. --continue-at - resumes from whatever is already on disk.
# --speed-limit/--speed-time abort a socket that goes quiet, which is exactly the
# failure the hub downloader sat through silently; the retry loop then reconnects.
"${PY}" - "${MANIFEST}" "${DEST}" "${REPO_ID}" "${REVISION}" <<'PY'
import hashlib, json, os, subprocess, sys

manifest, dest, repo_id, revision = sys.argv[1:5]
files = json.load(open(manifest))["files"]
MAX_ATTEMPTS = 12


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verified(path, entry):
    if not os.path.exists(path) or os.path.getsize(path) != entry["bytes"]:
        return False
    if not entry.get("sha256"):
        return True
    return sha256(path) == entry["sha256"]


failures = []
for entry in files:
    rel, want = entry["path"], entry["bytes"]
    target = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(target) or dest, exist_ok=True)

    if verified(target, entry):
        print(f"ok       {rel}  ({want} bytes, already verified)", flush=True)
        continue

    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{rel}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = os.path.getsize(target) if os.path.exists(target) else 0
        print(f"fetch    {rel}  attempt {attempt}, have {have} of {want}", flush=True)
        cmd = [
            "curl", "-fL", "--continue-at", "-", "-o", target, url,
            "--retry", "3", "--retry-delay", "5",
            "--connect-timeout", "30",
            # Give up on a socket delivering under 100 kB/s for 60 s, then reconnect.
            "--speed-limit", "100000", "--speed-time", "60",
            # -s suppresses the progress meter, which curl writes to the log on
            # every tick and which buries every other line; -S keeps real errors
            # visible. --no-progress-meter would be clearer but this curl is 7.61
            # and that option arrived in 7.67. File size on disk is the progress
            # indicator.
            "-s", "-S",
        ]
        proc = subprocess.run(cmd)
        # curl exits 33 when the server will not honour a range request; start over.
        if proc.returncode == 33 and os.path.exists(target):
            print(f"         server refused resume; restarting {rel} from zero",
                  flush=True)
            os.remove(target)
            continue
        if verified(target, entry):
            print(f"ok       {rel}  ({want} bytes, verified)", flush=True)
            break
        got = os.path.getsize(target) if os.path.exists(target) else 0
        if got == want:
            print(f"BAD HASH {rel}: size matches but SHA-256 does not. Discarding.",
                  file=sys.stderr, flush=True)
            os.remove(target)
    else:
        failures.append(rel)
        print(f"FAILED   {rel} after {MAX_ATTEMPTS} attempts", file=sys.stderr)

if failures:
    print(f"\n{len(failures)} file(s) incomplete:", file=sys.stderr)
    for rel in failures:
        print(f"  {rel}", file=sys.stderr)
    print("Re-run this script; verified files are skipped.", file=sys.stderr)
    sys.exit(1)

total = sum(f["bytes"] for f in files)
print(f"\nall {len(files)} files present and verified, {total / 1024**3:.2f} GiB")
PY

echo "${REVISION}" > "${REPO}/bridges2/checkpoint_revision.txt"
echo "=== done ==="
echo "checkpoint at ${DEST}"
du -sh "${DEST}"
