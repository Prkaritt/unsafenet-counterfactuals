#!/usr/bin/env python
"""Stage a self-contained, downloadable review package for the LoRA comparison.

Everything a reviewer needs on a laptop, with nothing that has to be fetched
again: the clips, the contact sheets, the adapter weights, every manifest, the
documents that say what the numbers do and do not establish, a checksum file, and
the rsync command that moves it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEST = Path("/ocean/projects/cis260288p/puprety/transfer/lora-counterfactual-review")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dest: Path, copied: list[tuple[Path, Path]]) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    copied.append((src, dest))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--mac-dest", default="~/Downloads/lora-counterfactual-review")
    ap.add_argument("--ssh-user", default=os.environ.get("USER", "<user>"),
                    help="Bridges-2 login name used in the printed rsync command")
    args = ap.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    copied: list[tuple[Path, Path]] = []
    missing: list[str] = []

    edit = REPO / "outputs" / "lora_edit"
    lora = REPO / "outputs" / "lora"

    videos = {
        "0_source_preprocessed.mp4": REPO / "outputs" / "preprocessed" / "0_tr143_720x480x49.mp4",
        "1_vae_reconstruction.mp4": REPO / "outputs" / "stage3_vae" / "reconstruction.mp4",
    }
    for mp4 in sorted(edit.glob("base_*.mp4")):
        videos["2_base_no_adapter.mp4"] = mp4
    for mp4 in sorted(edit.glob("lora_*.mp4")):
        videos["3_lora_adapter.mp4"] = mp4
    for name, src in videos.items():
        if not copy(src, dest / "00_videos" / name, copied):
            missing.append(str(src))

    for src in [edit / "comparison_base_vs_lora.png",
                edit / "walkway_overlay.png",
                edit / "worker_crops.png",
                edit / "pose_strip.png",
                REPO / "outputs" / "stage3_vae" / "source_vs_reconstruction.png"]:
        copy(src, dest / "01_comparisons" / src.name, copied)
    for src in sorted((lora / "contact_sheets").glob("*.png")):
        copy(src, dest / "01_comparisons" / "training_clips" / src.name, copied)

    for src in [edit / "manifest_base_vs_lora.json",
                lora / "train_manifest.json",
                lora / "interval_verification.json",
                lora / "probe_result.json",
                lora / "train_loss_summary.json",
                lora / "su_actuals.json",
                edit / "worker_position.json",
                REPO / "data" / "lora_safe_walkway" / "dataset_manifest.json",
                REPO / "configs" / "lora_captions.json",
                REPO / "configs" / "experiment.json",
                REPO / "configs" / "clip_0_tr143.json"]:
        if not copy(src, dest / "02_manifests" / src.name, copied):
            missing.append(str(src))

    # Only the final adapter. The checkpoint-N/ directories each carry a 225 MB
    # optimizer.bin, which is state for resuming training and is of no use to a reviewer;
    # including them made the package 745 MB. They stay on Ocean and train_manifest.json
    # says where.
    adapter = lora / "adapter"
    if adapter.exists():
        for src in sorted(adapter.glob("*")):
            if src.is_file():
                copy(src, dest / "03_adapter" / src.name, copied)
    else:
        missing.append(str(adapter))

    for src in [REPO / "data" / "lora_safe_walkway" / "videos.txt",
                REPO / "data" / "lora_safe_walkway" / "prompts.txt"]:
        copy(src, dest / "02_manifests" / src.name, copied)

    for src in [REPO / "docs" / "ACCEPTANCE_CRITERIA.md",
                REPO / "docs" / "DIAGNOSTIC_AMENDMENTS.md",
                REPO / "docs" / "GROUPING_CALIBRATION.md",
                REPO / "docs" / "SU_LEDGER.md",
                REPO / "templates" / "REVIEW_TEMPLATE.md",
                lora / "FINDINGS.md"]:
        copy(src, dest / src.name, copied)

    for src in sorted((REPO / "logs").glob("loratrain-*.out")) + \
               sorted((REPO / "logs").glob("loraprobe-*.out")) + \
               sorted((REPO / "logs").glob("loragen-*.out")) + \
               sorted((REPO / "logs").glob("loraverify-*.out")):
        copy(src, dest / "04_logs" / src.name, copied)

    lines = []
    for _, d in sorted(copied, key=lambda p: str(p[1])):
        lines.append(f"{sha256(d)}  {d.relative_to(dest)}")
    (dest / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

    total = sum(d.stat().st_size for _, d in copied)
    rsync = (
        f"rsync -avP --partial "
        f"{args.ssh_user}@data.bridges2.psc.edu:{dest}/ {args.mac_dest}/"
    )
    (dest / "HOW_TO_DOWNLOAD.txt").write_text(
        "Run this on the Mac, not on Bridges-2:\n\n"
        f"    mkdir -p {args.mac_dest}\n"
        f"    {rsync}\n\n"
        "Then verify the transfer:\n\n"
        f"    cd {args.mac_dest} && shasum -a 256 -c SHA256SUMS.txt\n\n"
        "Open 00_videos/ and play 0, 2 and 3 side by side. The comparison contact\n"
        "sheet is in 01_comparisons/. Judge the four criteria separately on\n"
        "REVIEW_TEMPLATE.md; the clips are candidate counterfactuals until you do.\n"
    )

    summary = {
        "destination": str(dest),
        "files": len(copied),
        "total_bytes": total,
        "missing": missing,
        "rsync_command": rsync,
    }
    (dest / "package_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if missing:
        print("\nWARNING: expected artefacts were not found and are absent from the package:")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
