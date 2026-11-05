#!/usr/bin/env python
"""Stage the downloadable review package for the multi-source comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = Path("/ocean/projects/cis260288p/puprety/transfer/multi-walkway-review")
MAC = str(Path.home() / "workspace/unsafenet-counterfactuals/unsafenet_cf_outputs/multi-walkway-review")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(DEST))
    ap.add_argument("--mac-dest", default=MAC)
    ap.add_argument("--ssh-user", default=os.environ.get("USER", "<user>"),
                    help="Bridges-2 login name used in the printed rsync command")
    args = ap.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    mw = REPO / "outputs" / "multi_walkway"
    plan = json.loads((mw / "generation_plan.json").read_text())
    copied: list[Path] = []
    missing: list[str] = []

    def cp(src: Path, rel: str) -> None:
        if not src.exists():
            missing.append(str(src)); return
        d = dest / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        copied.append(d)

    for s in plan["sources"]:
        cid = s["clip_id"]
        cp(REPO / "data" / "raw" / "class0_train" / f"{cid}.mp4",
           f"00_original_intervals/{cid}.mp4")
        cp(REPO / s["preprocessed"], f"01_preprocessed/{cid}_720x480x49.mp4")
        for arm in ("base", "lora"):
            cp(mw / "videos" / f"{cid}_{arm}.mp4", f"02_generated/{cid}_{arm}.mp4")
        cp(mw / "overlays" / f"{cid}_footpath.png", f"03_comparisons/{cid}_footpath.png")
    cp(mw / "source_sheet.png", "03_comparisons/source_sheet.png")

    for n in ("generation_plan.json", "selection_record.json", "source_selection.json",
              "source_selection_viewC.json", "selection_dense.json", "evaluation.json",
              "evaluation_untracked_50249193.json", "manifest_multi_walkway.json",
              "preprocess_0_tr48.json", "candidates.json", "candidates_viewC.json"):
        cp(mw / n, f"04_manifests/{n}")
    for n in ("experiment.json", "clip_0_tr143.json"):
        cp(REPO / "configs" / n, f"04_manifests/{n}")
    cp(REPO / "outputs" / "lora" / "train_manifest.json", "04_manifests/lora_train_manifest.json")
    cp(REPO / "outputs" / "lora" / "su_actuals.json", "04_manifests/su_actuals.json")

    for pat in ("cfsel-*.out", "cfsel3-*.out", "cfmulti-*.out", "cfeval2-*.out"):
        for f in sorted((REPO / "logs").glob(pat)):
            cp(f, f"05_logs/{f.name}")

    cp(mw / "FINDINGS.md", "FINDINGS.md")
    cp(REPO / "templates" / "REVIEW_TEMPLATE.md", "REVIEW_TEMPLATE.md")
    cp(REPO / "docs" / "ACCEPTANCE_CRITERIA.md", "ACCEPTANCE_CRITERIA.md")
    cp(REPO / "docs" / "DIAGNOSTIC_AMENDMENTS.md", "DIAGNOSTIC_AMENDMENTS.md")
    cp(REPO / "docs" / "SU_LEDGER.md", "SU_LEDGER.md")

    rev = mw / "review"
    if (rev / "index.html").exists():
        cp(rev / "index.html", "index.html")
        for f in sorted((rev / "media").glob("*")):
            cp(f, f"media/{f.name}")

    prompts = {s["clip_id"]: s["prompt"] for s in plan["sources"]}
    prompts["_negative_prompt"] = plan["settings"]["negative_prompt"]
    (dest / "04_manifests" / "prompts.json").write_text(json.dumps(prompts, indent=2))
    copied.append(dest / "04_manifests" / "prompts.json")

    lines = [f"{sha256(p)}  {p.relative_to(dest)}" for p in sorted(copied, key=str)]
    (dest / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

    rsync = (f"rsync -avP --partial "
             f"{args.ssh_user}@data.bridges2.psc.edu:{dest}/ {args.mac_dest}/")
    (dest / "HOW_TO_DOWNLOAD.txt").write_text(
        "Run on the Mac, not on Bridges-2:\n\n"
        f"    mkdir -p {args.mac_dest}\n    {rsync}\n\n"
        "Verify:\n\n"
        f"    cd {args.mac_dest} && shasum -a 256 -c SHA256SUMS.txt\n\n"
        "Open index.html in a browser. Play each source against its base and LoRA edit.\n"
        "Judge the four criteria separately on REVIEW_TEMPLATE.md. Every clip is a\n"
        "candidate counterfactual until you do.\n\n"
        "The clips show identifiable workers who consented to the original study, not to\n"
        "redistribution. Keep them local.\n")

    total = sum(p.stat().st_size for p in copied)
    summary = {"destination": str(dest), "files": len(copied),
               "total_bytes": total, "total_mb": round(total / 1e6, 1),
               "missing": missing, "rsync_command": rsync, "mac_dest": args.mac_dest}
    (dest / "package_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if missing:
        print("\nWARNING: not found, absent from the package:")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
