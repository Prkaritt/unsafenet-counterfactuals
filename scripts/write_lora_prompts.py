#!/usr/bin/env python
"""Write prompts.txt (and optionally prune videos.txt) from the caption file.

Separate from ``build_lora_dataset.py`` so captions can be revised after looking
at the clips without re-encoding every video. Order always follows videos.txt, so
line *n* of prompts.txt is the caption of line *n* of videos.txt, which is what
the trainer assumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lora_safe_walkway")
    ap.add_argument("--captions", default="configs/lora_captions.json")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="clip ids to remove from videos.txt as well")
    args = ap.parse_args()

    ds = REPO / args.dataset
    caps = json.loads((REPO / args.captions).read_text())
    caps.pop("_comment", None)
    dropped_meta = caps.pop("_dropped", {})

    rels = [l.strip() for l in (ds / "videos.txt").read_text().splitlines() if l.strip()]
    drop = set(args.drop) | set(dropped_meta)
    kept = [r for r in rels if Path(r).stem not in drop]
    removed = [r for r in rels if Path(r).stem in drop]

    missing = [Path(r).stem for r in kept if not caps.get(Path(r).stem)]
    if missing:
        raise SystemExit(f"no caption for: {missing}")

    (ds / "videos.txt").write_text("\n".join(kept) + "\n")
    (ds / "prompts.txt").write_text("\n".join(caps[Path(r).stem] for r in kept) + "\n")
    print(f"{len(kept)} clips kept, {len(removed)} removed: {[Path(r).stem for r in removed]}")
    for r in kept:
        print(f"  {Path(r).stem}: {caps[Path(r).stem][:90]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
