#!/usr/bin/env python
"""Render contact sheets of the preprocessed LoRA training clips.

One row per clip, evenly spaced frames across the columns, several clips per
sheet. The sheets exist so that each selected 49-frame interval can be *looked
at* before it is used as training supervision, and so that captions are written
from what is visible rather than from the dataset label alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unsafenet_cf.clipset import iter_frames  # noqa: E402
from unsafenet_cf.compare import comparison_grid, pick_indices  # noqa: E402


def read_clip(path: Path, width: int, height: int):
    return [f for f in iter_frames(path, width=width, height=height)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lora_safe_walkway")
    ap.add_argument("--out", default="outputs/lora/contact_sheets")
    ap.add_argument("--per-sheet", type=int, default=5)
    ap.add_argument("--columns", type=int, default=6)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--height", type=int, default=240)
    args = ap.parse_args()

    ds = Path(args.dataset)
    clips = [line.strip() for line in (ds / "videos.txt").read_text().splitlines() if line.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sheets = []
    for start in range(0, len(clips), args.per_sheet):
        chunk = clips[start : start + args.per_sheet]
        labelled = []
        for rel in chunk:
            frames = read_clip(ds / rel, args.width, args.height)
            labelled.append((Path(rel).stem, frames))
        n = min(len(f) for _, f in labelled)
        idx = pick_indices(n, args.columns)
        sheet_path = out / f"sheet_{start // args.per_sheet + 1:02d}.png"
        res = comparison_grid(labelled, indices=idx, out_path=sheet_path)
        res.pop("image")
        sheets.append(res)
        print(f"{sheet_path}  rows={res['order']}  frames={res['frame_indices']}")

    (out / "sheets.json").write_text(json.dumps(sheets, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
