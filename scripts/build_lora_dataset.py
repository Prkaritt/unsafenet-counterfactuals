#!/usr/bin/env python
"""Build the LoRA training dataset from verified class-4 (Safe Walkway) intervals.

Produces the layout the version-matched diffusers CogVideoX LoRA trainer expects
when ``--dataset_name`` is omitted:

    <out>/videos.txt      one path per line, relative to <out>
    <out>/prompts.txt     one caption per line, same order
    <out>/videos/<clip>.mp4

Every clip is preprocessed through ``unsafenet_cf.preprocess`` with the *same*
crop geometry and target shape as the counterfactual source clip ``0_tr143``, so
the training distribution matches what the video-to-video edit will be given.

Selection is read from the recorded split and group audits, never hardcoded:
  * class 4, ``train`` split only  (validation groups are held out)
  * the source clip and its recording group are excluded by the split itself
  * clips carrying label ambiguity are dropped (see ``--drop-ambiguous``)
  * the official test split is never touched; it has not been downloaded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unsafenet_cf.preprocess import (  # noqa: E402
    CropBox,
    PreprocessSpec,
    preprocess_clip,
    probe_video,
    resolve_ffmpeg,
)


def select_clips(splits: dict, groups: dict, drop_ambiguous: bool) -> tuple[list[str], dict]:
    split_of = splits["split_of"]
    train4 = sorted(
        (c for c, s in split_of.items() if c.startswith("4_") and s == "train"),
        key=lambda c: int(c.split("tr")[1]),
    )
    reasons: dict[str, str] = {}
    kept: list[str] = []
    ambiguous = groups.get("label_ambiguity", {})
    quarantined = set(groups.get("excluded_from_all_splits", {}))

    certain = groups.get("recording_overlap_pairs_certain", [])
    overlap_with_quarantined: dict[str, str] = {}
    for pair in certain:
        a, b = pair["a"], pair["b"]
        for x, y in ((a, b), (b, a)):
            if x in train4 and y in quarantined:
                overlap_with_quarantined[x] = (
                    f"shares a recording (min_mad {pair['min_mad']:.4f}) with "
                    f"{y}, which is quarantined for label ambiguity"
                )

    for clip in train4:
        if drop_ambiguous and clip in ambiguous:
            a = ambiguous[clip]
            reasons[clip] = (
                f"label-ambiguous: {a['shared_fraction']:.1%} of sampled frames also "
                f"appear in {a['with']}, published as class {a['other_class']}"
            )
            continue
        if clip in overlap_with_quarantined:
            reasons[clip] = overlap_with_quarantined[clip]
            continue
        kept.append(clip)
    return kept, reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="outputs/classifier/splits.json")
    ap.add_argument("--groups", default="outputs/classifier/groups.json")
    ap.add_argument("--source-spec", default="configs/clip_0_tr143.json")
    ap.add_argument("--raw-dir", default="data/raw/class4_train")
    ap.add_argument("--captions", default="configs/lora_captions.json")
    ap.add_argument("--out", default="data/lora_safe_walkway")
    ap.add_argument("--drop-ambiguous", action="store_true", default=True)
    ap.add_argument("--keep-ambiguous", dest="drop_ambiguous", action="store_false")
    ap.add_argument(
        "--contact-sheets",
        action="store_true",
        help="also render a frame contact sheet per clip, for visual verification",
    )
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    groups = json.loads(Path(args.groups).read_text())
    src = json.loads(Path(args.source_spec).read_text())

    kept, dropped = select_clips(splits, groups, args.drop_ambiguous)
    print(f"selected {len(kept)} clips; dropped {len(dropped)}")
    for c, why in dropped.items():
        print(f"  drop {c}: {why}")

    captions = {}
    cap_path = Path(args.captions)
    if cap_path.exists():
        captions = json.loads(cap_path.read_text())
        captions.pop("_comment", None)

    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    exe = resolve_ffmpeg()
    crop = CropBox(**src["crop"])

    records, video_lines, prompt_lines, missing_caption = [], [], [], []
    for clip in kept:
        raw = Path(args.raw_dir) / f"{clip}.mp4"
        info = probe_video(raw, ffmpeg=exe)
        total = info["frames"]
        n = src["num_frames"]
        # Keep the source clip's temporal stride where the clip is long enough;
        # otherwise take the largest stride that fits, so no clip is padded or
        # looped and every clip still yields exactly n frames.
        step = src["frame_step"]
        while step > 1 and 1 + (n - 1) * step > total:
            step -= 1
        if 1 + (n - 1) * step > total:
            raise SystemExit(f"{clip}: only {total} frames, cannot yield {n} at step 1")
        spec = PreprocessSpec(
            source=str(raw),
            crop=crop,
            start_frame=src["start_frame"],
            frame_step=step,
            num_frames=n,
            width=src["width"],
            height=src["height"],
            out_fps=src["out_fps"],
            crf=src["crf"],
        )
        dest = out / "videos" / f"{clip}.mp4"
        rec = preprocess_clip(spec, dest, ffmpeg=exe)
        rec["clip_id"] = clip
        rec["frame_step_reduced_from"] = src["frame_step"] if step != src["frame_step"] else None
        cap = captions.get(clip)
        if not cap:
            missing_caption.append(clip)
            cap = ""
        rec["caption"] = cap
        records.append(rec)
        video_lines.append(f"videos/{clip}.mp4")
        prompt_lines.append(cap)
        print(f"  {clip}: step={step} -> {dest.name}")

    (out / "videos.txt").write_text("\n".join(video_lines) + "\n")
    (out / "prompts.txt").write_text("\n".join(prompt_lines) + "\n")
    manifest = {
        "n_clips": len(kept),
        "clips": kept,
        "dropped": dropped,
        "drop_ambiguous": args.drop_ambiguous,
        "source_spec": src,
        "selection_policy": splits.get("policy"),
        "records": records,
        "captions_missing": missing_caption,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}/videos.txt, prompts.txt, dataset_manifest.json")
    if missing_caption:
        print(f"WARNING: {len(missing_caption)} clips have no caption yet: {missing_caption}")
        print("prompts.txt contains blank lines; write captions before training.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
