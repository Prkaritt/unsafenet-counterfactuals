"""Clip inventory, label mapping and split construction for the binary walkway task.

Scope is deliberately narrow. Only two dataset classes are used:

    dataset class 0  "Safe Walkway Violation"  -> binary index 0
    dataset class 4  "Safe Walkway"            -> binary index 1

The other six behaviour classes are excluded from this experiment entirely. The mapping is
written to disk with every artefact that depends on it, because a silently transposed label
map turns "guide toward safe" into "guide toward violation" and nothing downstream would
notice.

Two different groupings are computed here and they are **not** the same thing:

``recording group``
    Clips cut from one continuous camera recording. Overlapping or adjacent segments share
    frames, so splitting them across train and validation leaks. Establishing this needs
    evidence of temporal adjacency, not merely a similar-looking room.

``scene group``
    Clips that appear to share a camera position and background. Useful for matched-scene
    evaluation, where a class-0 and a class-4 clip from the same scene test whether the
    model reads behaviour rather than room identity.

A shared factory background alone does **not** establish a shared recording. Where the
evidence cannot settle it, this module says so instead of guessing, and the caller is
expected to treat the uncertain cases conservatively.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Dataset class id -> binary output index. Written to disk by save_label_map().
DATASET_TO_BINARY = {0: 0, 4: 1}
BINARY_NAMES = {0: "walkway_violation", 1: "safe_walkway"}
DATASET_CLASS_NAMES = {0: "Safe Walkway Violation", 4: "Safe Walkway"}

# The source clip of the counterfactual experiment. It and its recording group are held out
# of training, validation and test alike.
SOURCE_CLIP = "0_tr143"

FILENAME_RE = re.compile(r"^(?P<cls>\d+)_(?P<split>tr|te)(?P<index>\d+)\.mp4$")


@dataclass
class Clip:
    """One dataset clip and everything derived from it."""

    path: Path
    dataset_class: int
    official_split: str           # "tr" or "te", from the publisher's filename
    index: int
    sha256: str | None = None
    duration_s: float | None = None
    frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    phash: list[str] = field(default_factory=list)   # per sampled frame
    recording_group: str | None = None
    scene_group: str | None = None
    recording_group_certain: bool = False

    @property
    def clip_id(self) -> str:
        return self.path.stem

    @property
    def binary_label(self) -> int:
        return DATASET_TO_BINARY[self.dataset_class]


def parse_filename(path: Path) -> tuple[int, str, int] | None:
    """Return (dataset_class, split, index) or None if the name does not match."""
    m = FILENAME_RE.match(path.name)
    if m is None:
        return None
    return int(m["cls"]), m["split"], int(m["index"])


def discover(roots: list[Path], keep_classes=(0, 4)) -> list[Clip]:
    """Find every usable clip under the given directories.

    Files whose names do not match the publisher's convention are skipped rather than
    guessed at, because the label is carried by the filename and nothing else.
    """
    clips: list[Clip] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.mp4")):
            parsed = parse_filename(path)
            if parsed is None:
                continue
            cls, split, index = parsed
            if cls not in keep_classes:
                continue
            if path.stem in seen:
                # Same clip id under two roots. Keep the first and let the audit report it.
                continue
            seen.add(path.stem)
            clips.append(Clip(path=path, dataset_class=cls,
                              official_split=split, index=index))
    return clips


def save_label_map(out_path: Path) -> dict:
    """Persist the dataset-to-binary label mapping next to any artefact that uses it."""
    mapping = {
        "dataset_class_to_binary_index": {str(k): v for k, v in DATASET_TO_BINARY.items()},
        "binary_index_to_name": {str(k): v for k, v in BINARY_NAMES.items()},
        "dataset_class_names": {str(k): v for k, v in DATASET_CLASS_NAMES.items()},
        "excluded_dataset_classes": [1, 2, 3, 5, 6, 7],
        "note": ("Binary index 1 is the counterfactual target (Safe Walkway). Guidance "
                 "maximises the logit at index 1. A transposed map would silently guide "
                 "toward the violation class."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2))
    return mapping


def ffprobe_media(path: Path) -> dict:
    """Read container metadata without requiring a system ffmpeg.

    Bridges-2 login and compute nodes have no ``ffprobe`` on PATH, but the installed
    ``imageio_ffmpeg`` wheel carries its own ffmpeg binary. Its ``read_frames`` generator
    yields a metadata dict before the first frame, which is enough and costs one decode of
    the header rather than of the clip.

    ``nframes`` from that dict is derived from duration and fps and can be off by a frame,
    so it is recorded as an estimate. Nothing here depends on it being exact: it only sets
    which frame indices to sample.
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        return {}
    try:
        reader = imageio_ffmpeg.read_frames(str(path))
        meta = next(reader)
    except Exception:
        return {}
    finally:
        # Closing the generator makes ffmpeg exit, and imageio_ffmpeg reports that on
        # stderr ("We had to kill ffmpeg to stop it."). It is expected here: the header
        # is all we wanted. A failure to close must not discard the metadata we have.
        try:
            reader.close()
        except Exception:
            pass
    size = meta.get("size") or (None, None)
    nframes = meta.get("nframes")
    if nframes in (None, float("inf")):
        fps, duration = meta.get("fps"), meta.get("duration")
        nframes = int(fps * duration) if fps and duration else None
    return {
        "width": size[0],
        "height": size[1],
        "fps": meta.get("fps"),
        "frame_count": int(nframes) if nframes else None,
        "duration_s": meta.get("duration"),
    }


def iter_frames(path: Path, width: int | None = None, height: int | None = None):
    """Yield (H, W, 3) uint8 frames one at a time, using the bundled ffmpeg.

    Passing ``width`` and ``height`` makes ffmpeg scale during decoding instead of
    returning full-resolution frames for the caller to shrink. On this dataset that is the
    difference between moving 1920x1080x3 bytes per frame and moving a fraction of it, over
    hundreds of clips, so every caller that wants small frames should ask for them here.
    """
    import numpy as np
    import imageio_ffmpeg

    output_params = None
    if width and height:
        output_params = ["-vf", f"scale={int(width)}:{int(height)}"]
    reader = imageio_ffmpeg.read_frames(str(path), output_params=output_params)
    meta = next(reader)
    w, h = (int(width), int(height)) if (width and height) else meta["size"]
    try:
        for raw in reader:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        try:
            reader.close()
        except Exception:
            pass
