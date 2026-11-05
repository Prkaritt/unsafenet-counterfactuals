"""Deterministic preprocessing of one source clip into CogVideoX input shape.

All numbers here are derived from the THUDM/CogVideoX-2b checkpoint configs, not
from memory:

  vae/config.json          sample_height=480, sample_width=720,
                           temporal_compression_ratio=4, latent_channels=16,
                           scaling_factor=1.15258426
  transformer/config.json  sample_frames=49, sample_height=60, sample_width=90,
                           patch_size=2, temporal_compression_ratio=4

60 * 8 = 480 and 90 * 8 = 720, so the VAE spatial compression ratio is 8.

The pipeline is driven by ffmpeg rather than by frame-by-frame Python decoding so
that one recorded command reproduces the preprocessed clip exactly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

# --- Checkpoint constraints (THUDM/CogVideoX-2b) ---------------------------------

TARGET_WIDTH = 720
TARGET_HEIGHT = 480
SAMPLE_FRAMES = 49
TEMPORAL_COMPRESSION_RATIO = 4
SPATIAL_COMPRESSION_RATIO = 8
LATENT_CHANNELS = 16
VAE_SCALING_FACTOR = 1.15258426

#: CogVideoX-2b was trained at 8 fps with 49 frames, i.e. a ~6 second clip.
TARGET_FPS = 8


class PreprocessError(ValueError):
    """Raised when a requested preprocessing configuration is invalid."""


# --- Frame-count validation ------------------------------------------------------


def latent_frame_count(num_frames: int) -> int:
    """Number of temporal latent positions the VAE produces for ``num_frames``.

    CogVideoX's 3D VAE keeps the first frame and compresses the rest by
    ``temporal_compression_ratio``, so 49 pixel frames give 13 latent frames.
    """
    if num_frames < 1:
        raise PreprocessError(f"num_frames must be >= 1, got {num_frames}")
    if (num_frames - 1) % TEMPORAL_COMPRESSION_RATIO != 0:
        raise PreprocessError(
            f"num_frames must satisfy (n - 1) %% {TEMPORAL_COMPRESSION_RATIO} == 0 "
            f"for the CogVideoX VAE; got {num_frames}"
        )
    return (num_frames - 1) // TEMPORAL_COMPRESSION_RATIO + 1


def validate_frame_count(num_frames: int, strict: bool = True) -> None:
    """Validate a requested frame count against the checkpoint.

    ``strict=True`` accepts only ``SAMPLE_FRAMES`` (49), the count the checkpoint
    was trained at. Any other count is unverified for this checkpoint, so it is
    rejected unless the caller explicitly opts out.
    """
    latent_frame_count(num_frames)  # raises on a bad temporal shape
    if strict and num_frames != SAMPLE_FRAMES:
        raise PreprocessError(
            f"num_frames={num_frames} is not the checkpoint's sample_frames="
            f"{SAMPLE_FRAMES}. Pass strict=False only after verifying that the "
            f"chosen count actually works on the assigned GPU."
        )


def validate_resolution(width: int, height: int, strict: bool = True) -> None:
    """Validate a requested output resolution against the checkpoint."""
    for name, value in (("width", width), ("height", height)):
        if value % SPATIAL_COMPRESSION_RATIO != 0:
            raise PreprocessError(
                f"{name}={value} is not divisible by the VAE spatial compression "
                f"ratio {SPATIAL_COMPRESSION_RATIO}"
            )
    if strict and (width, height) != (TARGET_WIDTH, TARGET_HEIGHT):
        raise PreprocessError(
            f"({width}, {height}) is not the checkpoint's "
            f"({TARGET_WIDTH}, {TARGET_HEIGHT}). Pass strict=False only after "
            f"verifying another resolution on the assigned GPU."
        )


# --- Crop geometry ---------------------------------------------------------------


@dataclass(frozen=True)
class CropBox:
    """An ffmpeg ``crop=w:h:x:y`` box in source pixel coordinates."""

    width: int
    height: int
    x: int
    y: int

    def as_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


def aspect_preserving_crop(
    src_width: int,
    src_height: int,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    x: int | None = None,
    y: int | None = None,
) -> CropBox:
    """Largest crop of the source that matches the target aspect ratio.

    ``x`` and ``y`` default to a centred crop. Pass them explicitly to keep a
    specific region: for a 1920x1080 source cropped to 3:2, the crop is 1620x1080
    and ``x`` selects which 1620-pixel-wide column survives.
    """
    if src_width <= 0 or src_height <= 0:
        raise PreprocessError(f"invalid source size {src_width}x{src_height}")
    target_aspect = target_width / target_height

    crop_w = src_width
    crop_h = round(src_width / target_aspect)
    if crop_h > src_height:
        crop_h = src_height
        crop_w = round(src_height * target_aspect)

    # Keep the crop on even pixel boundaries: yuv420p chroma is subsampled by 2.
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    cx = (src_width - crop_w) // 2 if x is None else x
    cy = (src_height - crop_h) // 2 if y is None else y

    if cx < 0 or cy < 0 or cx + crop_w > src_width or cy + crop_h > src_height:
        raise PreprocessError(
            f"crop {crop_w}x{crop_h}+{cx}+{cy} falls outside the "
            f"{src_width}x{src_height} source"
        )
    return CropBox(crop_w, crop_h, cx, cy)


# --- Temporal segment ------------------------------------------------------------


def frame_indices(start: int, step: int, count: int) -> list[int]:
    """Source frame indices selected by ``start``, ``step`` and ``count``."""
    if start < 0:
        raise PreprocessError(f"start must be >= 0, got {start}")
    if step < 1:
        raise PreprocessError(f"step must be >= 1, got {step}")
    if count < 1:
        raise PreprocessError(f"count must be >= 1, got {count}")
    return [start + i * step for i in range(count)]


def validate_segment(start: int, step: int, count: int, total_frames: int) -> list[int]:
    """Check that the requested segment exists in a clip of ``total_frames``."""
    idx = frame_indices(start, step, count)
    if idx[-1] >= total_frames:
        raise PreprocessError(
            f"segment needs source frame {idx[-1]} but the clip has only "
            f"{total_frames} frames (0..{total_frames - 1}). Reduce start, step "
            f"or count."
        )
    return idx


def effective_source_fps(src_fps: float, step: int) -> float:
    """Sampling rate in source time after taking every ``step``-th frame."""
    if src_fps <= 0:
        raise PreprocessError(f"src_fps must be > 0, got {src_fps}")
    if step < 1:
        raise PreprocessError(f"step must be >= 1, got {step}")
    return src_fps / step


# --- ffmpeg command --------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessSpec:
    """A complete, reproducible description of one preprocessing run."""

    source: str
    crop: CropBox
    start_frame: int
    frame_step: int
    num_frames: int = SAMPLE_FRAMES
    width: int = TARGET_WIDTH
    height: int = TARGET_HEIGHT
    out_fps: int = TARGET_FPS
    crf: int = 8

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crop"] = asdict(self.crop)
        d["latent_frames"] = latent_frame_count(self.num_frames)
        d["latent_height"] = self.height // SPATIAL_COMPRESSION_RATIO
        d["latent_width"] = self.width // SPATIAL_COMPRESSION_RATIO
        d["selected_frame_indices"] = frame_indices(
            self.start_frame, self.frame_step, self.num_frames
        )
        return d


def build_ffmpeg_command(
    spec: PreprocessSpec, destination: str | Path, ffmpeg: str = "ffmpeg"
) -> list[str]:
    """Build the ffmpeg argv that produces the preprocessed clip.

    Filter order matters. ``select`` runs on source frames so that ``start_frame``
    and ``frame_step`` are indices into the original clip. ``crop`` then ``scale``
    reduce the surviving frames, and ``setpts`` renumbers their timestamps for the
    output frame rate.
    """
    validate_frame_count(spec.num_frames)
    validate_resolution(spec.width, spec.height)

    last = spec.start_frame + (spec.num_frames - 1) * spec.frame_step
    select = (
        rf"select='gte(n\,{spec.start_frame})"
        rf"*lte(n\,{last})"
        rf"*not(mod(n-{spec.start_frame}\,{spec.frame_step}))'"
    )
    vf = ",".join(
        [
            select,
            spec.crop.as_filter(),
            f"scale={spec.width}:{spec.height}:flags=lanczos",
            f"setpts=N/{spec.out_fps}/TB",
        ]
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(spec.source),
        "-vf", vf,
        "-frames:v", str(spec.num_frames),
        "-r", str(spec.out_fps),
        "-an",
        "-c:v", "libx264",
        "-crf", str(spec.crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-y",
        str(destination),
    ]


def resolve_ffmpeg() -> str:
    """Path to an ffmpeg binary: the system one, else the imageio-ffmpeg bundle."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PreprocessError(
            "no ffmpeg on PATH and imageio-ffmpeg is not installed; "
            "pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: str | Path, ffmpeg: str | None = None) -> dict:
    """Read width, height, fps and frame count by decoding the clip once.

    ffprobe is not always installed next to ffmpeg (the imageio-ffmpeg wheel ships
    only ffmpeg), so this decodes to the null muxer and parses the report instead.
    """
    exe = ffmpeg or resolve_ffmpeg()
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = proc.stderr
    info = _parse_ffmpeg_report(report)
    if info.get("width") is None:
        raise PreprocessError(f"could not parse video stream from {path}:\n{report}")
    info["source"] = str(path)
    return info


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` only outside parentheses.

    An ffmpeg stream line contains commas inside groups, as in
    ``yuvj420p(pc, progressive)``, so a plain split would tear the fields apart.
    """
    parts, depth, current = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return [p for p in parts if p]


def _parse_ffmpeg_report(report: str) -> dict:
    """Parse resolution, pixel format, fps and frame count from ffmpeg stderr."""
    import re

    out: dict = {"width": None, "height": None, "fps": None, "frames": None,
                 "codec": None, "pix_fmt": None, "duration_s": None}

    line = next(
        (
            l
            for l in report.splitlines()
            if re.search(r"Stream #\d+:\d+.*: Video:", l)
        ),
        None,
    )
    if line is not None:
        body = line.split(": Video:", 1)[1]
        fields = _split_top_level(body)
        if fields:
            # "h264 (High) (avc1 / 0x31637661)" -> "h264"
            out["codec"] = fields[0].split()[0]
        for field in fields[1:]:
            size = re.match(r"^(\d+)x(\d+)$", field.split()[0])
            if size and out["width"] is None:
                out["width"] = int(size.group(1))
                out["height"] = int(size.group(2))
                continue
            # "yuvj420p(pc, progressive)" -> "yuvj420p"
            pix = re.match(r"^(yuv[\w]*|gbr[\w]*|rgb[\w]*|gray[\w]*|bgr[\w]*)", field)
            if pix and out["pix_fmt"] is None:
                out["pix_fmt"] = pix.group(1)
            fps = re.match(r"^([\d.]+) fps$", field)
            if fps and out["fps"] is None:
                out["fps"] = float(fps.group(1))

    if out["fps"] is None:
        fps = re.search(r"([\d.]+) fps", report)
        if fps:
            out["fps"] = float(fps.group(1))

    frames = re.findall(r"frame=\s*(\d+)", report)
    if frames:
        out["frames"] = int(frames[-1])

    dur = re.findall(r"time=(\d+):(\d\d):(\d\d\.\d+)", report)
    if dur:
        h, m, s = dur[-1]
        out["duration_s"] = round(int(h) * 3600 + int(m) * 60 + float(s), 3)

    return out


def preprocess_clip(
    spec: PreprocessSpec, destination: str | Path, ffmpeg: str | None = None
) -> dict:
    """Run the preprocessing and return a record of exactly what was produced."""
    exe = ffmpeg or resolve_ffmpeg()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_info = probe_video(spec.source, ffmpeg=exe)
    if source_info["frames"] is not None:
        validate_segment(
            spec.start_frame, spec.frame_step, spec.num_frames, source_info["frames"]
        )

    cmd = build_ffmpeg_command(spec, destination, ffmpeg=exe)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise PreprocessError(
            f"ffmpeg failed with code {proc.returncode}:\n{proc.stderr.strip()}"
        )

    out_info = probe_video(destination, ffmpeg=exe)
    if out_info["frames"] != spec.num_frames:
        raise PreprocessError(
            f"expected {spec.num_frames} output frames, ffmpeg produced "
            f"{out_info['frames']}"
        )
    if (out_info["width"], out_info["height"]) != (spec.width, spec.height):
        raise PreprocessError(
            f"expected {spec.width}x{spec.height}, ffmpeg produced "
            f"{out_info['width']}x{out_info['height']}"
        )

    return {
        "spec": spec.to_dict(),
        "ffmpeg_command": cmd,
        "source_probe": source_info,
        "output_probe": out_info,
        "output_path": str(destination),
        "effective_source_fps": effective_source_fps(
            source_info["fps"] or 0.0, spec.frame_step
        )
        if source_info["fps"]
        else None,
    }


def load_spec(path: str | Path) -> PreprocessSpec:
    """Load a ``PreprocessSpec`` from a JSON clip config."""
    data = json.loads(Path(path).read_text())
    crop = data["crop"]
    return PreprocessSpec(
        source=data["source"],
        crop=CropBox(crop["width"], crop["height"], crop["x"], crop["y"]),
        start_frame=data["start_frame"],
        frame_step=data["frame_step"],
        num_frames=data.get("num_frames", SAMPLE_FRAMES),
        width=data.get("width", TARGET_WIDTH),
        height=data.get("height", TARGET_HEIGHT),
        out_fps=data.get("out_fps", TARGET_FPS),
        crf=data.get("crf", 8),
    )
