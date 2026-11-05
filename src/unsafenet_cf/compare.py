"""Comparison utilities for source, VAE reconstruction and edited clips.

numpy and PIL are imported lazily so the module loads without them.
"""

from __future__ import annotations

from pathlib import Path


def _np():
    import numpy as np

    return np


def to_array(frames):
    """Stack a list of PIL images or arrays into a uint8 array (T, H, W, 3)."""
    np = _np()
    out = np.stack([np.asarray(f) for f in frames])
    if out.ndim != 4 or out.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3) uint8 frames, got {out.shape}")
    return out.astype(np.uint8)


def psnr(a, b, max_value: float = 255.0) -> float:
    """Peak signal-to-noise ratio between two equally shaped uint8 clips.

    Returns ``float('inf')`` when the clips are identical.
    """
    np = _np()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_value ** 2) / mse))


def per_frame_psnr(a, b) -> list[float]:
    """PSNR for each frame, to expose reconstruction drift over time."""
    np = _np()
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return [psnr(a[i], b[i]) for i in range(a.shape[0])]


def mean_abs_diff(a, b) -> float:
    """Mean absolute pixel difference, a blunt 'how much changed' number."""
    np = _np()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b)))


def pick_indices(num_frames: int, count: int = 4) -> list[int]:
    """Evenly spaced frame indices including the first and last frame."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    count = max(1, min(count, num_frames))
    if count == 1:
        return [0]
    step = (num_frames - 1) / (count - 1)
    return [round(i * step) for i in range(count)]


def comparison_grid(labelled_clips, indices=None, out_path=None, scale: int = 1):
    """Build a contact sheet: one row per clip, one column per sampled frame.

    ``labelled_clips`` is a sequence of ``(label, frames)`` pairs. Rows are
    labelled in the returned ``order`` list rather than drawn into the image, so
    no font dependency is needed.
    """
    from PIL import Image

    np = _np()
    clips = [(label, to_array(frames)) for label, frames in labelled_clips]
    if not clips:
        raise ValueError("no clips given")

    shapes = {c.shape[1:] for _, c in clips}
    if len(shapes) != 1:
        raise ValueError(f"clips differ in frame size: {shapes}")
    num_frames = min(c.shape[0] for _, c in clips)

    if indices is None:
        indices = pick_indices(num_frames, 4)
    bad = [i for i in indices if i < 0 or i >= num_frames]
    if bad:
        raise ValueError(f"frame indices {bad} outside 0..{num_frames - 1}")

    rows = [np.concatenate([clip[i] for i in indices], axis=1) for _, clip in clips]
    sheet = np.concatenate(rows, axis=0)
    image = Image.fromarray(sheet)
    if scale != 1:
        image = image.resize(
            (image.width * scale, image.height * scale), Image.LANCZOS
        )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
    return {
        "order": [label for label, _ in clips],
        "frame_indices": list(indices),
        "path": str(out_path) if out_path else None,
        "image": image,
    }
