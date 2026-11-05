"""Perceptual hashing, duplicate detection, recording groups and scene groups.

The distinction this module exists to enforce: **a shared background is not a shared
recording.** Two clips of the same aisle filmed an hour apart share a scene and leak
nothing. Two clips cut from one continuous recording share actual frames and leak badly.
Conflating them either throws away usable data or silently inflates validation accuracy.

Evidence used, in decreasing strength:

1. **Identical sampled frames.** If two clips contain frames whose perceptual hashes are
   within ``DUP_FRAME_BITS`` of each other, those clips overlap in time. This is strong
   evidence of one recording and is treated as certain.
2. **Background similarity.** A clip's background signature is the per-pixel median over
   sampled frames, which suppresses the moving worker and keeps the room. Similar
   backgrounds indicate a shared camera position: a scene group, not a recording group.
3. **Filename index adjacency.** Weak and used only as a tiebreaker. Consecutive indices
   often but not always come from one recording, and one recording is not always
   consecutive, so adjacency alone never establishes a recording group.

Where evidence 1 is absent but evidence 2 and 3 both point the same way, the clips are
marked as an **uncertain** recording group. Callers are expected to exclude uncertain
groups from training when they touch the held-out source clip, and to report the
uncertainty rather than resolve it silently.
"""

from __future__ import annotations

import numpy as np

# A 64-bit aHash differing in at most this many bits means the two frames are, for our
# purposes, the same frame. Chosen conservatively: frames from one recording a second apart
# still differ by more than this once a worker moves, so this finds genuine overlap rather
# than mere similarity.
DUP_FRAME_BITS = 4

# Background signature distance below which two clips are taken to share a camera position.
# This is a scene judgement, never a recording judgement.
SCENE_CORR_THRESHOLD = 0.985


def ahash(frame: np.ndarray, size: int = 8) -> int:
    """64-bit average hash of one RGB frame.

    Downsample to size x size greyscale, then set one bit per pixel above the mean.
    """
    if frame.ndim == 3:
        grey = frame.astype(np.float64).mean(axis=2)
    else:
        grey = frame.astype(np.float64)
    h, w = grey.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    cells = np.empty((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            block = grey[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            cells[i, j] = block.mean() if block.size else 0.0
    bits = (cells > cells.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def background_signature(frames: np.ndarray, size: int = 32) -> np.ndarray:
    """Per-pixel median over frames, downsampled. Suppresses moving people, keeps the room.

    ``frames`` is (T, H, W, 3) uint8. Returns a flattened float vector, mean-centred and
    unit-normalised so two signatures can be compared with a dot product.
    """
    med = np.median(frames.astype(np.float32), axis=0)
    grey = med.mean(axis=2)
    h, w = grey.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    small = np.empty((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            block = grey[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            small[i, j] = block.mean() if block.size else 0.0
    vec = small.flatten()
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def scene_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of two background signatures, in [-1, 1]."""
    return float(np.dot(a, b))


def find_frame_overlaps(hashes: dict[str, list[int]],
                        bits: int = DUP_FRAME_BITS) -> list[tuple[str, str, int]]:
    """Clip pairs that share at least one near-identical sampled frame.

    Returns (clip_a, clip_b, best_distance) for each such pair. This is the strong evidence
    for a shared recording.
    """
    ids = sorted(hashes)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            best = min((hamming(x, y) for x in hashes[a] for y in hashes[b]),
                       default=64)
            if best <= bits:
                pairs.append((a, b, best))
    return pairs


def connected_components(ids: list[str],
                         edges: list[tuple[str, str]]) -> dict[str, str]:
    """Union-find over the given edges. Returns clip_id -> group label."""
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    groups: dict[str, str] = {}
    for i in ids:
        root = find(i)
        groups[i] = root
    return groups


# ---------------------------------------------------------------------------------
# Motion-residual signatures
#
# Added 2026-11-04 after the 64-bit average hash was measured and found unusable on this
# dataset. The clips are fixed-camera factory footage, so an 8x8 average hash is dominated
# by the room: class-0 and class-4 clips, which by definition come from different
# recordings, reached Hamming distance 0, and 36% of cross-class pairs fell within 4 bits.
# Union-find then merged almost every clip into one group.
#
# What distinguishes two frames from one continuous recording is not the room, which is
# shared by every clip from that camera, but the *moving content*: a specific person in a
# specific pose at a specific place. Subtracting the clip's own median background leaves
# exactly that. Two frames that overlap in time have near-identical residuals; two frames
# from the same camera at different times do not.
# ---------------------------------------------------------------------------------

# Cosine similarity between residual frames above which two frames are taken to be the
# same moment of one recording. Calibrated against measured distributions, not guessed;
# see docs/GROUPING_CALIBRATION.md.
RESIDUAL_SIM_THRESHOLD = 0.90

RESID_H = 36
RESID_W = 64


def _downsample(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Block-mean downsample of a 2D array to (out_h, out_w)."""
    h, w = plane.shape
    ys = np.linspace(0, h, out_h + 1).astype(int)
    xs = np.linspace(0, w, out_w + 1).astype(int)
    out = np.empty((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            block = plane[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out[i, j] = block.mean() if block.size else 0.0
    return out


def residual_signatures(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, RESID_H * RESID_W) float32, unit-normalised.

    Each row is one frame with the clip's median background removed, downsampled and
    normalised. A dot product between two rows is a cosine similarity: near 1 means the
    same moving content in the same place, which is evidence of one recording.

    A frame whose residual is empty, meaning nothing moved, is returned as a zero vector.
    Zero vectors have similarity 0 with everything, so a static frame never links two
    clips. That is deliberate: a still frame carries no evidence about which recording it
    came from.
    """
    grey = frames.astype(np.float32).mean(axis=3)
    med = np.median(grey, axis=0)
    out = np.empty((len(frames), RESID_H * RESID_W), dtype=np.float32)
    for i, plane in enumerate(grey):
        small = _downsample(plane - med, RESID_H, RESID_W)
        small = small - small.mean()
        norm = np.linalg.norm(small)
        out[i] = (small / norm).ravel() if norm > 1e-6 else 0.0
    return out


def find_recording_overlaps(signatures: dict[str, np.ndarray],
                            threshold: float = RESIDUAL_SIM_THRESHOLD
                            ) -> list[tuple[str, str, float]]:
    """Clip pairs sharing a moment, by residual cosine similarity.

    Returns (clip_a, clip_b, best_similarity) for pairs above the threshold. This replaces
    the average-hash test, which could not separate same-recording pairs from
    same-camera-different-time pairs on this dataset.
    """
    ids = sorted(signatures)
    pairs = []
    for i, a in enumerate(ids):
        sa = signatures[a]
        for b in ids[i + 1:]:
            best = float((sa @ signatures[b].T).max()) if sa.size and signatures[b].size \
                else 0.0
            if best >= threshold:
                pairs.append((a, b, best))
    return pairs


# ---------------------------------------------------------------------------------
# Duplicate detection by minimum mean absolute difference
#
# Third attempt, and the first one with a passing positive control. The two earlier
# metrics were rejected on measurement:
#
#   64-bit average hash   no specificity. Class-0 and class-4 pairs reached Hamming 0,
#                         and 36% of them fell within the 4-bit threshold.
#   motion residual       no sensitivity. Two overlapping halves of ONE clip, which share
#                         frames by construction, scored 0.4418 against a 0.90 threshold.
#                         It flagged only what it flagged correctly; it missed real
#                         overlaps.
#
# What works is the plainest thing: densely sample frames, and ask how close the closest
# pair of frames between two clips is, in mean absolute pixel difference. Measured on
# 160x90 greyscale frames at stride 4:
#
#   overlapping halves of one clip (true positive, shares frames)   0.00
#   0_tr98 vs 4_tr37 (verified duplicate footage)                   0.28
#   disjoint halves of one clip (same recording, no shared frame)   1.50
#   25 random cross-class pairs                                     minimum 2.19
#
# Hence two tiers rather than one threshold. Frames shared outright are certain; a near
# miss is probable and is reported as such instead of being decided silently.
# ---------------------------------------------------------------------------------

# Clips whose closest frames differ by at most this much share frames outright.
MAD_CERTAIN = 1.0
# Closer than this, but above MAD_CERTAIN: probably one recording, not proven. The
# evidence for this tier is a single measured case at 1.50 against a nearest random pair
# at 2.19, so the margin is thin and the tier is reported as uncertain.
MAD_PROBABLE = 2.0
# Correlation pre-filter. Every case of interest measured above 0.997, while the random
# pairs' maximum was 0.99478. The filter exists to avoid computing exact differences for
# every frame pair; it is set loose enough to let through anything the tiers might catch.
CORR_PREFILTER = 0.990

DENSE_H = 90
DENSE_W = 160
DENSE_STRIDE = 4
DENSE_CAP = 80


def pairwise_duplicate_scan(frames_by_clip: dict[str, np.ndarray],
                            corr_prefilter: float = CORR_PREFILTER,
                            mad_certain: float = MAD_CERTAIN,
                            mad_probable: float = MAD_PROBABLE,
                            device: str | None = None):
    """Find clip pairs that share frames, or probably share a recording.

    ``frames_by_clip`` maps a clip id to (T, H, W) uint8 greyscale frames.

    Returns (certain, probable), each a list of (clip_a, clip_b, min_mad).

    Exact minimum mean absolute difference over every frame pair is too expensive to
    compute for every clip pair, so a correlation pre-filter runs first as one matrix
    product. Correlation is cheap and, measured on this dataset, never excluded a pair
    that the tiers would have caught.
    """
    import torch

    ids = sorted(frames_by_clip)
    if not ids:
        return [], []
    dev = torch.device(device) if device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    counts = [len(frames_by_clip[i]) for i in ids]
    offsets = np.cumsum([0] + counts)
    flat = torch.from_numpy(
        np.concatenate([frames_by_clip[i].reshape(len(frames_by_clip[i]), -1)
                        for i in ids]).astype(np.float32)).to(dev)

    centred = flat - flat.mean(dim=1, keepdim=True)
    normed = centred / centred.norm(dim=1, keepdim=True).clamp_min(1e-6)

    certain, probable = [], []
    for a in range(len(ids)):
        sa = slice(int(offsets[a]), int(offsets[a + 1]))
        block_a = normed[sa]
        corr = block_a @ normed[int(offsets[a + 1]):].T
        if corr.numel() == 0:
            continue
        # Per later clip, the best correlation against any frame of clip a.
        for b in range(a + 1, len(ids)):
            rel = slice(int(offsets[b] - offsets[a + 1]),
                        int(offsets[b + 1] - offsets[a + 1]))
            best_corr = float(corr[:, rel].max())
            if best_corr < corr_prefilter:
                continue
            # Exact minimum mean absolute difference, only for surviving pairs.
            fa = flat[sa]
            fb = flat[int(offsets[b]):int(offsets[b + 1])]
            mad = torch.cdist(fa, fb, p=1).min() / fa.shape[1]
            value = float(mad)
            if value <= mad_certain:
                certain.append((ids[a], ids[b], value))
            elif value <= mad_probable:
                probable.append((ids[a], ids[b], value))
    return certain, probable
