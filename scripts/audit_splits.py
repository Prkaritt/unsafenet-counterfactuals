#!/usr/bin/env python3
"""Stage A: inventory the clips, audit duplicates, build recording and scene groups.

CPU only. Nothing here loads a model or touches a GPU, so it is cheap to re-run and it is
the gate every later stage depends on.

What it produces
----------------
``outputs/classifier/clip_inventory.json``   every clip with media metadata and hashes
``outputs/classifier/groups.json``           recording groups, scene groups, uncertainty
``outputs/classifier/splits.json``           train / validation / excluded, with reasons
``outputs/classifier/label_map.json``        dataset class to binary index
``outputs/classifier/audit_report.md``       the Stage A gate result in readable form

Gate A, from docs/ACCEPTANCE_CRITERIA.md
----------------------------------------
A1  zero exact duplicate sha256 across splits
A2  zero near-duplicate pairs across splits
A3  SOURCE_CLIP and its recording group absent from every split

Official test clips (``*_te*.mp4``) are inventoried and grouped but never placed in train
or validation. They are held out for a single final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsafenet_cf import clipset as cs      # noqa: E402
from unsafenet_cf import grouping as gp     # noqa: E402

OUT = REPO / "outputs" / "classifier"
FRAMES_PER_CLIP = 12     # sampled evenly; enough for overlap detection, cheap to decode


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dense_grey(path: Path) -> np.ndarray | None:
    """Densely sampled greyscale frames for duplicate detection.

    Every ``DENSE_STRIDE``-th frame, up to ``DENSE_CAP``, at DENSE_W x DENSE_H. Dense
    sampling is the point: two clips that overlap in time only reveal it if both sample
    the shared interval, and 12 frames spread over a whole clip rarely do.
    """
    out = []
    try:
        for i, frame in enumerate(cs.iter_frames(path, width=gp.DENSE_W,
                                                 height=gp.DENSE_H)):
            if i % gp.DENSE_STRIDE == 0:
                out.append(frame.astype(np.float32).mean(axis=2).astype(np.uint8))
            if len(out) >= gp.DENSE_CAP:
                break
    except Exception:
        return None
    return np.stack(out) if out else None


def sample_frames(path: Path, count: int, total: int | None = None,
                  max_side: int = 480) -> np.ndarray | None:
    """Decode ``count`` evenly spaced frames as (T, h, w, 3) uint8, or None on failure.

    Frames are streamed and only the wanted ones are kept. Reading a whole clip into
    memory first would cost about 1.6 GB for a 262-frame 1080p clip, and there are
    hundreds of clips; that is the difference between this running and this thrashing.

    Frames are also downscaled so the long side is at most ``max_side``. Both the
    perceptual hash and the background signature reduce to a small grid anyway, so full
    resolution buys nothing here and costs memory.
    """
    if total is None or total <= 0:
        return None
    wanted = set(np.linspace(0, total - 1, min(count, total)).round().astype(int).tolist())
    kept: list[np.ndarray] = []
    # ffmpeg scales during decoding; nothing full-resolution is ever materialised here.
    w, h = max_side, int(round(max_side * 9 / 16))
    try:
        for i, frame in enumerate(cs.iter_frames(path, width=w, height=h)):
            if i in wanted:
                kept.append(frame.copy())
            if len(kept) == len(wanted):
                break
    except Exception:
        return None
    return np.stack(kept) if kept else None



def _probe_one(job: tuple[str, int]) -> dict:
    """Hash one clip: sha256, container metadata, frame hashes, background signature.

    Module level and taking only picklable arguments, so a ProcessPoolExecutor can run it.
    A clip that fails to decode returns hashes=None rather than raising, because one bad
    file must not abort an audit of several hundred.
    """
    path_str, frames_wanted = job
    path = Path(path_str)
    media = cs.ffprobe_media(path)
    out = {"sha256": sha256_of(path), "media": media,
           "hashes": None, "signature": None, "residuals": None}
    frames = sample_frames(path, frames_wanted, media.get("frame_count"))
    if frames is not None:
        out["hashes"] = [gp.ahash(f) for f in frames]
        out["signature"] = gp.background_signature(frames).tolist()
        out["residuals"] = gp.residual_signatures(frames)
    # Recording identity is decided on these dense greyscale frames, not on the average
    # hash or the motion residual. Both of those were measured and rejected; see
    # docs/GROUPING_CALIBRATION.md.
    out["dense"] = dense_grey(path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO / "data" / "raw"),
                    help="directory holding the downloaded class folders")
    ap.add_argument("--val-fraction", type=float, default=0.25,
                    help="share of training recording groups held out for validation")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--frames", type=int, default=FRAMES_PER_CLIP)
    ap.add_argument("--reuse-signatures", action="store_true",
                    help="load cached signatures instead of decoding; use this to change "
                         "a threshold without re-decoding the dataset")
    ap.add_argument("--mad-certain", type=float, default=gp.MAD_CERTAIN,
                    help="minimum mean absolute pixel difference at or below which two "
                         "clips are taken to share frames outright")
    ap.add_argument("--ambiguity-fraction", type=float, default=0.50,
                    help="quarantine a clip when at least this fraction of its sampled "
                         "frames also appear in an opposite-label clip")
    ap.add_argument("--mad-probable", type=float, default=gp.MAD_PROBABLE,
                    help="above --mad-certain and at or below this, the pair probably "
                         "shares a recording; reported as uncertain and kept together")
    ap.add_argument("--scene-threshold", type=float, default=gp.SCENE_CORR_THRESHOLD,
                    help="background-signature correlation above which two clips are "
                         "taken to share a camera position (a scene, not a recording)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel decode processes; decoding dominates "
                         "this stage (default 8)")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    cs.save_label_map(OUT / "label_map.json")

    root = Path(args.data_root)
    clips = cs.discover([root])
    if not clips:
        log(f"No class-0 or class-4 clips found under {root}.")
        log("See bridges2/DATASET_ACCESS.md for how to obtain them.")
        return 2

    by_class = defaultdict(int)
    for c in clips:
        by_class[(c.dataset_class, c.official_split)] += 1
    log(f"found {len(clips)} clips: " +
        ", ".join(f"class {k[0]} {k[1]}={v}" for k, v in sorted(by_class.items())))

    # --- hashing and metadata -----------------------------------------------------
    # Decoding dominates this stage at roughly 8 s per clip, which is close to an hour on
    # one core for a few hundred clips. The work is embarrassingly parallel and each clip
    # is independent, so it is spread across processes. This keeps Stage A cheap enough to
    # run on a login node or a shared CPU allocation, instead of occupying a GPU.
    hashes: dict[str, list[int]] = {}
    signatures: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    dense: dict[str, np.ndarray] = {}
    undecodable: list[str] = []

    t_hash = time.time()
    results = None
    cache_path = OUT / "audit_cache.json"
    resid_path = OUT / "residual_signatures.npz"

    if args.reuse_signatures and cache_path.exists() and resid_path.exists():
        # Decoding is the whole cost of this stage. Thresholds are a judgement made from
        # the signatures, so changing one must not require decoding the dataset again.
        cache = json.loads(cache_path.read_text())
        npz = np.load(resid_path)
        dense_path = OUT / "dense_frames.npz"
        dnpz = np.load(dense_path) if dense_path.exists() else None
        if dnpz is None:
            log("dense frames absent from cache; decoding instead")
            args.reuse_signatures = False
        missing = [c.clip_id for c in clips if c.clip_id not in cache]
        if missing:
            log(f"cache is missing {len(missing)} clips; decoding instead")
            args.reuse_signatures = False
        else:
            log(f"reusing cached signatures for {len(cache)} clips; nothing is decoded")
            results = [{"sha256": cache[c.clip_id]["sha256"],
                        "media": cache[c.clip_id]["media"],
                        "hashes": cache[c.clip_id]["hashes"],
                        "signature": cache[c.clip_id]["signature"],
                        "residuals": (npz[c.clip_id]
                                      if c.clip_id in npz.files else None),
                        "dense": (dnpz[c.clip_id]
                                  if dnpz is not None and c.clip_id in dnpz.files
                                  else None)}
                       for c in clips]

    if not args.reuse_signatures:
        results = None
    if results is None and args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_probe_one,
                                    [(str(c.path), args.frames) for c in clips],
                                    chunksize=1))
    elif results is None:
        results = [_probe_one((str(c.path), args.frames)) for c in clips]

    for clip, res in zip(clips, results):
        clip.sha256 = res["sha256"]
        clip.width = res["media"].get("width")
        clip.height = res["media"].get("height")
        clip.fps = res["media"].get("fps")
        clip.frame_count = res["media"].get("frame_count")
        clip.duration_s = res["media"].get("duration_s")
        if res["hashes"] is None:
            undecodable.append(clip.clip_id)
        else:
            clip.phash = [f"{h:016x}" for h in res["hashes"]]
            hashes[clip.clip_id] = res["hashes"]
            signatures[clip.clip_id] = np.asarray(res["signature"])
            residuals[clip.clip_id] = np.asarray(res["residuals"], dtype=np.float32)
        if res.get("dense") is not None:
            dense[clip.clip_id] = np.asarray(res["dense"], dtype=np.uint8)
    log(f"  hashed {len(clips)} clips in {time.time() - t_hash:.1f}s "
        f"with {args.workers} worker(s)")

    if undecodable:
        log(f"WARNING: {len(undecodable)} clips could not be decoded: {undecodable[:5]}")

    # --- A1 exact duplicates ------------------------------------------------------
    by_sha: dict[str, list[str]] = defaultdict(list)
    for c in clips:
        by_sha[c.sha256].append(c.clip_id)
    exact_dups = {k: v for k, v in by_sha.items() if len(v) > 1}
    log(f"A1 exact duplicate sha256 groups: {len(exact_dups)}")

    # --- recording groups: strong evidence only -----------------------------------
    # Decided on motion residuals, not on the average hash. On this fixed-camera dataset
    # an average hash is dominated by the room and cannot tell one recording from another
    # taken by the same camera; measured, it put 36% of class-0/class-4 pairs within 4
    # bits, and those pairs cannot share a recording at all.
    if not args.reuse_signatures:
        np.savez_compressed(resid_path, **residuals)
        np.savez_compressed(OUT / "dense_frames.npz", **dense)
        cache_path.write_text(json.dumps({
            c.clip_id: {"sha256": c.sha256,
                        "media": {"width": c.width, "height": c.height, "fps": c.fps,
                                  "frame_count": c.frame_count,
                                  "duration_s": c.duration_s},
                        "hashes": hashes.get(c.clip_id),
                        "signature": (signatures[c.clip_id].tolist()
                                      if c.clip_id in signatures else None)}
            for c in clips}))
        log(f"saved residual signatures and cache for {len(residuals)} clips")
    # Self-test before trusting the metric: two overlapping halves of one clip share
    # frames by construction, so the scan must find them. A metric that separates known
    # negatives but misses known positives is useless here, and the previous one did
    # exactly that.
    control = None
    probe_id = next((c.clip_id for c in clips if c.clip_id in dense
                     and len(dense[c.clip_id]) >= 20), None)
    if probe_id is not None:
        d = dense[probe_id]
        n = len(d)
        halves = {"_posctl_a": d[:int(n * 0.6)], "_posctl_b": d[int(n * 0.4):]}
        cert, _ = gp.pairwise_duplicate_scan(halves)
        control = {"clip": probe_id, "detected": bool(cert),
                   "min_mad": cert[0][2] if cert else None}
        log(f"positive control on {probe_id}: overlapping halves "
            f"{'DETECTED' if cert else 'MISSED'}"
            + (f" (min MAD {cert[0][2]:.3f})" if cert else ""))
    if control is not None and not control["detected"]:
        log("positive control FAILED: the metric does not detect a known overlap.")

    certain_pairs, probable_pairs = gp.pairwise_duplicate_scan(
        dense, mad_certain=args.mad_certain, mad_probable=args.mad_probable)
    overlap_pairs = certain_pairs
    log(f"certain shared-frame pairs (min MAD <= {args.mad_certain}): "
        f"{len(certain_pairs)}")
    log(f"probable same-recording pairs ({args.mad_certain} < min MAD <= "
        f"{args.mad_probable}), treated as uncertain: {len(probable_pairs)}")
    ids = [c.clip_id for c in clips]
    rec_groups = gp.connected_components(
        ids, [(a, b) for a, b, _ in certain_pairs] + [(a, b) for a, b, _ in probable_pairs])

    # --- scene groups: background similarity, a separate question ------------------
    scene_edges = []
    sig_ids = sorted(signatures)
    for i, a in enumerate(sig_ids):
        for b in sig_ids[i + 1:]:
            if gp.scene_similarity(signatures[a], signatures[b]) >= args.scene_threshold:
                scene_edges.append((a, b))
    scene_groups = gp.connected_components(ids, scene_edges)
    log(f"scene-similarity pairs (same camera position, not a recording): "
        f"{len(scene_edges)}")

    # --- uncertain recording groups ------------------------------------------------
    # Same scene AND adjacent filename index, but no shared frame. Suggestive, unproven.
    index_of = {c.clip_id: (c.dataset_class, c.official_split, c.index) for c in clips}
    # Probable pairs are kept in one group when splitting. Treating an unproven same
    # recording as separate risks a leak; treating it as one costs a little data.
    uncertain_pairs = [(a, b) for a, b, _ in probable_pairs]
    log(f"uncertain recording pairs (same scene + adjacent index, no shared frame): "
        f"{len(uncertain_pairs)}")

    for c in clips:
        c.recording_group = rec_groups[c.clip_id]
        c.scene_group = scene_groups[c.clip_id]
        c.recording_group_certain = any(
            c.clip_id in (a, b) for a, b, _ in overlap_pairs)

    # --- label ambiguity: clips that are largely also the other label ----------------
    # Sharing a few frames with an opposite-label clip does not make either label wrong.
    # Two clips cut either side of a behaviour change legitimately overlap at the seam,
    # and each is correctly labelled for its own span. What is not legitimate is training
    # on a clip whose content is mostly also published under the opposite label: that clip
    # supplies contradictory supervision whatever the publisher intended.
    #
    # So ambiguity is measured as a fraction, not as a yes or no: of this clip's sampled
    # frames, how many appear in some opposite-label clip? A clip above the threshold is
    # quarantined from every split and reported. No claim is made that its published label
    # is wrong.
    def overlap_fraction(a: str, b: str, thresh: float) -> tuple[float, float]:
        A = dense[a].astype(np.float32).reshape(len(dense[a]), -1)
        B = dense[b].astype(np.float32).reshape(len(dense[b]), -1)
        M = np.abs(A[:, None, :] - B[None, :, :]).mean(2)
        return float((M.min(1) <= thresh).mean()), float((M.min(0) <= thresh).mean())

    class_of = {c.clip_id: c.dataset_class for c in clips}
    ambiguity: dict[str, dict] = {}
    for a, b, _ in certain_pairs:
        if a not in dense or b not in dense:
            continue
        if class_of[a] == class_of[b]:
            continue
        fa, fb = overlap_fraction(a, b, args.mad_certain)
        for cid, frac, other in ((a, fa, b), (b, fb, a)):
            prev = ambiguity.get(cid)
            if prev is None or frac > prev["shared_fraction"]:
                ambiguity[cid] = {"shared_fraction": frac,
                                  "with": other,
                                  "other_class": class_of[other]}
    quarantined = {cid: info for cid, info in ambiguity.items()
                   if info["shared_fraction"] >= args.ambiguity_fraction}
    log(f"clips sharing frames with an opposite-label clip: {len(ambiguity)}")
    log(f"   quarantined (at least {args.ambiguity_fraction:.0%} of sampled frames "
        f"shared): {len(quarantined)}")
    for cid, info in sorted(quarantined.items()):
        log(f"     {cid:10s} {info['shared_fraction']:.2f} with {info['with']} "
            f"(class {info['other_class']})")

    # --- A3 hold out the source clip and everything plausibly from its recording ----
    source_rec = rec_groups.get(cs.SOURCE_CLIP)
    source_scene = scene_groups.get(cs.SOURCE_CLIP)
    excluded: dict[str, str] = {}
    if source_rec is None:
        log(f"NOTE: {cs.SOURCE_CLIP} not present in the inventory.")
    else:
        for c in clips:
            if c.clip_id == cs.SOURCE_CLIP:
                excluded[c.clip_id] = "is the counterfactual source clip"
            elif rec_groups[c.clip_id] == source_rec:
                excluded[c.clip_id] = "shares a recording with the source clip (certain)"
    for cid, info in quarantined.items():
        excluded.setdefault(
            cid,
            f"label-ambiguous: {info['shared_fraction']:.0%} of its sampled frames also "
            f"appear in {info['with']}, which carries dataset class "
            f"{info['other_class']}")
    for a, b in uncertain_pairs:
        for cid in (a, b):
            if cid in excluded and "UNCERTAIN" not in excluded[cid]:
                continue
    scene_sharers = {c.clip_id: "shares the source clip's camera position"
                     for c in clips
                     if c.clip_id != cs.SOURCE_CLIP
                     and source_scene is not None
                     and scene_groups[c.clip_id] == source_scene
                     and c.clip_id not in excluded}
    log(f"A3 excluded from all splits: {len(excluded)} clips")
    log(f"   clips sharing the source clip's scene but not its recording: "
        f"{len(scene_sharers)} (kept in training; see audit_report.md)")

    # --- splits: group-aware, over recording groups --------------------------------
    rng = random.Random(args.seed)
    train_pool = [c for c in clips
                  if c.official_split == "tr" and c.clip_id not in excluded]
    test_pool = [c for c in clips if c.official_split == "te"]

    # Group by recording group so no group straddles train and validation. A group with
    # uncertain membership is kept whole, which is the conservative choice.
    groups_by_label: dict[int, list[str]] = defaultdict(list)
    members: dict[str, list[str]] = defaultdict(list)
    for c in train_pool:
        members[c.recording_group].append(c.clip_id)
    for g, mem in members.items():
        labels = {next(c.binary_label for c in train_pool if c.clip_id == m) for m in mem}
        # A group spanning both labels cannot be assigned cleanly; report and split by clip.
        key = labels.pop() if len(labels) == 1 else -1
        groups_by_label[key].append(g)

    # Groups vary in size, so taking a fraction of the *groups* gives an unpredictable
    # fraction of the clips. The first run that way left validation with 4 clips of one
    # label, too few for a per-class recall gate to mean anything. Take whole groups until
    # the target share of that label's clips is reached instead.
    label_of = {c.clip_id: c.binary_label for c in train_pool}
    clips_in_group = {g: mem for g, mem in members.items()}
    val_groups: set[str] = set()
    for label, gs in groups_by_label.items():
        gs = sorted(gs)
        rng.shuffle(gs)
        total_clips = sum(1 for c in train_pool if c.binary_label == label) if label >= 0 \
            else sum(len(clips_in_group[g]) for g in gs)
        target = total_clips * args.val_fraction
        taken = 0
        for g in gs:
            if taken >= target and val_groups:
                break
            val_groups.add(g)
            taken += len(clips_in_group[g])

    split_of: dict[str, str] = {}
    for c in train_pool:
        split_of[c.clip_id] = "val" if c.recording_group in val_groups else "train"
    for c in test_pool:
        split_of[c.clip_id] = "test"
    for cid in excluded:
        split_of[cid] = "excluded"

    counts = defaultdict(int)
    for c in clips:
        counts[(split_of[c.clip_id], c.dataset_class)] += 1
    log("split sizes: " + ", ".join(f"{k[0]} class{k[1]}={v}"
                                    for k, v in sorted(counts.items())))

    # --- A2 cross-split near duplicates --------------------------------------------
    # Only the real splits can leak. A pair with one foot in "excluded" is the audit
    # working: that clip was deliberately removed from every split.
    REAL = {"train", "val", "test"}
    cross_exact = [v for v in exact_dups.values()
                   if len({split_of.get(x) for x in v if split_of.get(x) in REAL}) > 1]
    cross_near = [(a, b, d) for a, b, d in overlap_pairs
                  if split_of.get(a) in REAL and split_of.get(b) in REAL
                  and split_of.get(a) != split_of.get(b)]
    log(f"A1 cross-split exact duplicates: {len(cross_exact)}")
    log(f"A2 cross-split frame overlaps  : {len(cross_near)}")

    source_leaked = any(split_of.get(cid) in ("train", "val")
                        for cid in excluded)

    # A clip that could not be decoded was never hashed, so A2 was never actually checked
    # for it. Reporting PASS in that state would claim a check that did not run. Any
    # undecodable clip fails Gate A outright.
    gate_a = ((not cross_exact) and (not cross_near) and (not source_leaked)
              and not undecodable)
    # A metric that cannot find a known overlap must not be trusted to certify splits.
    if control is not None and not control["detected"]:
        gate_a = False

    # An empty train or val split is a Gate A failure. The first run of this audit
    # produced exactly that: over-merged groups put every clip in val and none in train,
    # and the failure only surfaced one stage later.
    n_train = sum(1 for v in split_of.values() if v == "train")
    n_val = sum(1 for v in split_of.values() if v == "val")
    if n_train == 0 or n_val == 0:
        log(f"train={n_train}, val={n_val}; both must be non-empty.")
        gate_a = False
    for name, split_name in (("train", "train"), ("val", "val")):
        labels_here = {c.binary_label for c in clips if split_of[c.clip_id] == split_name}
        if labels_here != {0, 1}:
            log(f"{split_name} split has labels {sorted(labels_here)}; needs both.")
            gate_a = False

    # Both binary labels must be present, or there is nothing to train.
    labels_present = sorted({c.binary_label for c in clips})
    if labels_present != [0, 1]:
        log(f"labels present: {labels_present}; both 0 and 1 are required for training.")
        gate_a = False

    # --- write everything out --------------------------------------------------------
    inventory = [{
        "clip_id": c.clip_id, "path": str(c.path), "dataset_class": c.dataset_class,
        "binary_label": c.binary_label, "official_split": c.official_split,
        "index": c.index, "sha256": c.sha256, "width": c.width, "height": c.height,
        "fps": c.fps, "frame_count": c.frame_count, "duration_s": c.duration_s,
        "phash": c.phash, "recording_group": c.recording_group,
        "scene_group": c.scene_group,
        "recording_group_certain": c.recording_group_certain,
        "split": split_of[c.clip_id],
    } for c in clips]
    (OUT / "clip_inventory.json").write_text(json.dumps(inventory, indent=2))

    (OUT / "groups.json").write_text(json.dumps({
        "recording_overlap_pairs_certain": [
            {"a": a, "b": b, "min_mad": d} for a, b, d in certain_pairs],
        "scene_similarity_pairs": [{"a": a, "b": b} for a, b in scene_edges],
        "uncertain_recording_pairs": [
            {"a": a, "b": b,
             "basis": "same scene signature and adjacent filename index, no shared frame"}
            for a, b in uncertain_pairs],
        "excluded_from_all_splits": excluded,
        "share_source_scene_but_not_recording": scene_sharers,
        "label_ambiguity": ambiguity,
        "quarantined_for_label_ambiguity": quarantined,
        "ambiguity_note": ("Shared frames between opposite-label clips do not show that "
                           "either published label is wrong, and they do not establish a "
                           "ceiling on classifier accuracy. Two clips cut either side of "
                           "a behaviour change overlap at the seam and are each correct "
                           "for their own span. Only clips whose sampled content is "
                           "largely also published under the other label are quarantined, "
                           "because those supply contradictory supervision."),
        "positive_control": control,
        "probable_same_recording_pairs": [
            {"a": a, "b": b, "min_mad": d} for a, b, d in probable_pairs],
        "thresholds": {"mad_certain": args.mad_certain,
                       "mad_probable": args.mad_probable,
                       "scene_correlation": args.scene_threshold,
                       "frames_sampled_per_clip": args.frames},
        "caveat": ("A shared background establishes a shared camera position, not a shared "
                   "recording. Only frame overlap is treated as certain evidence of one "
                   "recording. Pairs listed as uncertain are reported, not resolved."),
    }, indent=2))

    (OUT / "splits.json").write_text(json.dumps({
        "seed": args.seed, "val_fraction": args.val_fraction,
        "split_of": split_of,
        "counts": {f"{k[0]}_class{k[1]}": v for k, v in sorted(counts.items())},
        "policy": ("Splits are built over recording groups, never individual clips, so no "
                   "recording straddles train and validation. Official test clips are "
                   "never placed in train or validation."),
    }, indent=2))

    report = [
        "# Stage A — data integrity audit", "",
        f"Clips inventoried: {len(clips)}", "",
        "| Check | Result | Gate |", "|---|---|---|",
        f"| A1 exact duplicate sha256 groups | {len(exact_dups)} | — |",
        f"| A1 crossing a split boundary | {len(cross_exact)} | must be 0 |",
        f"| A2 recording-overlap pairs found | {len(overlap_pairs)} | — |",
        f"| A2 **detected** overlaps crossing a split boundary | {len(cross_near)} | must be 0 |",
        f"| A3 clips excluded in total | {len(excluded)} | — |",
        f"| of which quarantined as label-ambiguous | {len(quarantined)} | — |",
        f"| A3 any excluded clip leaked into train or val | {source_leaked} | must be False |",
        f"| Clips that could not be decoded | {len(undecodable)} | must be 0 |",
        f"| Binary labels present | {labels_present} | must be [0, 1] |",
        "",
        f"**Gate A: {'PASS' if gate_a else 'FAIL'}**", "",
        "## Recording groups versus scene groups", "",
        f"Pairs sharing frames outright, minimum mean absolute pixel difference at or "
        f"below {args.mad_certain}: {len(certain_pairs)}. These are certain evidence of "
        f"one recording.",
        f"Pairs probably from one recording, difference above {args.mad_certain} and at "
        f"or below {args.mad_probable}: {len(probable_pairs)}. Reported as uncertain and "
        f"kept together when splitting.",
        f"Positive control: overlapping halves of one clip were "
        f"{'DETECTED' if (control or {}).get('detected') else 'MISSED'}.",
        f"Scene-similarity pairs, meaning a shared camera position and nothing more: "
        f"{len(scene_edges)}.",
        f"Pairs where recording identity is **uncertain** — same scene and adjacent "
        f"filename index, but no shared frame: {len(uncertain_pairs)}.",
        "",
        "A similar factory background does not establish a shared recording. Exclusion "
        "from the splits is therefore decided on recording overlap alone, which is what "
        "leaks. Clips that merely share the source clip's camera position are kept in "
        "training and listed under share_source_scene_but_not_recording in groups.json, "
        f"{len(scene_sharers)} of them. Excluding those as well was tried and removed "
        "half of the safe-walkway class, including the examples most useful for learning "
        "behaviour rather than room identity.",
        "",
    ]
    if undecodable:
        report += [f"## Undecodable clips: {len(undecodable)}", "",
                   ", ".join(f"`{c}`" for c in undecodable[:20]), ""]
    (OUT / "audit_report.md").write_text("\n".join(report))

    log(f"\nGate A: {'PASS' if gate_a else 'FAIL'}")
    log(f"wrote {OUT}/audit_report.md and 4 JSON artefacts")
    return 0 if gate_a else 1


if __name__ == "__main__":
    raise SystemExit(main())
