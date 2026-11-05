#!/usr/bin/env python3
"""Stage B: probe, then train and validate the binary walkway classifier.

Gate order, from docs/ACCEPTANCE_CRITERIA.md and the approved execution plan:

``--mode probe``   Measure peak GPU memory and per-step time for the real training step at
                   the real batch size and input shape. Tensor shapes fitting in a forward
                   pass does not establish that training fits; this measures it. Writes a
                   manifest and exits without training.
``--mode train``   Train, selecting the checkpoint by validation balanced accuracy. The
                   test split is never read by this script.

Clips are decoded once into a cached uint8 tensor per clip, because decoding dominates and
the same clips are read every epoch. The cache lives beside the outputs and is keyed by the
clip's sha256, so a changed file can never be served from a stale cache entry.

The classifier operates on exactly the preprocessing in ``unsafenet_cf.classifier.prepare``,
which is also what the guidance loop uses. They cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsafenet_cf import classifier as C    # noqa: E402
from unsafenet_cf import clipset as cs      # noqa: E402
from unsafenet_cf import manifest as mf     # noqa: E402

OUT = REPO / "outputs" / "classifier"
CACHE = OUT / "clip_cache"

# Every clip whose declared frame count exceeded what could be decoded, with the exact
# substitution made. Written to disk so the record survives the run rather than scrolling
# past in a log.
SUBSTITUTION_LOG: list[dict] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def load_inventory() -> list[dict]:
    path = OUT / "clip_inventory.json"
    if not path.exists():
        raise SystemExit(f"{path} missing. Run scripts/audit_splits.py first.")
    return json.loads(path.read_text())


def decode_cached(entry: dict, num_frames: int, height: int, width: int) -> np.ndarray:
    """Decode a clip to (T, h, w, 3) uint8, caching the result keyed by sha256."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{entry['clip_id']}_{entry['sha256'][:12]}_{num_frames}x{height}x{width}.npy"
    cached = CACHE / key
    if cached.exists():
        side = cached.with_suffix(".subs.json")
        if side.exists():
            SUBSTITUTION_LOG.append(json.loads(side.read_text()))
        return np.load(cached)

    # Stream and keep only the wanted frames. A 262-frame 1080p clip held whole is about
    # 1.6 GB, and every clip in the split would be decoded that way.
    total = entry.get("frame_count")
    if not total:
        raise ValueError(f"{entry['clip_id']}: frame_count missing from the inventory; "
                         "re-run scripts/audit_splits.py")
    wanted = np.linspace(0, total - 1, num_frames).round().astype(int)
    want_set = set(wanted.tolist())
    by_index: dict[int, np.ndarray] = {}
    for i, frame in enumerate(cs.iter_frames(Path(entry["path"]),
                                             width=width, height=height)):
        if i in want_set:
            by_index[i] = frame.copy()
        if len(by_index) == len(want_set):
            break
    if not by_index:
        raise ValueError(f"{entry['clip_id']}: no frames could be decoded")
    if len(by_index) < len(want_set):
        # frame_count comes from fps x duration and is an estimate, so the last wanted
        # index can sit past the real end of the clip. Substitute the nearest frame that
        # was actually decoded from the same clip. The substitute is always a real,
        # temporally adjacent frame of that clip; no frame from elsewhere is ever used.
        available = sorted(by_index)
        last_real = available[-1]
        subs = []
        for i in wanted:
            if i not in by_index:
                nearest = min(available, key=lambda j: abs(j - int(i)))
                by_index[i] = by_index[nearest]
                subs.append({"wanted_index": int(i), "used_index": int(nearest),
                             "distance_frames": int(abs(int(i) - nearest))})
        record = {"clip_id": entry["clip_id"],
                  "declared_frame_count": int(total),
                  # Wanted indices that were actually found, not the clip's frame count.
                  "wanted_indices_found": len(available),
                  "wanted_indices_total": len(want_set),
                  "last_real_index": int(last_real),
                  "substitutions": subs}
        log(f"  frame substitution in {entry['clip_id']}: declared {total} frames, "
            f"decoded {len(available)}, substituted {len(subs)} index(es) "
            f"with the nearest real frame (max distance "
            f"{max(x['distance_frames'] for x in subs)})")
        SUBSTITUTION_LOG.append(record)
    # Repeat an index if the clip is shorter than num_frames, preserving order.
    out = np.stack([by_index[i] for i in wanted])
    np.save(cached, out)
    if SUBSTITUTION_LOG and SUBSTITUTION_LOG[-1].get("clip_id") == entry["clip_id"]:
        cached.with_suffix(".subs.json").write_text(json.dumps(SUBSTITUTION_LOG[-1]))
    return out


def _decode_job(job):
    """Module-level wrapper so a process pool can build the decode cache in parallel."""
    entry, num_frames, height, width = job
    decode_cached(entry, num_frames, height, width)
    return entry["clip_id"]


def collect_substitutions(entries: list[dict], num_frames: int, height: int,
                          width: int) -> list[dict]:
    """Gather substitution records written by the cache builder, including by workers.

    A process pool has its own memory, so the in-process list is not shared. Each cached
    clip carries its record in a side file, which is the authoritative record.
    """
    found = []
    for e in entries:
        key = (f"{e['clip_id']}_{e['sha256'][:12]}_"
               f"{num_frames}x{height}x{width}.npy")
        side = (CACHE / key).with_suffix(".subs.json")
        if side.exists():
            found.append(json.loads(side.read_text()))
    return found


def prebuild_cache(entries: list[dict], num_frames: int, height: int, width: int,
                   workers: int) -> None:
    """Decode every clip once, in parallel, before training touches the GPU.

    Decoding is the slow part and it is embarrassingly parallel, while training is not.
    Doing it serially inside the first epoch wasted most of a GPU allocation on ffmpeg.
    """
    todo = [(e, num_frames, height, width) for e in entries]
    if workers <= 1:
        for job in todo:
            _decode_job(job)
        return
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(_decode_job, todo, chunksize=1), start=1):
            if i % 50 == 0 or i == len(todo):
                log(f"  decoded {i}/{len(todo)}")
    log(f"  cache built in {time.time() - t0:.1f}s with {workers} workers")


class ClipTensors:
    """Decoded clips for one split, held in memory as uint8."""

    def __init__(self, entries: list[dict], num_frames: int, height: int, width: int):
        self.entries = entries
        self.data = [decode_cached(e, num_frames, height, width) for e in entries]
        self.labels = np.array([e["binary_label"] for e in entries], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.entries)

    def batch(self, idx: np.ndarray, device, augment: bool = False):
        import torch
        arr = np.stack([self.data[i] for i in idx]).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).to(device)              # (B, T, h, w, 3) in [0, 1]
        if augment:
            # Horizontal flip only. The label is about a worker's position relative to a
            # painted lane, and a flip preserves that relation while changing the pixels.
            # No colour jitter: the lane's green is part of the geometry being judged, and
            # recolouring it would teach the model to ignore the very cue it must read.
            flip = torch.rand(x.shape[0], device=device) < 0.5
            x[flip] = torch.flip(x[flip], dims=[3])
        y = torch.from_numpy(self.labels[idx]).to(device)
        return x, y


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in (0, 1):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    m = [[0, 0], [0, 0]]
    for t, p in zip(y_true, y_pred):
        m[int(t)][int(p)] += 1
    return m


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for cls in (0, 1):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[cs.BINARY_NAMES[cls]] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int((y_true == cls).sum()),
        }
    return out



def evaluate(model, data: ClipTensors, device, batch_size: int) -> dict:
    """Validation pass. Returns metrics plus raw probabilities for later diagnostics."""
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(start + batch_size, len(data)))
            x, _ = data.batch(idx, device)
            logits = model(C.prepare(x, *PREP))
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, 2))
    y_pred = probs.argmax(axis=1)
    y_true = data.labels
    return {
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "confusion_matrix_true_by_pred": confusion(y_true, y_pred),
        "per_class": per_class_metrics(y_true, y_pred),
        "probabilities": probs.tolist(),
        "clip_ids": [e["clip_id"] for e in data.entries],
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


PREP = (C.DEFAULT_FRAMES, C.DEFAULT_HEIGHT, C.DEFAULT_WIDTH)


def main(argv=None) -> int:
    global PREP
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["probe", "train"])
    ap.add_argument("--frames", type=int, default=C.DEFAULT_FRAMES)
    ap.add_argument("--height", type=int, default=C.DEFAULT_HEIGHT)
    ap.add_argument("--width", type=int, default=C.DEFAULT_WIDTH)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--probe-steps", type=int, default=5)
    ap.add_argument("--amp", action="store_true", default=True,
                    help="bfloat16 autocast for training; native on sm_89")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--decode-workers", type=int, default=8,
                    help="parallel processes used to build the decode cache")
    args = ap.parse_args(argv)
    PREP = (args.frames, args.height, args.width)

    if not torch.cuda.is_available():
        log("No CUDA device. This script must run on a GPU node, never on a login node.")
        return 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    # The L40S is sm_89 and supports bfloat16 natively, so autocast is a real speedup
    # rather than emulation. Measured in fp32: 4.321 s per training step at batch 8 and
    # 16x240x320, which is 41 minutes of pure compute over 30 epochs before validation.
    # Weights stay fp32; only the forward and backward arithmetic is reduced.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if args.amp else None
    OUT.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory()
    by_split = defaultdict(list)
    for e in inventory:
        by_split[e["split"]].append(e)
    log("split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_split.items())))

    if not by_split["train"] or not by_split["val"]:
        log("train or val split is empty; cannot proceed. Check outputs/classifier/splits.json.")
        return 2
    labels_present = {e["binary_label"] for e in by_split["train"]}
    if labels_present != {0, 1}:
        log(f"training split contains only labels {sorted(labels_present)}; both are "
            "required. The class-4 clips are probably missing: see "
            "bridges2/DATASET_ACCESS.md.")
        return 2

    model = C.build_model(num_classes=2, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---------------------------------------------------------------- probe mode
    if args.mode == "probe":
        log(f"probe: {args.probe_steps} real training steps at batch {args.batch_size}, "
            f"input {args.frames}x{args.height}x{args.width}")
        mf.reset_peak_memory()
        x = torch.rand(args.batch_size, args.frames, args.height, args.width, 3,
                       device=device)
        y = torch.randint(0, 2, (args.batch_size,), device=device)
        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.probe_steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                loss = loss_fn(model(C.prepare(x, *PREP)), y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        step_s = (time.time() - t0) / args.probe_steps
        peak = mf.peak_memory_gb()
        log(f"peak GPU {peak} GiB, {step_s:.3f} s per training step")

        # Reduced precision must be verified on the card that will run it, not assumed.
        # bfloat16 has the dynamic range of float32 but far fewer mantissa bits, and a
        # loss or gradient that quietly becomes NaN or Inf would train a model that looks
        # fine in the log and is worthless.
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
            logits = model(C.prepare(x, *PREP))
            loss = loss_fn(logits, y)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)))
        numeric = {
            "amp_enabled": bool(args.amp),
            "amp_dtype": str(amp_dtype),
            "logits_dtype": str(logits.dtype),
            "loss_value": float(loss),
            "loss_finite": bool(torch.isfinite(loss)),
            "params_with_grad": len(grads),
            "params_total": sum(1 for _ in model.parameters()),
            "all_grads_finite": all(bool(torch.isfinite(g).all()) for g in grads),
            "grad_global_norm": total_norm,
            "grad_norm_nonzero": total_norm > 0.0,
            "device": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "bf16_native": bool(torch.cuda.get_device_capability(0) >= (8, 0)),
        }
        log(f"BF16 check on {numeric['device']} (sm_{numeric['capability'].replace('.', '')}): "
            f"loss {numeric['loss_value']:.6f} finite={numeric['loss_finite']}, "
            f"grads finite={numeric['all_grads_finite']}, "
            f"global norm {total_norm:.6g} over {numeric['params_with_grad']} tensors")
        if not (numeric["loss_finite"] and numeric["all_grads_finite"]
                and numeric["grad_norm_nonzero"]):
            log("BF16 numeric check FAILED; refusing to train in reduced precision.")
            mf.write_manifest(
                mf.build_run_manifest(
                    stage="classifier_train_probe",
                    environment=mf.environment_snapshot(),
                    settings={"batch_size": args.batch_size, "amp": args.amp},
                    inputs={}, outputs={"numeric_check": numeric},
                    timing={"s_per_training_step": step_s},
                    memory={"peak_gpu_gib": peak},
                    notes="BF16 numeric check failed. Training was not started.",
                ),
                OUT / "probe_train.json")
            return 1
        mf.write_manifest(
            mf.build_run_manifest(
                stage="classifier_train_probe",
                environment=mf.environment_snapshot(),
                settings={"batch_size": args.batch_size, "frames": args.frames,
                          "height": args.height, "width": args.width,
                          "model": "r3d_18", "weights": "KINETICS400_V1",
                          "optimizer": "AdamW", "lr": args.lr},
                inputs={"note": "random tensors of the real shape; no clips decoded"},
                outputs={"numeric_check": numeric},
                timing={"s_per_training_step": step_s},
                memory={"peak_gpu_gib": peak},
                notes=("Memory and time probe only. Measures a real forward, backward and "
                       "optimiser step at the real shape. No training was performed and no "
                       "checkpoint was written."),
            ),
            OUT / "probe_train.json")
        log(f"wrote {OUT}/probe_train.json")
        return 0

    # ---------------------------------------------------------------- train mode
    log("building decode cache for train and val")
    prebuild_cache(by_split["train"] + by_split["val"],
                   args.frames, args.height, args.width, args.decode_workers)
    train = ClipTensors(by_split["train"], args.frames, args.height, args.width)
    val = ClipTensors(by_split["val"], args.frames, args.height, args.width)
    log(f"train {len(train)} clips, val {len(val)} clips")

    subs = collect_substitutions(by_split["train"] + by_split["val"],
                                 args.frames, args.height, args.width)
    (OUT / "frame_substitutions.json").write_text(json.dumps(subs, indent=2))
    if subs:
        log(f"clips needing frame substitution: {len(subs)} "
            f"({', '.join(r['clip_id'] for r in subs[:8])}"
            f"{', ...' if len(subs) > 8 else ''})")
    else:
        log("no clip needed frame substitution")

    counts = np.bincount(train.labels, minlength=2)
    log(f"train label counts: {counts.tolist()}")
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)
    weights = (weights / weights.sum() * 2).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    log(f"class weights: {weights.tolist()}")

    mf.reset_peak_memory()
    best = {"balanced_accuracy": -1.0, "epoch": -1}
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train))
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            x, y = train.batch(idx, device, augment=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                loss = loss_fn(model(C.prepare(x, *PREP)), y)
            loss.backward()
            opt.step()
            running += float(loss) * len(idx)
        train_loss = running / max(len(order), 1)

        metrics = evaluate(model, val, device, args.batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_balanced_accuracy": metrics["balanced_accuracy"],
                        "val_accuracy": metrics["accuracy"]})
        log(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
            f"val balanced acc {metrics['balanced_accuracy']:.4f}")

        # Checkpoint selection is on validation only. The test split is not read here.
        if metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = {"balanced_accuracy": metrics["balanced_accuracy"], "epoch": epoch,
                    "metrics": metrics}
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "preprocess": {"frames": args.frames, "height": args.height,
                                       "width": args.width},
                        "label_map": json.loads((OUT / "label_map.json").read_text()),
                        "val_balanced_accuracy": metrics["balanced_accuracy"]},
                       OUT / "classifier_best.pt")

    total_s = time.time() - t_start
    peak = mf.peak_memory_gb()
    log(f"\nbest epoch {best['epoch']}, val balanced accuracy "
        f"{best['balanced_accuracy']:.4f}, {total_s / 60:.1f} min, peak {peak} GiB")

    gate_b1 = best["balanced_accuracy"] >= 0.80
    recalls = {k: v["recall"] for k, v in best["metrics"]["per_class"].items()}
    gate_b2 = all(r >= 0.75 for r in recalls.values())
    log(f"B1 balanced accuracy >= 0.80 : {gate_b1} ({best['balanced_accuracy']:.4f})")
    log(f"B2 every per-class recall >= 0.75 : {gate_b2} ({recalls})")

    result = {
        "best_epoch": best["epoch"],
        "val_metrics": best["metrics"],
        "history": history,
        "gates": {"B1_balanced_accuracy_at_least_0.80": gate_b1,
                  "B2_each_recall_at_least_0.75": gate_b2},
        "note": ("Validation only. The held-out test split has not been read by this "
                 "script. Gate B3, original-versus-VAE consistency, is evaluated by "
                 "scripts/eval_classifier.py and is also required before Stage D."),
    }
    (OUT / "train_result.json").write_text(json.dumps(result, indent=2))

    mf.write_manifest(
        mf.build_run_manifest(
            stage="classifier_train",
            environment=mf.environment_snapshot(),
            settings={"model": "r3d_18", "weights": "KINETICS400_V1",
                      "frames": args.frames, "height": args.height, "width": args.width,
                      "batch_size": args.batch_size, "epochs": args.epochs,
                      "lr": args.lr, "weight_decay": args.weight_decay,
                      "seed": args.seed, "class_weights": weights.tolist(),
                      "augmentation": "horizontal flip only",
                      "amp_bfloat16": args.amp,
                      "tf32": True,
                      "checkpoint_selection": "validation balanced accuracy"},
            inputs={"train_clips": len(train), "val_clips": len(val),
                    "label_map": str(OUT / "label_map.json")},
            outputs={"checkpoint": str(OUT / "classifier_best.pt"),
                     "best_val_balanced_accuracy": best["balanced_accuracy"],
                     "gates": result["gates"]},
            timing={"total_s": total_s, "s_per_epoch": total_s / max(args.epochs, 1)},
            memory={"peak_gpu_gib": peak},
            notes=("Binary walkway classifier, dataset class 0 to index 0 and class 4 to "
                   "index 1. Trained on official training clips only, with the source "
                   "clip's recording and scene group excluded. The test split was not "
                   "read."),
        ),
        OUT / "manifest_train.json")
    log(f"wrote {OUT}/train_result.json and {OUT}/manifest_train.json")
    return 0 if (gate_b1 and gate_b2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
