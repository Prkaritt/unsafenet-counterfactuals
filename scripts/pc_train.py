#!/usr/bin/env python
"""Fine-tune R3D-18 for the verified class-2 against class-6 distinction.

This is a **new** classifier. The walkway classifier from the earlier stage is not reused,
not relabelled, and not loaded: it was trained to separate *Safe Walkway Violation* from
*Safe Walkway*, a different pair of classes with a different decision surface.

Label map, stated explicitly because a transposed map would silently invert every later
conclusion and every guidance gradient:

    dataset class 2, "Opened Panel Cover", unsafe  ->  binary index 0
    dataset class 6, "Closed Panel Cover", safe    ->  binary index 1

Binary index 1 is the counterfactual target. The map is written to disk beside the
checkpoint and read back by every script that consumes this model.

Three constraints that are not negotiable here:

* **Recording-group separation.** Two fixed cameras in one factory means near-duplicate
  footage. Clips are split by recording group, never individually, so no moment of footage
  appears in both training and validation. Groups come from the audit's measured frame
  overlap, not from filename adjacency.
* **Evaluation sources excluded.** Every selected source clip, every donor clip, and every
  clip sharing a recording group with any of them is held out of training and validation
  alike. A classifier that trained on the donor's footage cannot be asked whether the
  composite looks closed.
* **Identical preprocessing.** ``unsafenet_cf.classifier.prepare`` is the single
  preprocessing path for training, evaluation and, later, guidance. It is differentiable
  end to end so the same function can carry a gradient.

Class 6 has 19 training clips in the published dataset. That is the binding limitation on
this stage and it is reported with every number this script produces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsafenet_cf import classifier as C     # noqa: E402
from unsafenet_cf import clipset as cs       # noqa: E402
from unsafenet_cf import manifest as mf      # noqa: E402

OUT = REPO / "outputs" / "panel_cover"

DATASET_TO_BINARY = {2: 0, 6: 1}
BINARY_NAMES = {0: "opened_panel_cover", 1: "closed_panel_cover"}
DATASET_CLASS_NAMES = {2: "Opened Panel Cover", 6: "Closed Panel Cover"}
TARGET_BINARY_INDEX = 1


def log(m: str) -> None:
    print(m, flush=True)


def save_label_map(path: Path) -> dict:
    m = {
        "dataset_class_to_binary_index": {str(k): v for k, v in DATASET_TO_BINARY.items()},
        "binary_index_to_name": {str(k): v for k, v in BINARY_NAMES.items()},
        "dataset_class_names": {str(k): v for k, v in DATASET_CLASS_NAMES.items()},
        "target_binary_index": TARGET_BINARY_INDEX,
        "excluded_dataset_classes": [0, 1, 3, 4, 5, 7],
        "note": ("Binary index 1 is Closed Panel Cover, the counterfactual target. "
                 "Guidance maximises the log-probability at index 1. This map is NOT the "
                 "walkway classifier's map and the two models are not interchangeable."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2))
    return m


# --- data ------------------------------------------------------------------------


def decode_tensor(path: Path, frames: int, height: int, width: int) -> np.ndarray:
    """Decode one clip to (frames, H, W, 3) uint8, evenly sampled in time."""
    got = [f.copy() for f in cs.iter_frames(path, width=width, height=height)]
    if not got:
        raise RuntimeError(f"decoded 0 frames from {path}")
    arr = np.stack(got)
    idx = np.linspace(0, len(arr) - 1, frames).round().astype(int)
    return arr[idx]


def build_cache(entries: list[dict], cache_dir: Path, frames: int,
                height: int, width: int) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        dest = cache_dir / f"{e['clip_id']}_{frames}x{height}x{width}.npy"
        if not dest.exists():
            np.save(dest, decode_tensor(REPO / e["path"], frames, height, width))
        out[e["clip_id"]] = dest
        if i % 25 == 0 or i == len(entries):
            log(f"  cached {i}/{len(entries)} in {time.time() - t0:.0f}s")
    return out


def load_split(entries: list[dict], cache: dict[str, Path]):
    import torch
    x = np.stack([np.load(cache[e["clip_id"]]) for e in entries])
    y = np.array([DATASET_TO_BINARY[e["dataset_class"]] for e in entries])
    return (torch.from_numpy(x), torch.from_numpy(y),
            [e["clip_id"] for e in entries])


# --- metrics ---------------------------------------------------------------------


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    m = np.zeros((2, 2), int)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m.tolist()


def metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    recalls = {}
    for k in (0, 1):
        sel = y_true == k
        recalls[BINARY_NAMES[k]] = (round(float((y_pred[sel] == k).mean()), 4)
                                    if sel.any() else None)
    present = [v for v in recalls.values() if v is not None]
    return {
        "n": int(len(y_true)),
        "support": {BINARY_NAMES[k]: int((y_true == k).sum()) for k in (0, 1)},
        "accuracy": round(float((y_true == y_pred).mean()), 4) if len(y_true) else None,
        "balanced_accuracy": round(float(np.mean(present)), 4) if present else None,
        "per_class_recall": recalls,
        "confusion_matrix": confusion(y_true, y_pred),
        "confusion_matrix_layout": "rows are true classes [0, 1], columns predicted",
        "mean_prob_target_class": round(float(probs[:, TARGET_BINARY_INDEX].mean()), 4)
        if len(probs) else None,
        "_balanced_accuracy_note": ("Mean of per-class recalls. With 19 class-6 clips in "
                                    "the whole published training folder, a validation "
                                    "split holds very few, so this number has a wide "
                                    "confidence interval that no amount of training fixes."),
    }


def evaluate(model, x, y, device, batch_size: int, frames: int, height: int,
             width: int) -> dict:
    import torch
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(device).float() / 255.0
            logits = model(C.prepare(batch, frames, height, width))
            p = torch.softmax(logits.float(), dim=-1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(dim=-1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.array([])
    prob = np.concatenate(probs) if probs else np.zeros((0, 2))
    return metrics(y.numpy(), pred, prob) | {
        "per_clip": [{"pred": int(p), "prob_closed": round(float(q[1]), 4)}
                     for p, q in zip(pred, prob)]}


def main() -> int:
    import torch
    from torch import nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="outputs/panel_cover/audit/audit.json")
    ap.add_argument("--plan", default="configs/panel_cover_plan.json")
    ap.add_argument("--out", default="outputs/panel_cover/classifier")
    ap.add_argument("--cache", default="/ocean/projects/cis260288p/puprety/scratch/pc_cache")
    ap.add_argument("--frames", type=int, default=C.DEFAULT_FRAMES)
    ap.add_argument("--height", type=int, default=C.DEFAULT_HEIGHT)
    ap.add_argument("--width", type=int, default=C.DEFAULT_WIDTH)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--val-fraction", type=float, default=0.25,
                    help="target share of each class's clips sent to validation, by group")
    ap.add_argument("--mode", choices=["probe", "train"], default="train")
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    audit_path = REPO / args.audit
    if not audit_path.exists():
        log(f"no audit at {audit_path}; run scripts/pc_audit.py first")
        return 2
    audit = json.loads(audit_path.read_text())
    label_map = save_label_map(out / "label_map.json")
    log(f"label map: {label_map['dataset_class_to_binary_index']}, "
        f"target index {TARGET_BINARY_INDEX}")

    # --- exclusions -------------------------------------------------------------
    plan_path = REPO / args.plan
    held: set[str] = set()
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        for case in plan.get("cases", []):
            held.add(case["source_clip"])
            held.add(case["donor_clip"])
    groups = audit["recording_groups"]["assignment"]
    held_groups = {groups[c] for c in held if c in groups}
    excluded = {c for c, g in groups.items() if g in held_groups} | held
    log(f"held-out evaluation clips: {sorted(held)}")
    log(f"excluded with their recording groups: {len(excluded)} clips")

    entries = [r for cid, r in sorted(audit["clips"].items()) if cid not in excluded]
    by_class = {2: [e for e in entries if e["dataset_class"] == 2],
                6: [e for e in entries if e["dataset_class"] == 6]}
    log(f"usable: class 2 {len(by_class[2])}, class 6 {len(by_class[6])}")
    if not by_class[2] or not by_class[6]:
        log("one class has no usable clips after exclusion; cannot train")
        return 2

    # --- group-separated split ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    val_ids: set[str] = set()
    split_detail = {}
    for cls, items in by_class.items():
        gs: dict[str, list[str]] = {}
        for e in items:
            gs.setdefault(groups.get(e["clip_id"], e["clip_id"]), []).append(e["clip_id"])
        order = sorted(gs)
        rng.shuffle(order)
        want = max(1, int(round(args.val_fraction * len(items))))
        taken: list[str] = []
        for g in order:
            if len(taken) >= want:
                break
            taken.extend(gs[g])
        val_ids |= set(taken)
        split_detail[str(cls)] = {"groups": len(gs), "clips": len(items),
                                  "val_clips": len(taken), "val_target": want}
        log(f"  class {cls}: {len(gs)} groups, {len(items)} clips, "
            f"{len(taken)} to validation")

    train_entries = [e for e in entries if e["clip_id"] not in val_ids]
    val_entries = [e for e in entries if e["clip_id"] in val_ids]
    tr_groups = {groups.get(e["clip_id"], e["clip_id"]) for e in train_entries}
    va_groups = {groups.get(e["clip_id"], e["clip_id"]) for e in val_entries}
    leak = tr_groups & va_groups
    if leak:
        log(f"recording groups appear in both splits: {sorted(leak)}; aborting")
        return 2
    log(f"train {len(train_entries)} clips / {len(tr_groups)} groups, "
        f"val {len(val_entries)} clips / {len(va_groups)} groups, no shared group")

    log("decoding clips to cache")
    cache = build_cache(entries, Path(args.cache), args.frames, args.height, args.width)
    xtr, ytr, tr_ids = load_split(train_entries, cache)
    xva, yva, va_ids = load_split(val_entries, cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = C.build_model(num_classes=2, pretrained=True).to(device)

    counts = np.array([(ytr.numpy() == k).sum() for k in (0, 1)], dtype=np.float64)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    log(f"train class counts {counts.tolist()}, loss weights "
        f"{[round(float(w), 4) for w in weights]}")
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.mode == "probe":
        model.train()
        mf.reset_peak_memory()
        idx = torch.arange(min(args.batch_size, len(xtr)))
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(C.prepare(xtr[idx].to(device).float() / 255.0,
                                     args.frames, args.height, args.width))
            loss = crit(logits, ytr[idx].to(device))
        scaler.scale(loss).backward()
        log(f"probe: loss {float(loss):.4f}, finite {bool(torch.isfinite(loss))}, "
            f"peak {mf.peak_memory_gb()} GiB")
        (out / "probe.json").write_text(json.dumps({
            "loss": round(float(loss), 6),
            "finite": bool(torch.isfinite(loss)),
            "peak_gpu_gib": mf.peak_memory_gb(),
            "batch": int(len(idx)),
            "input_shape": list(xtr[idx].shape),
        }, indent=1))
        return 0

    best = {"balanced_accuracy": -1.0}
    history = []
    order = np.arange(len(xtr))
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            sel = torch.from_numpy(order[i:i + args.batch_size].copy())
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(C.prepare(xtr[sel].to(device).float() / 255.0,
                                         args.frames, args.height, args.width))
                loss = crit(logits, ytr[sel].to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss) * len(sel)
        va = evaluate(model, xva, yva, device, args.batch_size,
                      args.frames, args.height, args.width)
        row = {"epoch": epoch, "train_loss": round(total / max(1, len(order)), 5),
               "val_balanced_accuracy": va["balanced_accuracy"],
               "val_accuracy": va["accuracy"]}
        history.append(row)
        log(f"  epoch {epoch:2d}: loss {row['train_loss']:.4f}, val balanced acc "
            f"{va['balanced_accuracy']}, val acc {va['accuracy']}")
        if (va["balanced_accuracy"] or 0) > best["balanced_accuracy"]:
            best = {"epoch": epoch, "balanced_accuracy": va["balanced_accuracy"] or 0.0,
                    "val": va}
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "label_map": label_map,
                        "preprocess": {"frames": args.frames, "height": args.height,
                                       "width": args.width,
                                       "function": "unsafenet_cf.classifier.prepare"},
                        "val_metrics": va},
                       out / "panel_classifier.pt")
    train_s = time.time() - t0

    tr_metrics = evaluate(model, xtr, ytr, device, args.batch_size,
                          args.frames, args.height, args.width)
    report = {
        "_what": ("Binary R3D-18 for dataset class 2 (Opened Panel Cover) against class 6 "
                  "(Closed Panel Cover). A new model: the walkway classifier is not "
                  "reused or relabelled."),
        "label_map": label_map,
        "generated": mf.utc_now(),
        "settings": {k: v for k, v in vars(args).items()},
        "preprocess": {"function": "unsafenet_cf.classifier.prepare",
                       "frames": args.frames, "height": args.height, "width": args.width,
                       "_note": "Identical in training, evaluation and guidance."},
        "exclusions": {
            "held_out_evaluation_clips": sorted(held),
            "excluded_with_recording_groups": sorted(excluded),
            "_why": ("Selected sources, their donors and every clip sharing a recording "
                     "group with them are out of training and validation alike."),
        },
        "split": {
            "train_clips": tr_ids, "val_clips": va_ids,
            "train_groups": len(tr_groups), "val_groups": len(va_groups),
            "shared_groups": sorted(leak),
            "per_class": split_detail,
            "_method": ("Split by recording group from the audit's measured frame overlap, "
                        "never by individual clip."),
        },
        "class_balance": {"train_counts": counts.tolist(),
                          "loss_weights": [round(float(w), 6) for w in weights]},
        "history": history,
        "best_epoch": best.get("epoch"),
        "validation_at_best": best.get("val"),
        "training_set_metrics_at_final_epoch": tr_metrics,
        "train_seconds": round(train_s, 1),
        "peak_gpu_gib": mf.peak_memory_gb(),
        "checkpoint": str((out / "panel_classifier.pt").relative_to(REPO)),
        "environment": mf.environment_snapshot(),
        "_limitation": ("The published dataset has 19 class-6 training clips in total. "
                        "After recording-group separation the validation split holds only "
                        "a handful of independent safe clips, so every validation number "
                        "here is an estimate from very few samples. It is reported, not "
                        "smoothed over, and it bounds what any later diagnostic can show."),
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=1))
    log(f"\nbest epoch {best.get('epoch')}, validation balanced accuracy "
        f"{best.get('balanced_accuracy')}")
    log(f"wrote {out / 'training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
