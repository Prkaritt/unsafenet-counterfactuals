# Three walkway sources, base CogVideoX-2b versus the trained LoRA adapter

**Date:** 2026-11-05. **Outcome: no source improved.** Two are *unchanged*, one is
*inconclusive*, and in every one of the six edits the marked lane itself was damaged or
relocated.

No retraining. No classifier-gradient guidance in any run: `scripts/multi_walkway_generate.py`
never imports the guidance module and never computes a classifier gradient. The adapter is
the one trained by job 50230058, sha256
`57b4e3e43523980c87dab22febba3354c2d9e008aefc9e8161c7df64ca8f53e0`, applied at scale 1.0.

Three clips cannot support a claim about generalisation, and none is made here.

---

## Sources, selected and frozen before any generation

Recorded in `generation_plan.json` and `selection_record.json`. The candidate pool was class
0, present on disk, excluding `0_tr143`, every quarantined or label-ambiguous clip, every
recording group overlapping LoRA training footage or `0_tr143`, and every clip with even an
uncertain overlap with a training clip. One clip per recording group.

| Source | Role | Median foot-to-lane, full resolution | Direction |
|---|---|---|---|
| `0_tr16` | worker near the lane | 186.8 px | right |
| `0_tr125` | worker far from the lane | 499.5 px | right |
| `0_tr48` | different walking direction | 206.1 px | left |

`0_tr16` comes from a scene cluster of 46 class-0 clips containing **zero** class-4 clips, so
no footage from that camera entered LoRA training. `0_tr125` shares a camera view with the
original source `0_tr143`.

`0_tr123` was rejected: its worker is already on the marked lane for essentially the whole
clip, median 0.4 px, despite the class-0 label. Reported, not relabelled, not used.

### Limitation: two camera views exist, not three

Uncropped like-for-like background correlation gives view A
(`0_tr1, 0_tr2, 0_tr16, 0_tr48, 0_tr61, 0_tr109, 0_tr119`, mutual r 0.74-0.99) and view B
(`0_tr124, 0_tr125, 0_tr135, 0_tr143`, mutual r 0.92-0.98), with r 0.06-0.19 between them.
`0_tr16` and `0_tr48` are therefore the **same** camera view, r 0.929, from different
recordings and different recording groups. The third source supplies a different walking
direction, not a third view. An earlier scan appeared to find a third camera family at
r~0.40; it compared uncropped candidates against a cropped reference and was wrong. Both the
wrong number and the correction are on the record.

---

## How the measurements are defined

**Worker position.** The lane is annotated once from each *preprocessed source* as the union
of its dense green pixels, and that single mask is applied unchanged to the source, the base
edit and the LoRA edit. Anchoring to the source matters: measured against its own repainted
green, the `0_tr16` base edit would have scored as a success while the worker never moved.
No fixed x threshold is used, so the measure is comparable across crops and views.

"Dense green" means `G - max(R, B) > 15` kept only where at least half of a 15x15
neighbourhood is green. A plain per-pixel test leaves speckle across the whole frame and
makes a minimum-distance measure meaningless, since one stray pixel beside a foot reports
zero.

**The worker is tracked, not re-chosen each frame.** An earlier pass took the largest
detection per frame. These clips contain distant figures as well as a foreground worker, so
the largest flipped between people and the foot path jumped across the frame. That pass is
preserved as `evaluation_untracked_50249193.json` and **its distance numbers must not be
quoted.** The current tracker seeds on the single largest detection and grows forwards and
backwards with a 120 px jump gate and a height-consistency gate. Frames with no acceptable
candidate are misses and stay empty; nothing is interpolated.

**Whole-frame PSNR and mean absolute difference** are reported but are dominated by the
static background and nearly blind to a worker occupying a small part of the frame. They are
not evidence that the worker was preserved.

---

## Results

| Source | Arm | Foot-to-lane median px | Change vs source | Frames on lane | Detector misses | Lane IoU |
|---|---|---|---|---|---|---|
| `0_tr16` | source | 262.4 | — | 0 / 49 | 0 | 1.0000 |
| | base | 247.3 | −15.1 | 0 / 49 | 0 | **0.0000** |
| | lora | 255.7 | −6.7 | 0 / 49 | 0 | 0.2443 |
| `0_tr125` | source | 394.6 | — | 0 / 49 | 1 | 1.0000 |
| | base | 393.2 | −1.4 | 0 / 49 | 1 | **0.0017** |
| | lora | 384.8 | −9.8 | 0 / 49 | 3 | 0.0477 |
| `0_tr48` | source | 107.7 | — | 0 / 49 | **24** | 1.0000 |
| | base | 144.1 | +36.4 | 0 / 49 | 11 | 0.0120 |
| | lora | 136.8 | +29.1 | 0 / 49 | 14 | 0.4207 |

Generation cost, all six: 72.1-79.2 s each, peak 27.84 GiB base and 27.95 GiB LoRA.
PSNR against source: `0_tr16` 19.660 base / 20.111 lora, `0_tr125` 20.313 / 21.367,
`0_tr48` 19.517 / 19.740.

Adapter state was verified in both directions rather than assumed: **0** LoRA parameter
tensors on the transformer during all three base runs, **240** during all three LoRA runs.
The base arm ran to completion before the adapter was ever loaded.

---

## Per-source verdict

### `0_tr16` — UNCHANGED

The worker never reaches the lane, 0 of 49 frames in both edits. Base closes 15.1 px of a
262.4 px gap, 5.8%; LoRA closes 6.7 px, 2.6%, so **the adapter moved the worker less than
base did**. Detection is clean throughout, 0 misses in all three clips, so this is the most
reliable of the three cases and it shows no improvement.

This is also the source whose camera view contributed no LoRA training footage. If the
adapter generalised to an unseen view, this is where it would show, and it does not.

### `0_tr125` — UNCHANGED

Base closes 1.4 px of 394.6, LoRA 9.8 px, 2.5%. Both are negligible against a gap of nearly
400 px, and 0 of 49 frames reach the lane. Detection is near-clean, 1 to 3 misses. This is
the source sharing a camera view with the original `0_tr143` experiment, and it reproduces
that result.

### `0_tr48` — INCONCLUSIVE

Not reported as a failure, because the measurement is not trustworthy here. The tracker loses
the worker in **24 of 49 source frames**, so the source baseline of 107.7 px is computed from
25 frames while the base figure comes from 38 and the LoRA figure from 35. Those are
different frame subsets and are not comparable; the apparent +36.4 and +29.1 px *away* from
the lane may be an artefact of which frames survived rather than movement.

The high miss rate is itself a finding worth keeping. The source is a genuinely hard tracking
case, and the edits do not fix it: 11 and 14 misses remain. What can be said is that
0 of 49 frames put the worker on the lane in any arm.

---

## The finding that holds across all six edits: the lane is destroyed or moved

Lane IoU against each source's own marking: base **0.0000**, **0.0017**, **0.0120**; LoRA
0.2443, 0.0477, 0.4207.

Base effectively erases or relocates the marking in all three sources. In the `0_tr16`
overlay the base edit has painted a broad green lane down the centre-right, where the
source's lane is a narrow strip at the far right. LoRA preserves the marking better in two of
three, and still badly.

This is precisely the outcome `configs/experiment.json` names as the one to reject: the
environment is altered instead of the behaviour. It is a stronger and more consistent effect
than anything measured about the worker, and it is why the distance measure had to be
anchored to the source's lane.

It also compounds the identity failure already recorded for `0_tr143` in
`outputs/lora/FINDINGS.md`. Taken together: at strength 0.6 this pipeline rewrites the
worker's appearance and the floor markings, while leaving the worker's path essentially where
it was.

---

## What this establishes, and what it does not

**Establishes.** Across three walkway-violation sources spanning two camera views, two
walking directions and foot-to-lane gaps from 108 to 395 px, editing at strength 0.6 with an
explicit safe-behaviour prompt and no classifier guidance does not move the worker onto the
marked lane, with or without this adapter. In no clip does any frame place the worker on the
lane. The marked lane is damaged in every edit.

**Does not establish.** That LoRA fine-tuning cannot work here. One adapter, one rank, one
step count, one strength, one adapter scale, one seed, three clips. Nor anything about
generalisation: three clips from two camera views cannot support such a claim. Nor anything
about safety. A worker moving onto painted lane is not proof of safe walking, and no
classifier prediction was used as evidence anywhere in this comparison.

All six outputs remain **candidate counterfactuals**. The human review in
`templates/REVIEW_TEMPLATE.md` is unsigned and is what decides the four criteria.
