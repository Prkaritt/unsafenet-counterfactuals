# Session log — 2026-11-04 into 2026-11-05

Written so a session with no transcript can pick up without re-deriving anything. Read
`HANDOFF.md` first for the standing state of the project; this file records what happened
in this working session, what was corrected and why, and what is still in flight.

Everything here is from `sacct`, from files on disk, or from measurements made during the
session. Where a number was wrong earlier and later corrected, both are kept.

---

## 1. What this session did

Four things, in order:

1. Verified Slurm accounting and found `docs/SU_LEDGER.md` badly stale.
2. Fine-tuned a LoRA adapter on CogVideoX-2b and ran a matched base-versus-LoRA edit of
   the original source clip `0_tr143`. Negative result.
3. Corrected that result twice after the user watched the clips. Both corrections matter
   more than the original reading.
4. Ran a multi-source comparison on three *new* walkway-violation clips with the same
   adapter. Generation finished; the final evaluation pass is still queued.

No classifier was repaired. Gate C still FAILS and was not revised. No classifier-gradient
guidance was used anywhere in this session.

---

## 2. Accounting, which was wrong on disk

`docs/SU_LEDGER.md` claimed 0.00 SU spent under the 10 SU budget. `sacct` said 1.5792 SU
had already been spent before this session. The ledger has been rewritten from `sacct` and
now lists every job including the failures, because a ledger that hides its waste cannot be
used to plan the next run.

**Current: 3.8503 SU spent of the approved 10. Remaining 6.1497.** Project total since
2026-11-01, including three pre-budget jobs, is 4.2153 SU.

`projects` lags and still printed `223 / 300 SU` all session. Treat `sacct` as authoritative
and recompute with `outputs/lora/su_actuals.json` or the snippet in `docs/SU_LEDGER.md`.

Jobs this session:

| Job | Name | Elapsed | SU | Outcome |
|---|---|---|---|---|
| 50226774 | `cflora-probe` | 00:04:10 | 0.0694 | interval verification done; trainer died, `protobuf` missing |
| 50226994 | `cflora-probe` | 00:30:04 | 0.5011 | TIMEOUT having printed nothing — see trap 1 |
| 50229034 | `cflora-probe` | 00:11:19 | 0.1886 | 4.14 s/step, 20.16 GiB peak; training worked, job exited 1 — see trap 3 |
| 50229842 | `cflora-train` | 00:02:42 | 0.0450 | died at step 0 — see trap 2 |
| 50230058 | `cflora-train` | 01:04:46 | 1.0794 | **adapter trained**, 1000 steps |
| 50232171 | `cflora-gen` | 00:08:22 | 0.1394 | base and LoRA edits of `0_tr143` |
| 50232395 | `cflora-verify` | 00:00:18 | 0.0050 | worker position measured |
| 50247622 | `cfsel` | 00:02:20 | 0.0389 | candidate characterisation, 15 clips |
| 50248886 | `cfsel3` | 00:00:24 | 0.0067 | characterisation of 6 more candidates |
| 50249193 | `cfmulti` | 00:11:51 | 0.1975 | **six edits generated** plus first evaluation |
| 50250149 | `cfeval2` | queued | — | re-evaluation with a tracked worker |

**0.5461 SU bought nothing**, lost to mistakes in job scripts (50226994 and 50229842), not
to the model or the cluster.

---

## 3. The LoRA experiment on `0_tr143`

### Setup

- Trainer: `diffusers` `examples/cogvideo/train_cogvideox_lora.py` **at tag v0.36.0**,
  vendored to `third_party/diffusers_v0.36.0/`. The `main` copy calls
  `check_min_version("0.42.0.dev0")` and refuses to run against the pinned
  `diffusers 0.36.0`.
- Added to the pinned env: `peft 0.21.2`, `decord 0.6.0`, `ftfy 6.3.1`,
  `protobuf 7.36.2`, `tensorboard 2.21.0`.
- Data: 16 class-4 *Safe Walkway* clips, `data/lora_safe_walkway/`, built by
  `scripts/build_lora_dataset.py` from the recorded audits. Four dropped by the
  contamination screen (`4_tr33`, `4_tr37` label-ambiguous; `4_tr31`, `4_tr38` sharing a
  recording with quarantined clips), two more dropped after verifying the 49-frame window
  (`4_tr16`, `4_tr36`).
- Hyperparameters: rank 64, `lora_alpha` 64, lr 1e-4, `constant_with_warmup`, 100 warmup
  steps, batch 1, fp16, gradient checkpointing, VAE slicing and tiling, seed 20261103,
  1000 steps, `--dataloader_num_workers 0`.
- Adapter: `outputs/lora/adapter/pytorch_lora_weights.safetensors`, 59,013,712 bytes,
  sha256 `57b4e3e43523980c87dab22febba3354c2d9e008aefc9e8161c7df64ca8f53e0`.
  Resume checkpoints at `checkpoint-750` and `checkpoint-1000` stay on Ocean; each holds a
  225 MB `optimizer.bin` and neither belongs in a review package.

### Result: negative

The worker did not move onto the walkway. Foreground worker foot x, median:
source 313.7, VAE reconstruction 314.1, base 311.2, **LoRA 311.3**, against a marked
walkway at x > 450, and **0 / 49** frames on the walkway in every clip. Base and LoRA agree
to 0.1 px.

### Two corrections, both after the user watched the clips

**Correction 1 — identity is not preserved, in either edit.** The source worker wears dark
navy overalls with red shoulder panels and a chest logo. Both edits remove the red panels
and alter the face; the LoRA garment reads as a grey-blue bib. `configs/experiment.json`
lists worker identity and clothing under `context_that_must_not_change`, so this fails
criterion 2, not only criterion 1. The automated check measured one scalar per frame and
could not see it. See `outputs/lora_edit/worker_crops.png`.

**Correction 2 — a claim about posture was withdrawn.** Correction 1 originally also said
the LoRA clip turned the worker's posture toward the machines. The user then saw the same
extended-arm posture in the no-adapter base clip. Measurement over frames 24-48 gives mean
worker box width/height of **0.375 source, 0.333 base, 0.354 LoRA**: the *source* silhouette
is widest, so the posture comes from the footage and is **not** an adapter effect. See
`outputs/lora_edit/pose_strip.png`.

The standing verdict is therefore: the editing pipeline, with or without the adapter, fails
to preserve the worker's identity while also failing to move him onto the walkway. Identity
loss belongs to the base strength-0.6 edit. The adapter's own measurable effects are small
and global: PSNR +0.88 dB, MAD -1.57, a marginally wider worker box.

**Rule that came out of this, and should be kept:** position, identity and pose are three
different questions needing three checks, and every claimed adapter effect must be tested
against the base arm, because anything present in all three arms belongs to the footage or
the pipeline and not to the thing under test.

Written up in `outputs/lora/FINDINGS.md`. Package at
`/ocean/projects/cis260288p/puprety/transfer/lora-counterfactual-review/` (45 files, 76 MB,
checksums verified). That directory name is misleading: no counterfactual was produced.

---

## 4. The multi-source comparison, `outputs/multi_walkway/`

Three new walkway-violation sources, each edited twice, adapter reused with no retraining
and no classifier gradients.

### Selection, frozen before generation

Recorded in `outputs/multi_walkway/generation_plan.json` and `selection_record.json`.
Candidate pool: class 0, on disk, excluding `0_tr143`, quarantined and label-ambiguous
clips, any recording group overlapping LoRA training footage or `0_tr143`, and any clip with
even an uncertain overlap with a training clip. One clip per recording group.

| Source | Role | Median foot-to-lane, full res | Direction |
|---|---|---|---|
| `0_tr16` | near the lane | 186.8 px | right |
| `0_tr125` | far from the lane | 499.5 px | right |
| `0_tr48` | different walking direction | 206.1 px | **left** |

`0_tr16` comes from a scene cluster holding 46 class-0 clips and **zero** class-4 clips, so
no footage from that camera entered LoRA training. `0_tr125` shares a camera view with
`0_tr143`.

`0_tr123` was rejected because its worker is already on the marked lane for essentially the
whole clip, median 0.4 px, despite its class-0 label. Reported, not relabelled, not used.

### Limitation: there are only two camera views, not three

Uncropped like-for-like background correlation:

- View A: `0_tr1, 0_tr2, 0_tr16, 0_tr48, 0_tr61, 0_tr109, 0_tr119`, mutual r 0.74-0.99
- View B: `0_tr124, 0_tr125, 0_tr135, 0_tr143`, mutual r 0.92-0.98
- Between views: r 0.06-0.19

`0_tr16` and `0_tr48` are the same view (r 0.929) from different recordings and different
recording groups. The third source supplies a different walking direction, which the brief
allowed as the alternative. **Do not describe this as three camera views.**

### Generated, job 50249193

Six clips in `outputs/multi_walkway/videos/`. Adapter state verified both ways: **0** LoRA
parameter tensors during all three base runs, **240** during all three LoRA runs, scale 1.0.

### Finding that does not depend on the tracker: the edits relocate the lane

Lane IoU against each source's own lane:

| Source | base | LoRA |
|---|---|---|
| `0_tr16` | 0.0000 | 0.2443 |
| `0_tr125` | 0.0017 | 0.0477 |
| `0_tr48` | 0.0120 | 0.4207 |

Base effectively destroys or moves the marking in all three. In the `0_tr16` overlay the
base edit has painted a broad green lane down the centre-right, where the source's lane is a
narrow strip at the far right. This is the failure mode `configs/experiment.json` names as
the one to reject: the environment is altered rather than the behaviour.

It is also why the worker-distance measure is anchored to the **source's** lane mask, applied
unchanged to all three arms. Measured against its own repainted green, `0_tr16` base would
have scored as a success while the worker never moved.

`0_tr48` degrades badly: the detector loses the worker in 11/49 base and 14/49 LoRA frames,
against 0/49 in the source.

### Why a re-evaluation is queued

The first evaluation took the largest detection in each frame as "the worker". These clips
contain distant figures as well as the foreground worker, so the largest flipped between
people and the foot path jumped across the frame, visible as long straight lines in the
first overlays. Any distance statistic from that pass is unusable and is preserved as
`outputs/multi_walkway/evaluation_untracked_50249193.json`.

`scripts/evaluate_multi_walkway.py` now has `track_worker`, which seeds on the single
largest detection and grows the track forwards and backwards with a 120 px jump gate and a
height-consistency gate. Misses stay empty; nothing is interpolated.

**Do not quote worker-distance numbers from the untracked pass.** The lane IoU figures above
are unaffected, since they never used the detector.

---

## 5. The multi-source run completed

Job **50250149** `cfeval2` finished at 02:54:49 on 2026-11-05, 00:02:17, 0.0381 SU. The
tracked evaluation is in `outputs/multi_walkway/evaluation.json`; the untracked pass is kept
as `evaluation_untracked_50249193.json` and its distance numbers must not be quoted.

**No source improved.** `0_tr16` unchanged, `0_tr125` unchanged, `0_tr48` inconclusive. The
worker reaches the marked lane in 0 of 49 frames in all six edits. Full table and per-source
reasoning in `outputs/multi_walkway/FINDINGS.md`.

The tracker changed one case materially. With the largest-detection rule `0_tr48` showed 0
source misses; with continuity gates it shows **24 of 49**, because the gates reject the
jumps between different people that the old rule silently accepted. That is why `0_tr48` is
inconclusive rather than a reported failure: its source baseline comes from 25 frames while
its base and LoRA figures come from 38 and 35, which are not comparable subsets.

The strongest result is about the environment, not the worker. Lane IoU against each
source's own marking: base 0.0000, 0.0017, 0.0120; LoRA 0.2443, 0.0477, 0.4207. The base
edit erases or relocates the marked lane in every source.

Delivered:

- `outputs/multi_walkway/FINDINGS.md`
- `outputs/multi_walkway/review/index.html`
- package `/ocean/projects/cis260288p/puprety/transfer/multi-walkway-review/`, 54 files,
  99 MB, all 54 checksums verified
- `STATUS.md` section 6 and `docs/SU_LEDGER.md` updated with actual charges

**Final accounting: 3.8883 SU spent of 10, 6.1117 remaining.** Project total since
2026-11-01 is 4.2533 SU. The multi-walkway stage cost 0.2811 SU against the 1.00 SU reported
as its maximum before submission.

Mac rsync:

```
mkdir -p ~/workspace/unsafenet-counterfactuals/unsafenet_cf_outputs/multi-walkway-review
rsync -avP --partial <user>@data.bridges2.psc.edu:/ocean/projects/cis260288p/puprety/transfer/multi-walkway-review/ ~/workspace/unsafenet-counterfactuals/unsafenet_cf_outputs/multi-walkway-review/
```

Nothing is queued or running. The human review in `templates/REVIEW_TEMPLATE.md` remains
unsigned, and every generated clip remains a candidate counterfactual.

## 6. Traps, measured this session

Added to the list already in `HANDOFF.md`.

1. **Never pipe a long-running trainer into `tail`.** Job 50226994 hit its wall clock having
   printed nothing, because everything it wrote was still in the pipe when Slurm killed it.
   0.5011 SU for no information. Let Slurm capture stdout and stderr directly and set
   `PYTHONUNBUFFERED=1`.
2. **`--dataloader_num_workers` must be 0** for the CogVideoX LoRA trainer. It pre-encodes
   the clips to CUDA latents before the loop, and forked workers cannot re-initialise CUDA:
   `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
3. **That trainer reloads the whole 13.8 GB pipeline after saving**, then calls
   `pipe.load_lora_weights(args.output_dir)` with no `weight_name`, which raises
   `ValueError: When using the offline mode, you must specify a weight_name.` under
   `HF_HUB_OFFLINE=1`. It killed job 50229034 *after* a successful save. Patched locally;
   look for `LOCAL PATCH` in the vendored script, original kept as
   `train_cogvideox_lora.py.orig`.
4. **A plain per-pixel green test is not a lane.** At 1920x1080 it leaves speckle across the
   whole frame — the densest 200 columns held only 70-87% of it — so a minimum-distance
   measure returns ~0 from one stray pixel beside a foot. Use
   `verify_lora_intervals.dense_green_mask`, which keeps a pixel only where at least half of
   a 15x15 neighbourhood is green. The first selection job reported `lane_x 0-1919` and
   `dist_min` near 0 before this was fixed.
5. **Compare like with like when correlating camera views.** An early scan appeared to find a
   third camera family at r~0.40; it had compared uncropped candidates against a cropped
   reference. Repeated correctly, there are two views.
6. **The largest detection per frame is not a worker track** when distant figures share the
   frame. Track with continuity gates instead.
7. **Whole-frame PSNR and MAD are nearly blind to the worker.** They are dominated by the
   static background. The LoRA clip scored better on both while the worker's identity was
   degraded. Never read them as evidence the subject was preserved.
8. **The checkpoint takes several minutes to load from Ocean.** 614 s from job start to the
   first optimiser step with 2 clips, dominated by the 13.8 GB read, not by the VAE
   pre-encode. Budget it per job.
9. Pipeline load for inference measured 161.9-208.2 s; one 49-frame edit at strength 0.6 and
   50 requested steps runs 30 actual steps in 72-82 s at about 1.7-1.8 s/step, peak 27.8-28.0
   GiB.

---

## 7. Files this session added

Scripts, all in `scripts/`:

- `build_lora_dataset.py` — LoRA training set from the recorded audits
- `write_lora_prompts.py` — rewrite `prompts.txt` without re-encoding video
- `lora_contact_sheets.py` — contact sheets of training clips
- `verify_lora_intervals.py` — person plus walkway check, and `dense_green_mask`
- `summarise_lora_probe.py` — turn a probe log into a recorded measurement
- `lora_generate.py` — matched base-versus-LoRA edit of `0_tr143`
- `verify_edit_outputs.py` — worker position in source, reconstruction, base, LoRA
- `build_lora_package.py` — stage the LoRA review package
- `select_walkway_sources.py` — characterise candidate sources
- `multi_walkway_generate.py` — six matched edits in one job
- `evaluate_multi_walkway.py` — per-source lane annotation, tracked worker
- `make_footpath_overlays.py` — foot-path overlays
- `build_multi_walkway_page.py` — HTML review page

Slurm: `slurm/lora_probe.sbatch`, `lora_train.sbatch`, `lora_generate.sbatch`,
`verify_edits.sbatch`, `select_sources.sbatch`, `select_sources_viewC.sbatch`,
`multi_walkway.sbatch`, `reeval_multi.sbatch`.

Config and data: `configs/lora_captions.json`, `data/lora_safe_walkway/`,
`data/lora_probe_subset/`, `third_party/diffusers_v0.36.0/`.

Outputs: `outputs/lora/`, `outputs/lora_edit/`, `outputs/multi_walkway/`.

---

## 8. Standing constraints, unchanged

- GPU work runs under Slurm only. Login nodes are never used for model workloads.
- Gate C FAILS and is not revised. `docs/ACCEPTANCE_CRITERIA.md` is preregistered and
  unchanged; post-hoc changes live in `docs/DIAGNOSTIC_AMENDMENTS.md` with what they cannot
  claim. Keep that separation.
- The official test split has never been downloaded and stays untouched.
- No LD-ViCE reproduction is claimed. LD-ViCE fine-tunes CogVideoX per dataset **and**
  denoises with target-model gradients; this project has now done each separately and
  neither worked alone.
- Every generated clip is a **candidate counterfactual** until a human completes
  `templates/REVIEW_TEMPLATE.md`. That review is still unsigned.
- The clips show identifiable workers who consented to the original study, not to
  redistribution. `outputs/` is gitignored. Keep the media local.
- `git init` has never been run.
