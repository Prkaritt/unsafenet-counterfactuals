# Where this project stands — 2026-11-04 (updated after the LoRA run)

Read this first, then `SESSION_LOG_2026-11-04.md` for what the 2026-11-04 working session
did and corrected, then `STATUS.md`. All three are written so a session with no transcript
can pick up without re-deriving anything.

## The one-sentence state

Three experiments have now run on Bridges-2 — a prompt-guided baseline, a classifier-guided
run, and a LoRA fine-tune of CogVideoX-2b — and all three failed to produce a valid
counterfactual, each for a documented reason.

## Three experiments, all complete, all negative

**1. Prompt-guided editing baseline.** Five clips across strength 0.5 to 0.8, seed 20261103.
The trade-off is monotonic: strength low enough to preserve the scene changes nothing,
strength high enough to change things redraws the room. No tested point does both. Guidance
scale is not the lever; strength is. Outputs in `outputs/stage4_edit/`.

**2. Classifier-guided run.** Slurm job 50225049. The guidance machinery works — gradients
finite, correctly directed, memory fits. The result is still a negative, for two reasons
that must be stated together:

- A label flip cannot be claimed. The source is at p(safe) 0.4852, and passing it through
  the VAE with **no editing** already moves it to 0.5315, across the boundary. The unguided
  control is at 0.5037. The guided clip reaches 1.0000, but the baseline had already
  flipped.
- The worker never moved. Guidance painted person-shaped pink blobs into the machine bank
  and near the walkway instead. That is an adversarial perturbation, not a counterfactual.

Outputs in `outputs/guided/`, written up in
`/ocean/projects/cis260288p/puprety/transfer/guided-counterfactual-review/FINDINGS.md`.

## Gate status, which must not be quietly revised

| Gate | State |
|---|---|
| A data integrity, positive control | PASS |
| B0 memory probe, BF16 finite check | PASS |
| B1 balanced accuracy 0.8874 | PASS |
| B2 recalls 0.8148 / 0.96 | PASS |
| B3 VAE consistency, 79 clips, 0.9620 | PASS |
| **C shortcut diagnostics** | **FAIL** |
| D gradient probe | PASS |
| E generation | ran under explicit user authorisation **despite C failing** |

Gate C failing is the explanation for the guided run's outcome, not an inconvenience. With
the worker located by a COCO person detector at a 0.0150 miss rate, masking the worker moved
balanced accuracy by -0.0107 while masking a label-irrelevant control region moved it by
+0.0278. The classifier shows no measurable dependence on the worker, so it cannot steer the
worker.

`docs/ACCEPTANCE_CRITERIA.md` is preregistered and unchanged since before training. Every
post-hoc change lives in `docs/DIAGNOSTIC_AMENDMENTS.md` with what it cannot claim. Keep
that separation.

**3. LoRA fine-tune of CogVideoX-2b.** Authorised by the user on 2026-11-04, overriding the
earlier "no generator training" constraint. 16 verified class-4 safe-walkway intervals, rank
64, 1000 steps, lr 1e-4. Classifier guidance disabled in both arms so the comparison isolates
the adapter. Training completed and the adapter loads and is active. **The worker did not
move.** Measured with the same detector used on the training data: foreground worker foot x
median 313.7 in the source, 311.2 base, 311.3 LoRA, against a marked walkway at x>450, and
0/49 frames on the walkway in every clip. The base and LoRA edits agree on worker position to
0.1 px.

Human review then found two failures the automated measure had missed, and they matter more
than the position number. **Worker identity is not preserved in either edit**: the source's
dark navy overalls with red shoulder panels and chest logo become a grey top in base and a
grey-blue bib in LoRA, with the face smeared and altered in both. `configs/experiment.json`
lists worker identity and clothing under `context_that_must_not_change`, so this fails
criterion 2, not only criterion 1. A second claim, that the LoRA clip turned the
worker's posture toward the machines, was made and then **withdrawn**: the user saw the same
extended-arm posture in the no-adapter base clip, and measurement over frames 24-48 gives
mean box width/height of 0.375 source, 0.333 base, 0.354 LoRA. The source silhouette is the
widest of the three, so the posture comes from the footage, not the adapter.

So the verdict is: the pipeline fails to preserve the worker's identity, with or without the
adapter, while also failing to move him onto the walkway. Identity loss belongs to the base
strength-0.6 edit. The spurious outcome of experiment 2 did
**not** recur — the walkway is not repainted or extended — but the environment staying
honest while the worker is altered is a different unsafe failure, not a safe one.

PSNR 20.967 against base 20.088 dB favours LoRA, and should be discounted: whole-frame pixel
statistics are dominated by the static background and are blind to a subject occupying a
small part of the frame. Written up in `outputs/lora/FINDINGS.md`; package staged at
`/ocean/projects/cis260288p/puprety/transfer/lora-counterfactual-review/`.

## What has never been tried

**Fine-tuning and guidance have never been combined.** LD-ViCE does both; this project has
now done each separately and neither worked alone. That combination is the remaining
untried direction, and it inherits Gate C's failure: with a classifier that shows no
measurable dependence on the worker, guidance has nothing valid to steer toward, so
combining them would most likely produce better-looking spurious counterfactuals rather
than valid ones. Repairing the classifier is the prerequisite, not an optional extra.

**Only one LoRA configuration was tested.** One rank, one learning rate, one step count, one
strength, one adapter scale, one seed, one clip. The run does not establish that LoRA cannot
work here, only that this configuration did not. The most direct evidence points at the
prompt rather than the adapter: base and LoRA put the worker in the same place to within
0.1 px, so whatever the adapter learned, it did not touch the quantity the prompt was
failing to move.

## Budget

**3.6072 SU spent of 10.000 approved, 6.3928 remaining.** Project total since 2026-11-01 is
3.9722 SU. Per-job table in `docs/SU_LEDGER.md`, machine-readable in
`outputs/lora/su_actuals.json`. Both are written from `sacct`, not from requested wall time;
the old ledger claimed 0.00 spent and was wrong.

The LoRA stage cost 2.0279 SU in total, against the 2 to 4 SU it had been estimated at.
Of that, 0.5461 SU bought nothing and was lost to two mistakes in the job scripts.

## Traps that cost time, so they are not repeated

- A 64-bit average hash cannot tell one recording from another on fixed-camera factory
  footage; motion residuals cannot either. Duplicate detection uses minimum mean absolute
  pixel difference on densely sampled frames, with a positive control that must pass.
  See `docs/GROUPING_CALIBRATION.md`.
- Motion does not isolate a person here: presses and arms move. Use the person detector.
- `pipe.text_encoder = None` does not free GPU memory. Move the module to CPU first.
- Never pipe a long-running trainer into `tail`. Job 50226994 hit its wall clock having
  printed nothing, because everything it wrote was still in the pipe when Slurm killed it.
  0.5011 SU for no information. Let Slurm capture stdout and stderr directly.
- `--dataloader_num_workers` must be 0 for the CogVideoX LoRA trainer. It pre-encodes the
  clips to CUDA latents before the loop, and forked workers cannot re-initialise CUDA.
- The diffusers LoRA trainer on `main` requires diffusers 0.42; use the script from the tag
  matching the installed version. The v0.36.0 copy is vendored in `third_party/`.
- That trainer reloads the whole 13.8 GB pipeline after saving, then calls
  `load_lora_weights(output_dir)` with no `weight_name`, which raises under
  `HF_HUB_OFFLINE=1`. Patched locally; the original is kept beside it.
- Shrinking VAE tiles changes the encoder's output geometry and breaks `prepare_latents`.
- The gradient is float32; cast the update back to the latents' dtype or the transformer
  gets float32 and dies.
- Set both `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF`.
- Login nodes are CPU-throttled and cannot reach `download.pytorch.org`; fetch model weights
  inside a job. GPU-shared bills only the GPU, so its 24 cores are the cheapest CPU here.

## Not done, by user decision

- Official test split: not downloaded, final test evaluation **pending**. Untouched by the
  LoRA work, as instructed.
- Human playback review: **unsigned**, `templates/REVIEW_TEMPLATE.md`. Now covers the base
  and LoRA clips as well.
- `git init`: never run.
- Classifier repair: not attempted. Gate C still FAILS and is not revised.

## Lesson from the LoRA run, worth keeping

The automated check measured one scalar per frame, the worker's foot x. It answered that
question correctly and missed identity entirely, and this write-up initially concluded
"changed appearance, not behaviour" on its strength. The correction then over-reached in the
other direction, attributing an arm posture to the adapter without checking whether the
no-adapter base clip showed the same thing. It did, and so did the source.

Two rules out of that. Position, identity and pose are three different questions and need
three checks. And every claimed adapter effect must be tested against the base arm, because
anything present in all three arms belongs to the footage or the pipeline, not to the thing
under test.

---

## Update 2026-11-05: explicit relocation works where generation did not

Branch `experiment/manual-path-refinement`, **uncommitted**, nothing pushed or tagged.

A fifth experiment ran: segment the worker with SAM 2.1, recover the background, composite
him along a path I specified, then let a frozen CogVideoX-2b refine the result at strength
0.2. Three sources, `0_tr143`, `0_tr16` and `0_tr125`.

**The worker reaches the marked lane for the first time in this project.** Foot-to-lane
median falls from 291.0, 263.4 and 387.1 px to 123.3, 91.0 and 214.4 px, and frames on the
lane go from 0/49 in every earlier attempt to 11, 11 and 9. **Identity is preserved exactly**,
because the pixels are the worker's own, which is the sharpest contrast with the LoRA run.

**The refinement neither undid the relocation nor improved it.** Placement is unchanged to
within 1 px on two clips and improves by 52 px on the third. The pipeline enforces no
trajectory constraint, so that it held was an open question and is now a measurement.

**Two corrections worth keeping.**

1. `diffusers.utils.export_to_video` rescales a list of numpy frames by 255. For uint8 input
   that wraps modulo 256 and writes a colour-inverted video. Pass PIL images. Cost 0.0611 SU
   before it was caught by reading a written clip back.
2. A refined clip's lane IoU of 0.25 is **not** evidence that the marking was repainted. Best
   uniform scale aligning refined to composite is exactly 1.000, and mean RGB on the lane
   moves 4 to 6 units of 255. The project's paint test is a channel-difference test sitting
   1.6 to 3.3 units above its threshold, so a mild desaturation collapses it. This also puts
   an unconfirmed caveat on the multi-source LoRA run's reading of lane IoU 0.0000-0.0120 as
   erasure.

**`sbatch --export=ALL,CLIPS=a,b` splits on the comma** and exports `CLIPS=a` plus a variable
named `b`. One clip was silently skipped. Pass one value, or quote the comma.

**What this does not change.** Gate C still FAILS and is not revised. The official test split
is still untouched. The human review form is still unsigned, so every clip remains a
candidate counterfactual. No safety claim is made anywhere.

Budget: 4.1380 of 10 SU spent, 5.8620 remaining.
