# CogVideoX-2B LoRA versus base: what was done and what it shows

**Date:** 2026-11-04. **Outcome: negative.** The adapter trained and loads, both edits
generated, and the worker did not move onto the walkway in either of them. Human review
then found a failure the automated measurement had missed: the worker's identity is not
preserved in either edit. A second claim made in that correction, that the LoRA clip
turned the worker's posture toward the machines, was itself withdrawn after measurement
showed the posture is present in the base clip and strongest in the source. See
"Correction after human review" below, which supersedes the first reading of the Results
section.

## The question

Every earlier attempt in this project left CogVideoX-2b frozen. Both failed:

* the prompt-guided editing baseline could not both preserve the scene and change
  the behaviour at any tested strength, and
* the classifier-guided run painted person-shaped blobs instead of moving the worker,
  which is what a classifier that fails Gate C can be expected to steer toward.

LD-ViCE does two things this project had not: it fine-tunes CogVideoX per dataset
**and** it denoises with target-model gradients. This run does the first half only,
deliberately. **Classifier guidance is disabled in both arms of the comparison**, so
that whatever differs between them is attributable to the adapter and not to a
gradient from a classifier that is known not to depend on the worker.

This is therefore still not LD-ViCE, and it is not a safety claim.

## What Gate C's failure does and does not block

Gate C failed: masking the worker moved validation balanced accuracy by -0.0107 while
masking a label-irrelevant control region moved it by +0.0278, with the worker located
by a COCO person detector at a 0.0150 miss rate. The classifier shows no measurable
dependence on the worker.

That failure is **not revised** here and is not treated as passed. It blocks any claim
that a classifier probability says something about the worker. It does not block this
comparison, because this comparison does not use the classifier to steer anything. The
classifier probabilities reported in the generation manifest are informational only,
and they are additionally undercut by a fact already on the record: passing the source
clip through the VAE with no editing at all already moves p(safe) from 0.4852 to
0.5315, across the decision boundary.

## Training data

Built by `scripts/build_lora_dataset.py` from the recorded split and group audits, not
by hand. Starting set: the 22 class-4 *Safe Walkway* clips in the `train` split.

**Excluded by the audits (4 clips).** The selection reads
`outputs/classifier/groups.json` and drops any clip that is itself label-ambiguous or
that shares a confirmed recording with a quarantined clip:

| Clip | Why |
|---|---|
| `4_tr33` | label-ambiguous: 36.7% of sampled frames also appear in `0_tr30`, published as class 0 |
| `4_tr37` | label-ambiguous: 40.8% of sampled frames also appear in `0_tr98`, published as class 0 |
| `4_tr31` | shares a recording (min MAD 0.8286) with `4_tr32`, quarantined for label ambiguity |
| `4_tr38` | shares a recording (min MAD 0.7778) with `4_tr39`, quarantined for label ambiguity |

The counterfactual source clip `0_tr143` and the held-out validation groups are
excluded by the split itself; the official test split has never been downloaded and is
untouched. No clip in the dataset shares a recording with `0_tr143` — there are zero
certain or probable recording-overlap pairs involving it — so the clips that share its
camera position are legitimately retained, and they are the ones that carry the
scene's visual vocabulary.

**Excluded after verifying the interval (2 clips).** A dataset label applies to the
published clip, not to the 49-frame window this project cut out of it.
`scripts/verify_lora_intervals.py` measures each window directly: the COCO person
detector finds every person above score 0.5, and a person counts as on the walkway when
green paint (`G - max(R, B) > 15`) appears in a 9x9 window under their feet.

| Clip | Why |
|---|---|
| `4_tr36` | no person on the walkway in any of 49 frames; people detected in only 40 of them, median height 33 px |
| `4_tr16` | walkway contact in 30/49 frames, but every contact has a foot position near x=32, on green-painted machine base at the extreme left, not on the marked lane at x=515-640 where every other same-scene clip's contacts fall |

`4_tr16` is the reason this check was worth running. Read from contact sheets alone the
clip looked empty; the detector said 30/49; the overlay in
`contact_sheets/foot_check.png` showed both readings were picking up different things
and that the marked walkway is in fact unused. Neither the eye nor the metric settled
it alone.

**Honest caveat about this step.** The rule "keep a window only if someone actually
uses the marked walkway in it" was applied *after* seeing the verification numbers.
A rule fixed after seeing data is weaker evidence than one fixed before, which is why
both dropped clips, their measurements and this sentence are on the record rather than
folded silently into the dataset. The walkway test is also approximate: it keys on green
paint, and green paint appears on machine bases as well as on the lane.

**Result: 16 clips.** Each is cut with the source clip's exact crop
(x=300, 1620x1080) and target geometry (720x480, 49 frames, 8 fps, CRF 8), through the
same `unsafenet_cf.preprocess` path that produced the source, so the training
distribution matches what the video-to-video edit is given.

## Captions

Written per clip from the verification output and from the contact sheets, in
`configs/lora_captions.json`. Each names the scene, roughly how many people are
present, and where the person using the walkway is, with distance wording following the
measured median person height in pixels. Two scenes appear: the press line that
`0_tr143` was shot in, and a wider aisle with presses on both sides. No caption asserts
behaviour that was not measured or seen.

## Trainer provenance

`diffusers` `examples/cogvideo/train_cogvideox_lora.py` at **tag v0.36.0**, vendored
into `third_party/diffusers_v0.36.0/`. The `main`-branch copy of the same script calls
`check_min_version("0.42.0.dev0")` and refuses to run against the pinned
`diffusers 0.36.0` in this environment; the v0.36.0 copy calls
`check_min_version("0.36.0")` and matches. Dependencies added to the pinned
environment: `peft 0.21.2`, `decord 0.6.0`, `ftfy 6.3.1`, `protobuf 7.36.2`,
`tensorboard 2.21.0`.

Two properties of the script were read rather than assumed:

* It pre-encodes every clip through the VAE once before the first optimiser step, so
  training steps are transformer-only and the per-clip VAE cost is paid once per job.
* After saving the adapter it reloads the whole 13.8 GB pipeline unconditionally, even
  with no validation prompt set. That is dead time at the end of every run, and it
  happens after the weights are on disk.

## Service units

Read from `sacct`, not from requested wall time. Full table in `SU_LEDGER.md`.

| Job | Elapsed | SU | Outcome |
|---|---|---|---|
| 50226774 | 00:04:10 | 0.0694 | interval verification completed; trainer died, `protobuf` missing |
| 50226994 | 00:30:04 | 0.5011 | TIMEOUT having printed nothing: the trainer was piped into `tail`, so its output was still in the pipe when Slurm killed it |
| 50229034 | 00:11:19 | 0.1886 | timing probe, no pipe, two-clip subset |

The 0.5011 SU of job 50226994 bought nothing and was wasted by a mistake in the job
script, not by the model. It is recorded rather than omitted. The full stage table is in
the Results section below.

---

## Results

**The adapter did not produce the intended behaviour change. This is a negative result.**

### Did the worker move onto the walkway? No.

Measured with the same detector and the same green-paint foot test used to verify the
training intervals, so the claim rests on one instrument rather than on an impression
(`outputs/lora_edit/worker_position.json`, job 50232395). In this camera the marked
walkway occupies roughly x > 450 of a 720-pixel-wide frame.

| Clip | Foreground worker foot x, median | Frames with the worker on the walkway |
|---|---|---|
| source | 313.7 | 0 / 49 |
| VAE reconstruction | 314.1 | 0 / 49 |
| base, no adapter | 311.2 | 0 / 49 |
| **LoRA** | **311.3** | **0 / 49** |

The worker is detected in all 49 frames of every clip. Between the source and the LoRA
edit his feet move 2.4 pixels, away from the walkway rather than toward it. The intended
change is a lateral relocation of several hundred source pixels. Nothing of the kind
happened, and the base and LoRA edits are indistinguishable on this measure: 311.2
against 311.3.

### What did change

| | base | LoRA |
|---|---|---|
| Generation time | 82.2 s | 75.1 s |
| Peak GPU | 27.840 GiB | 27.953 GiB |
| PSNR vs source | 20.088 dB | **20.967 dB** |
| Mean absolute difference vs source | 16.7129 | **15.1458** |

The adapter scores slightly better on both: +0.88 dB PSNR, -1.57 MAD. Read this with
care. PSNR and mean absolute difference are whole-frame pixel statistics, dominated by
the large static background. They say the LoRA clip is globally closer to the source.
They say nothing about the worker, who occupies a small fraction of the frame, and the
correction below shows his identity is *not* preserved in either clip. A frame-level
preservation score improving while the subject is degraded is exactly the kind of
mismatch these aggregate metrics hide.

Both clips restyle the machine bank and wash out the floor, the same strength-0.6 trade
the earlier strength sweep documented.

### Correction after human review, same day

**Added after the user watched the clips. It corrects this document's own earlier
conclusion and is the more important finding.**

The measurement in `worker_position.json` records one quantity per frame: the horizontal
foot position of the largest detected person. On the strength of it this document first
concluded that the adapter "changed appearance, not behaviour" and that the failure was
that nothing moved. Both statements were too generous, because foot position cannot see a
face, a garment, an arm or a body orientation, and those are where the real changes are.

Visible in `outputs/lora_edit/worker_crops.png` (frames 6, 18, 30, 42, worker region
enlarged):

**1. Worker identity is not preserved, in either edit.** In the source the worker wears
dark navy overalls with distinct **red shoulder panels** and a chest logo, and is bearded
with dark hair. In the base edit the red panels are gone, the top is grey, the face is
smeared featureless and the hair is lighter. In the LoRA edit the red panels are likewise
gone and the garment reads as a grey-blue bib or apron rather than overalls, with a
different face again.

`configs/experiment.json` lists "worker identity, clothing type and build" under
`context_that_must_not_change`. This is a **failure of that constraint**, and it is a
failure of criterion 2 of the review template, not only of criterion 1.

**2. Arm position: an over-attribution, withdrawn.** This document first recorded, from
four sampled frames, that "in the LoRA clip the worker's posture turns toward the
machines", implying the adapter caused it. **That claim is withdrawn.** The user then
observed the same extended-arm posture in the base clip, which has no adapter, and
measurement over frames 24-48 agrees:

| Clip | Mean worker bounding-box width / height |
|---|---|
| source | **0.375** |
| base, no adapter | 0.333 |
| LoRA | 0.354 |

A wider box means arms further from the body. The **source** worker has the widest
silhouette of the three: he already walks with his hands raised in front of him, looking
down at them. Both edits narrow it, the LoRA clip slightly less than base. The posture
therefore originates in the source footage, survives into both edits, and **is not an
effect of the adapter**. Strip in `outputs/lora_edit/pose_strip.png`.

The 2.4 px leftward shift in median foot position cannot carry the claim on its own and
is not evidence of a turn toward the machines. It is within the noise of a detector box
on a restyled clip.

Whether the worker appears to reach toward or contact a machine in any of these clips is
a judgement for criterion 3 of the review form, against all 49 frames in motion. It is
not established here, and it is not attributable to the LoRA.

**What this changes.** The verdict is not "the adapter did nothing" and not "the adapter
moved the worker toward the hazard" either. It is that **the editing pipeline, with or
without the adapter, fails to preserve the worker's identity while also failing to move
him onto the walkway.** Identity loss is a criterion 2 failure and it belongs to the
base strength-0.6 edit, not to the LoRA. The adapter's own measurable effects are
confined to small global pixel statistics.

**What this says about the measurement, and about this document.** Two errors happened
in sequence and both are worth keeping on the record.

First, a single scalar per frame was too narrow an instrument for a criterion about a
person's appearance and conduct. Foot position answered its own question correctly and
that question was not the whole question, so "changed appearance, not behaviour" was
asserted on evidence that could not support it.

Second, the correction itself over-reached: a pose seen in four sampled frames of the
LoRA clip was attributed to the adapter without checking the base clip for the same
pose. The base clip had it, and so does the source. The fix for an under-measured claim
is measurement, not a differently-shaped guess.

Any future run needs identity, pose and position checked separately, each against the
base arm as well as the source, since an effect present in all three arms is a property
of the footage or the pipeline and not of the thing under test. The human review stayed
the authority throughout and caught both errors.

### The spurious failure mode did not occur

`configs/experiment.json` names the outcome to reject: a walkway repainted, widened or
extended underneath a worker who has not moved. That did not happen. In
`outputs/lora_edit/walkway_overlay.png`, which overlays detected green paint in cyan,
the walkway keeps its position and shape in both edits and stays unoccupied.

This matters because it is a *different* negative from the classifier-guided run. That
run flipped a prediction by painting person-shaped blobs into the machine bank; this one
left the environment honest.

That is the limit of the comparison, and the correction above narrows it further. The
environment was not faked, but the worker was: his identity is lost in both edits. An
edit that leaves the scene intact while changing who the person is is not a safe failure
either. It is a different one.

### Classifier probabilities, informational only

| Clip | p(class 4) |
|---|---|
| source | 0.4852 |
| VAE reconstruction | 0.5315 |
| base | 0.5025 |
| LoRA | 0.4748 |

These carry no evidential weight and no claim is made from them. Gate C failed, so the
classifier has no measurable dependence on the worker; and the VAE round trip alone
already moves the source across the decision boundary, from 0.4852 to 0.5315, with no
editing at all. The LoRA clip scores lowest of the four, which if anything points the
wrong way, and that too is not evidence of anything.

### Training

1000 steps in 1:01:55 at 3.72 s/step, 16 clips, rank 64, lora_alpha 64 (scale 1.0),
lr 1e-4 with 100 warmup steps, batch 1, fp16, gradient checkpointing. Adapter 59,013,712
bytes. Loss by 100-step bucket is in `train_loss_summary.json`: mean 0.2506 over steps
1-100 against 0.2249 over steps 901-1000.

That near-flat curve should not be read as either success or failure. The diffusion MSE
is taken at a randomly sampled timestep each step, so it is dominated by timestep
variance and would look much like this whether or not the adapter learned anything
useful. It is recorded because it was measured, not because it supports a conclusion.

### Why this probably failed, stated as hypotheses rather than findings

None of these is tested by this run.

1. **The prompt cannot express the edit.** The earlier baseline reached the same wall:
   strength low enough to preserve the scene changes nothing, strength high enough to
   change things redraws the room. Fine-tuning supplies visual vocabulary; it supplies no
   mechanism for *where to put a person*. At strength 0.6 the latents still carry the
   worker's original position strongly, and nothing in this setup pushes against that.
2. **16 clips is thin, and most show the behaviour at a distance.** Median walkway-person
   height is 80-150 px in the source camera's scene and 40-55 px in the other, against a
   source clip whose worker is large and close. The adapter may have learned "this
   factory" far better than "a worker walking on the lane, near the camera".
3. **62 epochs over 16 clips risks memorising rather than generalising.** Preservation
   improving while behaviour did not move is weakly consistent with an adapter that
   learned the scene's appearance and nothing about trajectories.
4. **Nothing in this run steers.** This was the deliberate design: classifier guidance
   was disabled in both arms so the comparison isolates the adapter. The cost of that
   choice is that the only force acting on the worker's position is the text prompt,
   which the baseline had already shown to be insufficient.

Hypothesis 1 is the one the evidence most directly supports, because the base and LoRA
edits agree to within 0.1 px on worker position: whatever the adapter changed, it did
not touch the thing the prompt was failing to move.

### Service units

| Job | Elapsed | SU | Outcome |
|---|---|---|---|
| 50226774 | 00:04:10 | 0.0694 | interval verification completed; trainer died, `protobuf` missing |
| 50226994 | 00:30:04 | 0.5011 | TIMEOUT with no output: trainer piped into `tail`, output lost when killed |
| 50229034 | 00:11:19 | 0.1886 | timing probe; training succeeded, job exited 1 in upstream's post-save block |
| 50229842 | 00:02:42 | 0.0450 | died at step 0: `--dataloader_num_workers 2` forks workers around CUDA latents |
| 50230058 | 01:04:46 | 1.0794 | **training completed**, 1000 steps |
| 50232171 | 00:08:22 | 0.1394 | **base and LoRA edits generated** |
| 50232395 | 00:00:18 | 0.0050 | worker position measured |

**LoRA stage total 2.0279 SU. Budget spent 3.6072 of 10. Remaining 6.3928.**

Of the stage total, **0.5461 SU bought nothing** and was lost to two mistakes in my own
job scripts, not to the model or the cluster: the lost-output pipe and the dataloader
workers. Both are recorded here rather than absorbed into the "training" line.

## What this run establishes and what it does not

**Establishes.** Fine-tuning CogVideoX-2b on 16 verified safe-walkway intervals from this
dataset, at rank 64 for 1000 steps, and then editing the source clip at strength 0.6 with
an explicit safe-behaviour prompt and no classifier guidance, does not move the worker
onto the marked walkway. The pipeline also fails to preserve his identity, with and
without the adapter. The adapter loads and is active; the run is reproducible from the
recorded seed, settings and manifests.

**Does not establish.** That LoRA fine-tuning cannot work here. One configuration was
tested: one rank, one learning rate, one step count, one strength, one adapter scale,
one seed, one clip. Nor does it establish anything about LD-ViCE, which this is still not:
LD-ViCE fine-tunes CogVideoX per dataset **and** denoises with target-model gradients,
and only the first half was done here, deliberately.

Both clips remain **candidate counterfactuals**, and in fact neither is a counterfactual
at all on the evidence above, since the behaviour they were meant to change did not
change. The human review in `REVIEW_TEMPLATE.md` is still unsigned and is what decides
the four criteria.

## What this will not establish

A visible improvement in the worker's trajectory under the adapter would show that
fine-tuning on safe-walkway footage changes what the editor draws. It would not show
that the result is safe, that the change is causal, or that LD-ViCE has been
reproduced. Both outputs stay **candidate counterfactuals** until a human completes
`REVIEW_TEMPLATE.md` against them, judging the four criteria separately, and explicitly
rejecting the spurious outcome recorded in `configs/experiment.json`: a walkway
repainted, widened or extended underneath a worker who has not moved.
