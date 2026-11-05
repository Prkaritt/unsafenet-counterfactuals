# Pre-registered acceptance criteria — classifier-guided counterfactual experiment

Written 2026-11-04, **before any classifier is trained**, so that the thresholds cannot be
chosen after seeing results. Every gate below is pass or fail. A failed gate stops all
dependent execution and the diagnostics are preserved.

The experiment has four stages. Each stage runs only if the stage before it passed.

| Stage | Gate | Runs only if |
|---|---|---|
| A | Data integrity | always |
| B | Classifier validation | A passed |
| C | Shortcut controls | B passed |
| D | Guided-step probe | C passed |
| E | One guided generation | D passed |

---

## The task

Binary classification, class 0 *Safe Walkway Violation* against class 4 *Safe Walkway*.
Binary rather than 8-way because only these two classes are relevant to the counterfactual,
and because a model that must also separate forklifts and panel covers spends capacity on
distinctions that cannot help here. The other six classes are excluded from training and
from every metric.

Model: `torchvision.models.video.r3d_18`, initialised from `R3D_18_Weights.KINETICS400_V1`,
final `fc` replaced with 2 outputs. Verified on 2026-11-04 that the architecture accepts
input of shape `(1, 3, 49, 480, 720)` because it ends in `AdaptiveAvgPool3d`, so the
Kinetics 112x112 crop is not forced on us. 33.4M parameters.

CogVideoX weights stay frozen throughout. No LoRA training.

---

## Stage A — data integrity

Three failure modes are audited, in this order.

**A1. Exact duplicates across splits.** sha256 of every clip. Zero clips may appear in more
than one split. The Mendeley API publishes a sha256 per file and the downloader verifies
each one, so this is cheap and exact. Already checked within the class-0 training folder:
178 files, 0 duplicate hashes.

**A2. Near-duplicates across splits.** Exact hashing does not catch two clips cut from the
same continuous recording a few seconds apart, which is the realistic leak in a dataset
built from security-camera footage. Audit by perceptual hash of sampled frames; any
cross-split pair below the distance threshold is a failure and is resolved by moving the
whole group to one side.

**A3. Recording-group isolation for the source clip.** `0_tr143.mp4` and every clip that
shares its recording — same camera, same aisle, same session — must appear in **no** split.
They are removed from training, validation and test entirely. Grouping is by scene
similarity on background frames, not by filename index, because consecutive indices are not
a reliable indicator of a shared recording and a shared recording is not reliably
consecutive.

**Gate A passes when:** zero exact duplicates across splits, zero near-duplicate pairs
across splits, and the `0_tr143` recording group is absent from all three splits. The group
membership is written out for inspection, since an over-wide group costs training data and
an over-narrow one leaks the target scene.

---

## Stage B — classifier validation

All decisions — architecture, learning rate, epochs, augmentation, early stopping, input
resolution — are made against the **validation** split. The test split is evaluated once,
at the end, and never tuned against.

Reported for both validation and the final single test evaluation:

- Per-class precision, recall and F1 for class 0 and class 4.
- The 2x2 confusion matrix, as counts.
- Balanced accuracy.
- Original-versus-VAE prediction consistency (B3 below).

**B1. Balanced accuracy on validation at least 0.80.** Below that the classifier is too weak
to steer anything, whatever its average accuracy looks like.

**B2. Each per-class recall at least 0.75.** A model that reaches good balanced accuracy by
being excellent on one class and poor on the other is useless as a guidance target, because
guidance pushes toward class 4 specifically.

**B3. Original-versus-VAE prediction consistency at least 0.90 agreement, and mean absolute
change in p(class 4) at most 0.10.** This gate exists because guidance does not see the
source video. It sees pixels decoded from latents through the CogVideoX VAE. If the
classifier's prediction changes when a clip passes through the VAE untouched, it is reading
codec and autoencoder artifacts, and the gradient it supplies is a gradient about artifacts.
The measured VAE round trip is 32.60 dB PSNR, so this gate should be comfortable to pass; if
it is not, that is decisive information.

**Gate B passes when:** B1, B2 and B3 all hold on validation.

---

## Stage C — shortcut controls

Saliency maps are **not** used as a pass/fail test. They are suggestive, not decisive, and a
map that looks reasonable does not establish that the model would fail without that region.
The controls below are interventional: they change the input and measure what the prediction
does.

All of Stage C is computed on the held-out class-0 and class-4 **test** subset only, in a
single evaluation pass, together with Stage B's final test numbers.

**C1. Worker occlusion — the necessity test.** Mask the worker, leaving the scene intact.
If the classifier still separates the classes without the worker visible, it is not reading
the behaviour at all.

Pass: balanced accuracy drops by at least 0.20 absolute when the worker is masked.
Fail: accuracy is roughly unchanged, which means the label is being read off the scene.

**C2. Walkway-paint occlusion — the sufficiency check.** Mask the green walkway region,
leaving the worker intact. The class distinction is *relational*, so masking either side of
the relation should hurt.

Reported, and read together with C1. A large C2 drop with a small C1 drop is the specific
signature of the shortcut this project most fears: the model reading paint rather than
people. That combination fails Stage C regardless of C2's size.

**C3. Matched-scene evaluation — the decisive test.** Class 0 and class 4 clips that share a
camera and scene. Within such a pair the scene is constant and only the behaviour differs,
so a model that has learned scene identity instead of behaviour cannot separate them.

Pass: balanced accuracy on the matched-scene subset at least 0.75.

If no matched-scene pairs exist in the dataset, C3 cannot be run. That is itself a finding
and is recorded as such; Stage C then passes on C1 and C2 only, and every later claim is
qualified by the absence of C3. This is not a reason to substitute a weaker test.

**Gate C passes when:** C1 passes, the C1-and-C2 pattern does not show the paint-shortcut
signature, and C3 passes where it can be run.

---

## Stage D — guided-step probe

One denoising step, instrumented. No full generation.

**D1. Gradients are finite.** No NaN and no Inf anywhere in the gradient reaching the
optimised latent.

**D2. Gradients are nonzero.** The gradient norm at the latent is strictly greater than
zero. A frozen classifier must not block input gradients; frozen weights and a detached
graph are different things and this gate distinguishes them.

**D3. The update direction is correct.** Taking one guidance step must increase p(class 4)
as predicted by the gradient. Measured directly: evaluate the classifier before and after
applying one step, and check the sign. A gradient that is finite, nonzero and pointing the
wrong way is worse than no gradient.

**D4. Memory is measured, not assumed.** Peak GPU memory for the guided step is recorded.
The no-grad VAE round trip measured 21.611 GiB and the unguided generation peaked at 36.205
GiB on a 44.42 GiB card, so the backward path through VAE decoding has little headroom. If
the probe exhausts memory, the recorded blocker is the deliverable for this stage and
Stage E does not run.

The transformer's backward graph is **not** retained unless the guidance algorithm being
implemented requires it. Guidance on the decoded prediction does not require it; retaining
it anyway would waste the headroom D4 is measuring.

**Gate D passes when:** D1, D2 and D3 hold and D4 fits in the available memory.

---

## Stage E — one guided generation

A single classifier-guided candidate for `0_tr143`, compared against the matching unguided
baseline.

Held identical between the two: source clip, seed 20261103, checkpoint
`THUDM/CogVideoX-2b` revision `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01`, prompt, negative
prompt, scheduler, dtype, strength, guidance scale, steps, resolution and frame count. The
only difference is the classifier term. Anything else differing makes the comparison
worthless.

Exported for human review: both videos, a comparison contact sheet, PSNR and mean absolute
difference against the source, classifier probabilities on source, on VAE reconstruction,
on the unguided baseline and on the guided candidate, peak memory, timing, and a full run
manifest.

**Stage E has no pass/fail gate.** It produces a candidate, not a verdict.

---

## What none of this establishes

A classifier prediction flip is not corrected behaviour and is not safety. The absence of
*Safe Walkway Violation* is not the presence of *Safe Walkway*. A guided candidate that
scores p(class 4) = 0.99 while the worker stands exactly where he started, on a floor the
model has painted green, has flipped the prediction and changed nothing that matters. That
specific outcome is written into `configs/experiment.json` as the spurious change to reject,
and Stage C exists to make it detectable in advance rather than after the fact.

Every output stays a **candidate counterfactual** until a human completes
`templates/REVIEW_TEMPLATE.md` against it. Criterion 4 of that form records a prediction
flip as one signal among four, never as the verdict.

---

## Addendum, 2026-11-04 — findings from implementation, before any training

Recorded here because each one changes what a later gate means.

**The classifier sees 16 of the clip's 49 frames.** `unsafenet_cf.classifier.prepare`
selects 16 evenly spaced frames, so a gradient taken at the classifier's output reaches
those 16 and is exactly zero on the other 33. Verified on CPU: gradient finite, norm 0.238,
non-zero on 16 frames per sample.

This is correct — the classifier genuinely did not look at the other frames — but it means
naive guidance would edit a third of the clip and leave the rest to the prompt. The guidance
loop therefore varies the sampling offset between denoising steps, so that over a 30-step
run every frame receives gradient on some steps. The alternative, feeding all 49 frames to
the classifier, costs memory that Stage D has little of.

Gate D2 is read accordingly: the gradient must be non-zero at the latent, and the frames
that receive it must change between steps.

**Input geometry is 16 x 240 x 320, not the Kinetics 112 x 112 crop.** r3d_18 ends in
`AdaptiveAvgPool3d` and accepts any spatial size. 320x240 preserves the 3:2 aspect of the
720x480 preprocessed clip, so a foot near a painted line stays several pixels wide. The same
`prepare` function is used by training, evaluation and guidance, so they cannot drift apart.

**Occlusion thresholds in Stage C are diagnostics, not a single pass/fail.** As instructed,
the C1 and C2 numbers are reported together with a **control mask**: a mask of equal area
placed over a region irrelevant to the label. If the control mask moves predictions as much
as the worker mask does, the measured drop is a masking artifact rather than evidence about
what the model reads, and neither C1 nor C2 supports a conclusion. Both the worker and the
lane carry information the intended decision depends on, so a drop under either mask is
expected; the diagnosis rests on the pattern across worker, lane and control, not on one
number crossing one line.

Likewise a single prediction change under VAE reconstruction does not establish shortcut
learning. Gate B3 is an aggregate over the validation split — agreement rate and mean
probability shift — not a verdict drawn from one clip.

**Gate A now fails on any undecodable clip.** A clip that cannot be decoded was never
hashed, so the cross-split overlap check never ran for it. Passing Gate A in that state
would claim a check that did not happen. Gate A also fails if both binary labels are not
present.
