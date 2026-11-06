# Panel-cover counterfactual: findings

Branch `experiment/panel-cover-counterfactual`, uncommitted. PSC Bridges-2, `GPU-shared`,
1 x L40S-48 per job. Run on 2026-11-05. Every number here comes from a file in
`outputs/panel_cover/`; the decision history, including the wrong turns, is in `DECISIONS.md`
sections P1 to P13.

## The one-paragraph version

Classifier-guided generation produced a confident label flip **without closing the panel**,
in all three independent cases. A mask-restricted composite that genuinely closed the panel
moved the same classifier barely at all. The classifier's confidence ran opposite to the
physical change it is supposed to track, within one experiment, measured on one proxy. The
prompt-only baseline never closed the panel either. The intervention that worked was the one
with no model in it.

## What the two classes actually are

Verified against the paper before any data was touched, and recorded in
`docs/PANEL_COVER_DEFINITIONS.md` with the retrieval date. Class 2 is **Opened Panel Cover**,
unsafe; class 6 is **Closed Panel Cover**, safe. The paper's own Figure 9(a) shows a closed
panel inside a clip labelled with a different unsafe class, so **closing a panel in a clip is
not evidence that the clip became safe**, and nothing below should be read as a safety claim.

Folder counts match the paper's Table 2 exactly: 129 class-2 and **19** class-6 training
clips, every file sha256-verified. The official test split was never touched.

## Finding 1: the labelled object is an electrical cabinet door, found from the populations

A first identification from a single clip pair named a wire-mesh press guard. Measuring that
box over the whole folder disproved it: **96 of 120** aligned class-2 clips had that guard
closed. The replacement method averages each class's per-clip temporal medians over all
aligned clips and differences the two averages, so whichever worker happened to be in one
clip averages away. The surviving region is an electrical cabinet door low on the press
column, open across the class-2 average and closed across the class-6 average.

| | |
|---|---|
| Population difference in the region | 59.2 of 255 |
| Per-clip separation, class 2 | -0.530 +- 0.486 |
| Per-clip separation, class 6 | +0.754 +- 0.007 |
| Balanced accuracy, clip by clip | **0.935** |
| Class-6 clips, distinct recording groups | 12 clips, **11 groups** |

Scored over the whole folder afterwards: class-2 clips read open 0.81 / closed 0.09, class-6
clips read closed 0.998. **12 of 100** aligned class-2 clips read closed, so roughly one
class-2 clip in eight is labelled for something this door does not explain. None was selected.

## Finding 2: the composite closed the panel cleanly, with no model involved

Three cases, three source recording groups, three different donors, no shared footage.

| case | landmarks | residual rms | donor coverage | px changed outside region | region movement, source to composite |
|---|---|---|---|---|---|
| 1 | 6/6 | 0.0 px | 1.0000 | 0 | 4.175 to 0.156 |
| 2 | 6/6 | 0.0 px | 1.0000 | 0 | 3.709 to 0.130 |
| 3 | 6/6 | 0.0 px | 1.0000 | 0 | 4.937 to 0.146 |

Panel-state proxy: the composite region correlates **0.9997** with the closed reference in
all three cases, against 0.113 to 0.186 for the sources. Fidelity outside the region is 35.0
to 35.6 dB.

**This is mask-restricted compositing, not mask-conditioned generation.** No model was loaded
at this stage. The panel is closed in every frame; **no closing motion was generated**, and
no case may be described as one.

## Finding 3: the prompt alone never closed the panel

`base_edit` is CogVideoX asked by prompt to close the panel, no composite, no mask, 30
executed steps at strength 0.6.

| case | ncc vs open | ncc vs closed | PSNR vs input |
|---|---|---|---|
| 1 | 0.857 | 0.063 | 20.33 dB |
| 2 | 0.854 | 0.091 | 20.47 dB |
| 3 | 0.863 | 0.063 | 20.63 dB |

The panel stays open while the rest of the frame degrades to about 20 dB. It changed the
picture everywhere except the thing it was asked to change. Reported as a negative result,
not tuned until it passed.

## Finding 4: the classifier does read the panel region

Masking the panel box, **1.03%** of the classifier's 240x320 frame, against three equal-area
control regions:

| class | n | panel | mirror_x | mirror_y | centre |
|---|---|---|---|---|---|
| 2, Opened | 26 | **0.1311** | 0.0123 | 0.0132 | 0.0127 |
| 6, Closed | 3 | **0.1318** | 0.0174 | 0.0056 | 0.0188 |

About ten times every control, with the correct sign in both directions. So the classifier is
not indifferent to the panel, which is what makes Finding 6 a problem rather than a
curiosity.

## Finding 5: control arms move the classifier as much as the real change does

P(closed) by arm. The panel column states whether that arm actually closed the panel, from
Finding 2's proxy rather than from the classifier.

| arm | case 1 | case 2 | case 3 | panel closed? |
|---|---|---|---|---|
| source | 0.133 | 0.142 | 0.149 | no |
| `vae_source` | 0.223 | 0.250 | 0.207 | **no** |
| `base_edit` | 0.297 | 0.412 | 0.360 | **no** |
| `unguided_control` | 0.286 | 0.405 | 0.365 | **no** |
| composite | 0.321 | 0.342 | 0.309 | **yes** |
| `refined` | 0.470 | 0.543 | 0.453 | yes |
| `refined_restricted` | 0.337 | 0.353 | 0.327 | yes |

A VAE round trip that changes no panel moves P(closed) by +0.07 to +0.11. A prompt-only edit
that leaves the panel open moves it by +0.16 to +0.27, **more than the composite's +0.17 to
+0.20 while the composite genuinely closed the panel**. `refined` falls from 0.470 to 0.337
once its pixels outside the region are restored, so most of that arm's rise came from outside
the panel. Real class-6 clips score 0.76 to 0.87; the composite at 0.31 to 0.34 is not read
as a closed-panel clip.

No flip in this table may be attributed to panel state, because the controls already move.

## Finding 6: guidance flips the label without closing the panel, three times out of three

Guided and unguided runs share seed 20261105, strength 0.6, 30 executed steps, guidance scale
6.0, prompt and initial noise. The classifier term is the only difference, 0.02 against 0.0.

| case | input P | unguided P | guided P | guided ncc vs **closed** | guided PSNR |
|---|---|---|---|---|---|
| 1 | 0.1325 | 0.2835, no flip | **0.99989**, flip | **0.109** | 20.06 dB |
| 2 | 0.1420 | 0.3611, no flip | **0.99992**, flip | **0.142** | 20.11 dB |
| 3 | 0.1487 | 0.3254, no flip | **0.99990**, flip | **0.099** | 20.38 dB |

The guided panel correlates 0.80 to 0.82 with the **open** reference, where the source sits.
**The panel never closed in any guided output.** The gradient was finite and applied at all
30 steps in all three runs.

The attribution rule is satisfied: the unguided control never flipped, so the flip belongs to
the classifier term. That makes it a sound statement about the classifier and no statement at
all about the panel.

Set against each other, the classifier's confidence runs backwards:

| arm | panel, ncc vs closed | P(closed) |
|---|---|---|
| composite | **0.9997**, genuinely closed | 0.31 to 0.34 |
| guided | **0.099 to 0.142**, still open | **0.9999** |

Three independent cases land within 0.00003 of the same near-certain value, which is an
optimisation saturating its objective rather than three scenes changing. The guided fidelity
of 20.06 to 20.38 dB sits in the `base_edit` band, not near the composite's 35 dB.

**This is an adversarial counterfactual.** The optimisation found a cheaper route to the label
than the physical change the label is supposed to track.

## What this does and does not support

It supports: a changed classifier prediction is not evidence of a changed scene, and on this
dataset the two can point in opposite directions in the same experiment. The honest
intervention here was the one with no model in it.

It does not support: any claim that a scene became safe, that a door-closing action was
generated, that this reproduces LD-ViCE, or that the classifier is bad in general. It also
does not establish why the composite fails to convince a classifier that demonstrably reads
the region. One candidate is that the donor patch is a temporal median and perfectly still,
27 times stiller than its surroundings, while a real closed panel carries sensor noise. **That
is a hypothesis.** Testing it needs a `per_frame` donor arm, which the compositing script
supports and which was not run.

## Limitations, stated rather than smoothed

1. **19 class-6 training clips exist in total.** After recording-group separation the
   validation split holds **4** independent safe clips. Validation balanced accuracy 0.9849
   was identical at all 25 epochs because with 4 clips the metric takes few discrete values.
   Every classifier number here has a wide interval that no amount of training fixes.
2. Only 3 class-6 clips reached the region-masking table in Finding 4.
3. The panel region is 2.22% of the generated frame and 1.03% of the classifier's frame.
4. Each clip contributed to the class mean it was scored against, 1% and 8% respectively, so
   the separation in Finding 1 is optimistic.
5. The region is a convex hull of a thresholded difference, not a traced door outline, and
   the threshold choice is argued in DECISIONS P8 rather than being obvious.
6. Guidance was run from the `source` arm only. Guiding from the composite was not tried.
7. One acceptance criterion failed as written: seam step against a flat-scene control. The
   criterion was the wrong test and the right comparison is reported beside it in P8.
8. Everything rests on one camera and one machine.

## Cost

2.1686 SU over 18 jobs. Of that, **0.2428 SU bought negative or void results**: the
disproved panel identification (0.1817), a diagnostic that measured nothing through a code
bug (0.0603), and a job submitted without its environment variable (0.0008). They are
itemised in `docs/SU_LEDGER.md` rather than folded into the total.

## Where things are

| | |
|---|---|
| Per-case videos, overlays, manifests | `outputs/panel_cover/<case>/` |
| Guidance runs and the gradient probe | `outputs/panel_cover/<case>/guided/` |
| Evaluation of every arm | `outputs/panel_cover/evaluation.json` |
| Classifier and its report | `outputs/panel_cover/classifier/` |
| Diagnostics | `outputs/panel_cover/diagnostics.json` |
| Population difference images | `outputs/panel_cover/population/` |
| Job ledger | `outputs/panel_cover/JOBS.tsv` |
| Decision history, including the wrong turns | `DECISIONS.md`, P1 to P13 |

The clips show identifiable factory workers. Nothing here has been uploaded anywhere, and
the review package is staged for local viewing only.

## Decision and correction record

Kept here so this file stands alone. The full text of each entry, with the evidence that
forced it, is in `DECISIONS.md` under the same labels.

| | Decision or correction |
|---|---|
| P1 | Plain branch `experiment/panel-cover-counterfactual` off the manual-path branch, no worktree: no jobs were running at the time. |
| P2 | Class 2 and class 6 definitions verified against the paper before any data was touched, with the retrieval date. Two errors in the project's own records corrected: the first author's name, and a declared 24 fps against a measured ~25. |
| P3 | Mendeley's root listing is not reachable; eight routes recorded as failed. Resolved by asking for one file link per folder, which the task permitted. |
| P4 | 129 class-2 and 19 class-6 clips obtained, matching the paper's Table 2, every file sha256-verified. |
| P5 | **Wrong.** A wire-mesh press guard identified as the labelled object, from a single clip pair. |
| P6 | **Correction to P5.** Measured over the folder, 96 of 120 aligned class-2 clips had that guard closed. P5 generalised one true observation to 129 clips without measuring it. Cost 0.1817 SU, and it prevented three composites over an already-closed region. |
| P7 | The labelled object is an electrical cabinet door, found from the class populations rather than a pair: balanced accuracy 0.935 clip by clip, with the 12 class-6 clips spanning 11 recording groups. |
| P8 | The region is a convex hull of the thresholded class difference, not a rectangle. Threshold 20 of 255, chosen because 15 merges the door with the P5 mesh guard and 25 clips the door. **Acceptance criterion 5 failed as written** and is recorded as failed, with the comparison it should have specified reported beside it. |
| P9 | The classifier reads the panel region, yet the composite that closes the panel barely moves it. The temporal-median stiffness explanation is labelled a hypothesis, with the untested arm named. |
| P10 | **Correction.** The region-masking diagnostic skipped all 37 clips on a stale key lookup while exiting 0 and writing its file. 0.0603 SU produced an empty main section. |
| P11 | **Correction.** The gradient probe's NaN came from its own default step of 50 times the latent norm, not from the gradient. Replaced with a sweep. |
| P12 | Guidance flips the label without closing the panel, in all three cases. The main finding. |
| P13 | **Correction.** The evaluation script and the review page both omitted the guided arm. Left alone, the guided run's only number would have been the classifier's own 0.9999, reported by the script under test. |

## Known gap against the task specification

The task asked to **quarantine ambiguous sampled intervals**. Overlapping footage was kept
within one split by recording group, but clips whose *sampled window contradicts their label*
were never quarantined. Twelve class-2 clips read closed in the panel region, and they
entered the classifier:

| split | clips | share |
|---|---|---|
| training | `2_tr1`, `2_tr110`, `2_tr125`, `2_tr42`, `2_tr46`, `2_tr47`, `2_tr88`, `2_tr89`, `2_tr90` | 9 of 98 |
| validation | `2_tr124`, `2_tr5`, `2_tr75` | 3 of 37 |

Finding 4 reported those three validation clips as reversing sign under masking and read that
as consistent. That reading was correct as far as it went, but it should have been treated as
a quarantine trigger rather than a curiosity.

**Effect on Finding 6: none.** It compares a guided run against its own unguided control at
the same seed and reads the panel from a region proxy, not from the classifier; neither
depends on which clips trained the classifier. **Effect on Findings 4 and 5: unquantified.**
Closing the gap properly means retraining and redoing diagnostics, evaluation and the three
guided pairs, about 1.0 SU. Not done.

Two further items were left open and are stated above in context: the `per_frame` donor arm
that would test the stiffness hypothesis, and guidance started from the composite arm.
