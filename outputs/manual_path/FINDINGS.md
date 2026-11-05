# Agent-specified relocation, then frozen CogVideoX refinement

**Date:** 2026-11-05. **Outcome: the compositing worked on all three clips; the refinement
did not improve it and measurably degraded the lane marking's colour.**

This is the first time in this project that a worker has reached the marked pedestrian lane
in any clip. It happened because the relocation was done explicitly in pixels, not asked for
in a prompt. That is a result about the method, not about safety, and not about CogVideoX.

**What this is not.** Not an LD-ViCE reproduction. Not evidence of causal validity. Not an
automatically discovered classifier explanation. The paths were proposed by me from image
evidence and are labelled throughout as **agent-specified interventions**. No clip here is a
verified counterfactual: every one remains a candidate until a human completes
`templates/REVIEW_TEMPLATE.md` against all four criteria.

---

## The question

Four earlier attempts asked a model to invent the relocation: prompt-only editing across a
strength sweep, classifier-gradient guidance, a LoRA fine-tune, and a three-source
replication of that fine-tune. In every one of them the worker reached the lane in **0 of 49
frames**. This experiment removes the invention step. The worker is segmented, the background
behind him is recovered, and his own silhouette is composited along a path that I specified.
A frozen CogVideoX-2b is then asked only to refine what is already in place.

---

## Results

Foot-to-lane distance is measured from the tracked worker's foot point to the nearest painted
lane pixel. The lane is annotated once from each source and applied unchanged to all three
arms, so no arm can satisfy the measure by moving the paint.

| Clip | Arm | Foot-to-lane median, px | After intervention | Frames on lane | Detector misses | Lane IoU |
|---|---|---|---|---|---|---|
| `0_tr143` | source | 291.0 | 305.7 | 0/49 | 0 | 1.0000 |
| | composite | **123.3** | **97.8** | **11/49** | 0 | 0.9523 |
| | refined | 123.5 | 97.4 | 11/49 | 0 | 0.2461 |
| `0_tr16` | source | 263.4 | 267.4 | 0/49 | 1 | 1.0000 |
| | composite | **91.0** | **79.6** | **11/49** | 1 | 0.9432 |
| | refined | 90.4 | 78.6 | 10/49 | 1 | 0.2896 |
| `0_tr125` | source | 387.1 | 379.6 | 0/49 | 5 | 1.0000 |
| | composite | **214.4** | **200.2** | **9/49** | 1 | 0.9193 |
| | refined | 162.6 | 156.2 | 9/49 | **9** | 0.0126 |

### Verdict per case, compositing and refinement judged separately

These are different successes and conflating them is how this project has gone wrong before.

| Clip | Compositing | Refinement, against the composite |
|---|---|---|
| `0_tr143` | **improved** — 291.0 to 123.3 px, 0 to 11 frames on lane | **unchanged** on placement (123.3 to 123.5), **worsened** on lane colour |
| `0_tr16` | **improved** — 263.4 to 91.0 px, 0 to 11 frames on lane | **unchanged** on placement (91.0 to 90.4, 11 to 10 frames), **worsened** on lane colour |
| `0_tr125` | **improved** — 387.1 to 214.4 px, 0 to 9 frames on lane | **mixed, read as worsened** — placement improves (214.4 to 162.6) but the detector loses the worker in 9 frames against 1, and lane colour degrades most of the three |

The refinement preserved the relocation in all three cases. That is worth stating plainly,
because the pipeline enforces no trajectory and no mask constraint, so it was an open
question and not a guarantee: a low-strength video-to-video pass at 10 executed steps did not
drag the worker back toward where the source had him.

---

## The lane IoU collapse is a colour shift, not a repainting

This needs care, because "lane IoU 0.25" reads like the spurious outcome this project exists
to reject, and that is **not** what happened.

**First reading, wrong.** Looking at composite and refined frames side by side, I judged that
the refinement had zoomed the scene, which would move the paint into different pixels and
collapse the IoU.

**Measurement, which overturned it.** Searching for the uniform scale about the frame centre
that best aligns the refined frame to the composite returns **exactly 1.000** on all three
clips, with no improvement over the unscaled comparison. There is no zoom and no geometric
drift. Recorded in `scale_drift.json`.

**What is actually happening.** The project's paint test is a channel-difference test:
`G - max(R, B) > 15`, kept only where half of a 15x15 neighbourhood passes. Measured over the
source lane's own pixels on frame 24:

| Clip | Arm | Mean G − max(R,B) on the lane | Fraction above the threshold | Mean RGB on the lane |
|---|---|---|---|---|
| `0_tr143` | source | 16.64 | 0.594 | 54.5, 121.2, 103.5 |
| | composite | 16.37 | 0.586 | 54.6, 121.1, 103.7 |
| | refined | 13.42 | 0.223 | 51.6, 117.0, 102.5 |
| `0_tr16` | source | 18.31 | 0.719 | 52.6, 104.5, 83.8 |
| | refined | 14.15 | 0.281 | 52.5, 98.8, 82.2 |
| `0_tr125` | source | 17.13 | 0.679 | 48.9, 114.3, 96.6 |
| | refined | 12.05 | 0.113 | 47.0, 109.3, 96.6 |

The green channel falls by 4 to 6 units of 255 and the marking stays visibly green and
visibly in place. That small shift is enough to take most of the lane below a threshold the
lane was only 1.6 to 3.3 units above to begin with. So the honest statement is: **the
refinement desaturates the paint, and the lane metric is more sensitive to that than a human
eye is.** It did not repaint, widen, erase or relocate the marking.

**A caveat this raises about earlier work in this project.** The multi-source LoRA comparison
reported base-edit lane IoUs of 0.0000, 0.0017 and 0.0120 at strength 0.6 and read them as
the marking being erased or relocated. Those edits were six times stronger than this one, so
erasure there is plausible, but the same desaturation mechanism would also drive IoU toward
zero. That earlier reading has not been re-measured and should be treated as unconfirmed on
this point until it is.

---

## What the composite can and cannot do

Stated as properties of the method, because they are visible in the clips and no amount of
tuning removes them.

* **Identity is preserved exactly, because the pixels are the worker's own.** This is the
  sharpest contrast with the LoRA run, where the worker's navy overalls with red shoulder
  panels became a grey top in both arms and the face was smeared. Here the overalls, the red
  panels and the chest logo arrive at the lane intact.
* **No body turn, no new gait, no new viewing angle.** The worker arrives at the lane facing
  the way he was photographed walking somewhere else. In `0_tr16` the intervention reverses
  his lateral direction, so this is most visible there.
* **No cast shadow at the destination,** and the original contact shadow is removed only as
  far as the 7-px erase dilation reaches. In `0_tr143` a dark patch remains near the coil
  where the worker stood; the refinement did **not** repair it.
* **No occluder is modelled.** All three paths were drawn over open floor so that nothing
  should pass in front of the relocated worker.
* **The background estimate is a plausible still, not an empty room.** It is a per-pixel
  temporal median with every detected person withheld, so machinery that moves is flattened
  to whichever state held for more than half the clip.

---

## Settings, as executed

Identical for all three clips. Frozen `CogVideoX-2b`, revision
`1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01`. **No LoRA adapter was loaded** — the transformer
was inspected before and after every generation and carried 0 LoRA parameter tensors. **No
classifier gradient** — the guidance module is never imported.

Seed 20261103, strength 0.2, 50 requested steps of which **10 executed**, text guidance 6.0,
`CogVideoXDDIMScheduler`, float16, 49 frames at 720x480. Generation 40.3 to 46.0 s per clip
at about 4.0 to 4.6 s per executed step, peak 27.84 GiB.

Segmentation: SAM 2.1 `hiera_small`, sha256 `6d1aa6f3…`, seeded from the tracked worker's
detector box and propagated. 49 of 49 frames segmented on every clip, median mask area 6051,
4202 and 7707 px. No fallback to Mask R-CNN was needed. Peak GPU 1.707 GiB.

---

## Uncertainty, and what must not be concluded

* **Image-plane only.** There is no camera calibration and no ground plane. No distance here
  is a real-world measurement, and standing on a marked lane is not proof of safe walking.
* **Three clips, two camera views, one path per clip.** That supports no claim about
  generalisation and none is made.
* **One refinement configuration.** One strength, one seed, one step count. The run shows
  that *this* configuration neither undid nor improved the relocation. It does not show that
  no configuration could improve it.
* **Tracking uncertainty.** The detector missed the worker in 1 of 49 source frames on
  `0_tr16` and 5 on `0_tr125`, and in 9 frames of the refined `0_tr125`. Misses are counted,
  never interpolated.
* **Review basis.** Everything above is measured from decoded frames, plus my own inspection
  of sampled frames and contact sheets. **Full playback review has not been done**, and the
  human review form is unsigned. Identity, gait, shadow quality and occlusion are assessed by
  eye from sampled frames only, which is exactly the basis that proved insufficient in the
  LoRA run.
* **Classifier scores were not computed** for these clips. They would be a secondary
  diagnostic at best: the VAE round trip alone already flips this classifier's prediction,
  and Gate C still fails, so no probability it reports is evidence about the worker.

---

## Cost

0.2497 SU across six jobs, against 6.1117 SU remaining in the 10-SU experiment budget when
this started. Per-job figures are in `docs/SU_LEDGER.md`, read from `sacct`.

One job, 50275552 at 0.0611 SU, was superseded: it wrote colour-inverted video because
`export_to_video` rescales numpy frames by 255 and uint8 input wraps modulo 256. The masks
and compositing in it were correct. It is kept on the record rather than deleted, and the
cause is written up in `DECISIONS.md` D7.
