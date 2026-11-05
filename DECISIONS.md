# Decisions: agent-specified relocation, then frozen CogVideoX refinement

Branch `experiment/manual-path-refinement`, started 2026-11-05 from `main` at `e2946f8`.
Nothing on this branch is committed.

Every entry gives the evidence it rests on, the choice, why, what is uncertain, and any later
correction. Entries are appended in the order the decisions were taken. A decision that turns
out wrong is corrected in place with the correction marked, and the original reasoning stays
readable.

**What this experiment asks.** Earlier attempts asked CogVideoX to invent a relocation of the
worker from a text prompt, with and without classifier gradients and with and without a LoRA
adapter, and the worker never moved. This experiment instead relocates the worker explicitly
by compositing, and then asks a frozen CogVideoX only to refine what is already in the right
place. The comparison is between the rough composite and its refinement, with the untouched
source as the reference.

**What it is not.** Not an LD-ViCE reproduction. Not evidence of causal validity. Not an
automatically discovered classifier explanation. The paths are **agent-specified
interventions**: I proposed them from image evidence, and no part of them was discovered as
safe behaviour by a model or found in the dataset.

---

## D1. Branch and isolation

**Evidence.** `git status` clean on `main` at `e2946f8`, which is pushed to
`origin/main`. `squeue -u $USER` empty, so no job was in flight whose outputs could land
mid-experiment. `git worktree list` shows one worktree.

**Choice.** A branch in the existing working tree, `experiment/manual-path-refinement`, with
no separate worktree.

**Rationale.** A separate worktree isolates files, and the thing that actually needs isolation
here is the Python environment, not the checkout. The repository holds no uncommitted work to
protect: everything is either committed or gitignored. All generated media lands under
`outputs/`, which is gitignored, so no experiment output can be committed by accident.

**Uncertainty.** None material.

---

## D2. Segmentation tool: official SAM 2, not the PyPI package

**Evidence.** The task prefers SAM 2 if practical. `pip index` metadata for the PyPI project
`sam2` version 1.1.0 names `home_page = https://github.com/JinsuaFeito-dev/segment-anything-2`
and `author = Jorge Insua`, and declares no dependencies. That is a third-party fork, not
Meta's release, and installing it would run an unvetted `setup.py`. Meta's own repository at
`github.com/facebookresearch/sam2` is reachable from the login node, declares
`torch>=2.5.1, torchvision>=0.20.1, numpy, tqdm, hydra-core, iopath, pillow`, and ships the
SAM 2.1 configs inside the package directory.

**Choice.** Clone the official repository to
`/ocean/projects/cis260288p/puprety/third_party_ext/sam2_official/sam2_repo` and import it
from `PYTHONPATH` without installing it. Add only `tqdm`, `hydra-core` and `iopath` to a new
environment at `/ocean/projects/cis260288p/puprety/envs/sam2-seg`, which reaches the baseline
environment's `site-packages` through a single `.pth` file so that torch and torchvision are
shared rather than downloaded again.

**Rationale.** Using `PYTHONPATH` rather than an install means no third-party build or install
script executes. The `.pth` link keeps the three new packages out of the baseline environment
while reusing its 3 GB of CUDA wheels. Verified: `sam2.build_sam.build_sam2_video_predictor`
imports in `sam2-seg`, and `import hydra` still fails in `cogvx-cu128`, so the baseline is
untouched.

**Checkpoint.** `sam2.1_hiera_small.pt`, 184,416,285 bytes, sha256
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`, from
`dl.fbaipublicfiles.com/segment_anything_2/092824/`. The small variant is chosen over large
because the worker occupies roughly 50 to 200 px of frame height here, so model capacity is
not the limiting factor, and the smaller model leaves more of the 48 GB card and more of the
budget for the refinement stage.

**Uncertainty.** SAM 2's video predictor has not been run on this footage before. The
fallback, if its masks are unusable, is `torchvision`'s `maskrcnn_resnet50_fpn`, which is
already a trusted dependency. Which backend actually ran is recorded in each clip's mask
manifest, never assumed.

---

## D3. Source selection, frozen before any composite or edit was generated

**Evidence**, all read from measurements written by earlier GPU jobs, not re-derived here:
`outputs/multi_walkway/evaluation.json` (per-frame detector tracks and lane annotations for
`0_tr16`, `0_tr125`, `0_tr48`), `outputs/lora_edit/worker_position.json` (`0_tr143`),
`outputs/multi_walkway/FINDINGS.md` (camera-view clustering and recording-group exclusions),
and `outputs/multi_walkway/selection_record.json`.

| Clip | View | Detector misses in source | Median foot-to-lane | Lane pixels, lane x | Frames on lane |
|---|---|---|---|---|---|
| `0_tr143` | B | 0 of 49 | feet at x 315, lane x>450 | — | 0 of 49 |
| `0_tr16` | A | 0 of 49 | 262.4 px | 4757, x 595-684 | 0 of 49 |
| `0_tr125` | B | 1 of 49 | 394.6 px | 13537, x 541-718 | 0 of 49 |
| `0_tr48` | A | **24 of 49** | 107.7 px | 5926, x 581-686 | 0 of 49 |

**Choice.** Three sources: `0_tr143`, `0_tr16`, `0_tr125`. `0_tr48` is rejected.

**Rationale, per clip.**

- `0_tr143` is the source every earlier attempt used, so a result here is directly comparable
  with four recorded negatives. Its worker is fully inside the frame, feet at row 297 rather
  than at the bottom edge, which is the best case for segmenting a whole body.
- `0_tr16` is camera view A, which no LoRA training footage came from, and its worker is the
  closest of the three to the lane, so it is the easiest transition to make plausible. Zero
  detector misses.
- `0_tr125` is the hardest case: the worker is farthest from the lane at 394.6 px median, and
  the lane is the largest and clearest of the three at 13537 px. It shares camera view B with
  `0_tr143` but is a different recording, and the earlier selection had already excluded every
  clip sharing a recording group with `0_tr143`.

**Rejection, recorded rather than hidden.** `0_tr48` has the shortest distance to cover, which
makes it tempting, and the detector misses the worker in 24 of its 49 frames. An experiment
whose whole method is cutting out a worker and moving him cannot run on footage where the
worker is located in half the frames. Using it would have produced masks with 24 holes and a
composite whose failures could not be attributed. It is excluded for tracking quality, not for
its result.

**Unsafe behaviour survives cropping and sampling.** For all three selected clips the worker
reaches the marked lane in 0 of 49 frames of the *preprocessed* 720x480 49-frame interval,
measured against a lane annotated from those same preprocessed frames. The violation is
present in exactly the footage the pipeline will see.

**Variation covered.** Two camera views, three recordings, and a 132-px spread in how far the
worker has to travel. Three clips support no claim about generalisation, and none is made.

**Fourth clip.** Deferred until the three cases are complete and the remaining budget is
known. `0_tr2` is already preprocessed and is the only other candidate on disk.

**Uncertainty.** `0_tr16` and `0_tr125` both have the worker's feet reaching row 480, the
bottom frame edge, so their silhouettes are clipped by the frame in some frames. A clipped
silhouette can be translated sideways but cannot be moved up the frame without inventing the
part that was never photographed. Each clip's path is constrained accordingly, and D5 records
what that cost.

---

## D4. Lane annotation: green paint is a candidate, not a lane

**Evidence.** `scripts/mpr_annotate.py --mode survey` split each clip's dense-green union
into connected components and drew them numbered and coloured over a source frame, in
`outputs/manual_path/<clip>/overlays/01_green_components.png`. I looked at all three.

| Clip | Components | Chosen | Area | Extent |
|---|---|---|---|---|
| `0_tr143` | 1 | `[0]` | 13761 px | x 540-718, rows 105-479 |
| `0_tr16` | 2 | `[0]` | 4675 px | x 596-684, rows 344-479 |
| `0_tr125` | 1 | `[0]` | 13537 px | x 541-718, rows 109-479 |

**Choice and why.** In `0_tr143` and `0_tr125` the single component is bounded by painted
yellow lines on both sides, runs the length of the aisle, and is continuous with the walkway
visible at the top of the frame. Those two clips share camera view B, and their components
agree to within 224 px of area and a few pixels of extent, which is independent
corroboration that the component is the pedestrian lane and not painted equipment.

In `0_tr16` the second component is 72 px at rows 290-297. It is the same marking much
further down the hall, too small to anchor a polygon and far above the worker's foot rows,
so the lane is annotated only where the evidence is solid and that component is excluded.

**Uncertainty.** The polygon is the convex hull of the chosen paint, so it spans the gaps
between dashes and the bend in the lane, and is larger than the painted area. Foot placement
is therefore scored against the painted mask with a 9x9 window, the same test the earlier
runs used, and the hull is only a planning aid. `0_tr16`'s lane is annotated over 135 rows
against 374 for the other two, so its lane evidence is the thinnest of the three.

---

## D5. Target paths, frozen before any composite existed

**Choice.** Horizontal-only relocation. Each relocated frame keeps the foot row the worker
actually occupied in that frame (`y_mode: follow_source`), and only the horizontal position
changes.

**Rationale.** A silhouette cut from frame *f* is the right size for the depth it was
photographed at. Holding its row fixed means the composite never rescales a person, so no
invented body size enters the result. That matters because the perspective evidence is weak:
the empirical fit of box height against foot row gives R² 0.734 on `0_tr143` but 0.054 on
`0_tr16` and 0.324 on `0_tr125`, the latter with a physically wrong negative slope caused by
the worker's feet reaching the bottom frame edge and the box being clipped there. Relying on
those fits would have scaled two of three workers by a factor derived from noise.

**Cost of that choice, stated plainly.** The intervention cannot move the worker nearer to
or further from the camera, only across the frame.

**The three paths.** Keyframes and rationale are frozen in `configs/manual_path_plan.json`
and the per-frame paths in `outputs/manual_path/<clip>/annotation.json`.

| Clip | Untouched frames | Lateral speed | Worker's own median / max step | Reaches lane | Checks |
|---|---|---|---|---|---|
| `0_tr143` | 0-5 | 9.5 then 3.1 px/frame | 5.79 / 20.01 | frame 40 | no teleport, no machinery |
| `0_tr16` | 0-3 | 6.3 then 0.6 px/frame | 3.54 / 25.36 | frame 40 | no teleport, no machinery |
| `0_tr125` | 0-1 | 14.5 then 0.6 px/frame | 7.20 / 24.50 | frame 40 | no teleport, no machinery |

Each clip keeps its opening frames exactly as shot so the violation is visible before
anything changes. Each path reaches the lane at frame 40 rather than at frame 48: a path
that only touched the marking in its last two frames would score as lane contact without
ever showing the worker walking there.

**Correction made during planning.** The first version of all three paths arrived at the
lane only at frame 48 and gave 2 to 3 frames of lane contact. It also reported
`teleports: true` on every clip, because the step limit was being applied to the *total*
step, which under `follow_source` contains the worker's own vertical jitter copied from the
source track. Both were fixed: the keyframes now reach the lane at frame 40, and
`path_checks` reports the lateral step, which the intervention controls, separately from the
total step, which it does not. The limit is applied to the lateral step only, and the total
is reported beside it.

**Direction, stated per clip.** `0_tr125`'s worker is already walking toward the lane, so the
intervention extends his own direction. `0_tr143`'s crosses the aisle. `0_tr16`'s **reverses**
his lateral direction: he is walking away from the marking, so this is a return toward it.
None of the three is a claim that the clip becomes safe; each is a relocation of one person
inside one annotated region.

**Uncertainty.** Paths are drawn in the image plane of a 720x480 frame. There is no camera
calibration and no ground plane, so no step, distance or clearance here is a real-world
measurement, and placing a silhouette inside the lane polygon says where the worker appears
to stand and nothing about whether walking there is safe. `0_tr125`'s lateral speed of
14.5 px/frame is marginally above his own 90th-percentile step of 15.4 and well below his
maximum of 24.5; it is the price of a 551 px transition in six seconds and is recorded rather
than smoothed away. In `0_tr16` and `0_tr125` the worker's feet sit at or near row 480 for
much of the clip, so those silhouettes are clipped by the frame edge and the composite
inherits that clipping.

## D6. Refinement settings

**Evidence.** The project's own strength sweep on `0_tr143`: strength 0.5 preserves the
scene and changes nothing, 0.8 redraws the room, and the trade-off between the two is
monotonic. In this experiment the relocation has already happened in pixels, so the
generator is not being asked to change anything structural.

**Choice.** Frozen CogVideoX-2b, seed 20261103, strength 0.2 of 50 requested steps, which
executes 10, text guidance 6.0, `CogVideoXDDIMScheduler`, float16. Identical for all three
clips. No LoRA adapter is loaded and no classifier gradient is computed, both verified in
code rather than assumed: the transformer is inspected for LoRA parameter tensors before and
after every generation, and the guidance module is never imported.

**Rationale.** A low strength is what keeps the refinement from undoing the relocation. The
pilot on `0_tr143` was run alone first and inspected before the other two clips were
submitted, which is also what caught that the setting was behaving as intended.

**Uncertainty.** One configuration only. The result shows that this configuration neither
undid nor improved the relocation; it does not show that no configuration could improve it.

---

## D9. Correction: the lane IoU collapse is a colour shift, not a repainting

**First reading, wrong.** Refined lane IoU came out at 0.2461, 0.2896 and 0.0126 against
0.9193 to 0.9523 for the composites. Comparing frames by eye, I judged that the refinement
had enlarged the scene, which would move the paint into different pixels.

**Measurement, which overturned it.** The best uniform scale about the frame centre aligning
refined to composite is exactly 1.000 on all three clips, with no improvement over the
unscaled comparison (`outputs/manual_path/scale_drift.json`). There is no zoom and no
geometric drift.

**What is actually happening.** Measured over the source lane's own pixels, mean
`G - max(R, B)` falls from 16.64, 18.31 and 17.13 to 13.42, 14.15 and 12.05, and the fraction
of lane pixels above the threshold of 15 falls from 0.594, 0.719 and 0.679 to 0.223, 0.281
and 0.113 (`outputs/manual_path/lane_colour.json`). Mean RGB on the lane moves by 4 to 6
units of 255 in the green channel. The marking stays visibly green and visibly in place.

**Why this matters beyond this experiment.** The lane was only 1.6 to 3.3 units above its own
threshold to begin with, so the metric is far more sensitive to a mild desaturation than a
human eye is. Reporting "IoU 0.25" as "the lane was repainted" would have been the same class
of over-claim this project has already had to withdraw once. It is reported as a colour-
fidelity failure, which is what it is.

---

## D10. No fourth clip

**Evidence.** Three cases, two camera views, and a unanimous outcome: compositing improved
foot placement in all three, refinement preserved it in all three and improved none of them.
5.86 SU remain in the budget, so cost is not the constraint.

**Choice.** Stop at three.

**Rationale.** A fourth clip would need its own survey, lane inspection, path design and
three more job submissions, and the only candidate already preprocessed is `0_tr2`, which
shares camera view A with `0_tr16`. It would add a fourth instance of an outcome that is
already unanimous rather than a new kind of variation, and the budget is better kept for the
classifier repair that Gate C still blocks.

**Uncertainty.** Three clips support no claim about generalisation, and the write-up makes
none.

---

## D7. Correction: the first composite run wrote colour-inverted video

**Evidence.** Job 50275552 completed in 3 minutes 40 seconds and reported healthy masks,
49 of 49 frames on all three clips. The composite frames looked wrong on inspection: the
green lane rendered pink and the yellow boundary lines rendered blue. Reading a written clip
back and comparing it against the preprocessed source settled it. Mean absolute difference
against the source was 84.63 per channel; against the *inverted* source it was 3.10, which is
ordinary H.264 loss.

**Cause, verified not guessed.** `diffusers.utils.export_to_video` rescales a list of numpy
frames with `frame * 255`. That is correct for float frames in [0, 1] and silently
catastrophic for uint8, where the multiply wraps modulo 256 and yields `256 - v`, an exact
colour inversion. Every earlier script in this project passed PIL images straight out of the
pipeline and never hit it.

**Choice.** `save_video` now converts to PIL images before writing, and job 50276648 re-ran
the whole stage. The inverted outputs were not patched by subtracting from 255: re-running
costs 0.0611 SU and also re-verifies the pipeline, where arithmetic on an already-compressed
inversion would leave a result nobody could check.

**Why this is recorded rather than quietly fixed.** The masks and the compositing logic were
correct throughout. Had the inversion not been caught, the refinement stage would have run
on inverted frames, CogVideoX would have produced something plausible-looking from them, and
the comparison would have been meaningless while every number in it stayed finite. The cost
of the miss was 0.0611 SU and the cost of catching it was looking at the picture.

## D8. Budget

Spent on this experiment so far: 0.0611 SU (job 50275552, superseded) plus job 50276648.
Against the 10-SU classifier-experiment budget, 6.1117 SU remained before this experiment
started. Every job prints its requested resources and maximum charge in its sbatch header:
segmentation 0.667 SU maximum, refinement 0.583, evaluation 0.333.
