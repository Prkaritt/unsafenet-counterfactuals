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

---

# Decisions: panel-cover counterfactual, class 2 against class 6

Branch `experiment/panel-cover-counterfactual`, taken from
`experiment/manual-path-refinement` at `b8ee99e`. Started 2026-11-05. Nothing in this
section is committed, pushed, merged or tagged.

## P1. Branch base and isolation

**Evidence.** `git status --short` was empty and the working tree was at `b8ee99e`, the tip
of `experiment/manual-path-refinement`, already pushed to `origin`. `git worktree list`
showed one worktree. `squeue` carried no running jobs from the previous experiment.

**Choice.** A plain branch off `experiment/manual-path-refinement`, in the same working
directory. No separate worktree.

**Rationale.** A worktree buys isolation from concurrent jobs, and there are none. The
previous experiment's six jobs have all completed. The panel-cover work also builds directly
on that branch's code, in particular `src/unsafenet_cf/grouping.py` for recording identity
and `src/unsafenet_cf/manualpath.py` for numpy-only geometry, so sharing one tree avoids
keeping two copies of a changing library in step.

**Uncertainty.** If a long training job and an interactive edit later need to overlap, a
worktree can still be added without rebasing anything.

## P2. Class definitions verified before any data was requested

**Evidence.** Table 1 of the paper was fetched today and gives class 2 as *Opened Panel
Cover*, unsafe, and class 6 as *Closed Panel Cover*, safe, matching the provisional names in
the task. The body text defines the pair as "leaving panel covers open/forgetting them"
against "closed panel covers". Full record, with every source and retrieval date, in
`docs/PANEL_COVER_DEFINITIONS.md`.

**Choice.** Proceed with class 2 as the unsafe source class and class 6 as the safe donor
class, and state in every output that the intervention changes an **object state**, not a
behaviour.

**Rationale.** This pair is structurally unlike class 0 / class 4. There the label is a
property of a person's position, so the intervention had to move a person. Here the label is
a property of a panel, so the intervention is a region replacement and no person needs to
move. That is why compositing is expected to be easier here and why the measurement of
success has to be about the panel, not about a worker.

**What the definitions explicitly do not support.** The paper's own Figure 9(a) is one video
holding frames of the unsafe class Safe Walkway Violation together with frames of the safe
class Closed Panel Cover. A closed panel therefore coexists with an unsafe scene in the
authors' own example, so **no output of this experiment may be described as a safe scene on
the strength of a closed cover.** The authors also do not say an open cover is unsafe in
every context; class 5, Authorized Intervention, exists for permitted work.

**Correction to earlier project records.** `README.md` and `data/raw/PROVENANCE.json` name
the first author "Onur Önal". The paper and the Mendeley contributor record both give
"Oğuzhan Önal". The existing records are wrong and should be corrected.

## P3. The data access blocker, and what was tried before asking

**Evidence.** The Mendeley public API serves a folder listing only when the folder id, or a
file id inside it, is already known. Eight distinct routes to enumerate the dataset root were
tried today and all failed; the full table is in `bridges2/DATASET_ACCESS.md`. This includes
the dataset metadata endpoint, which returns 200 with no file list and `size: 0`, the
rendered landing page, which contains the string `mp4` zero times, every UUID that appears
anywhere in the metadata or page source, each tried as both a folder id and a file id, and
the S3 zipfile cache, which returns 403.

**Choice.** Ask for two links, one `2_tr*.mp4` and one `6_tr*.mp4`, and continue every
independent deliverable in the meantime.

**Rationale.** The blocker is a browser-only step in a third-party web application. It is not
resolvable from this machine, and guessing random UUIDs is not an option. The same blocker
was hit and resolved the same way for class 4 on 2026-11-04, so the route is known to work.

**Uncertainty, stated before the data arrives.** Class 6 has 19 training clips. Whether three
independent class-2 source intervals exist with a clearly visible panel and an unobstructed
cover, and whether any class-6 clip shows the **same physical panel** in the closed state
from the same camera, cannot be known until the folders are inspected. Both are prerequisites
for the primary intervention. If the same panel never appears in both states, the honest
fallback is a same-camera, same-machine donor with the difference documented, and that is a
weaker claim which will be labelled as such rather than quietly accepted.

## P4. Data obtained and verified against the paper

**Evidence.** Both folder links arrived and resolved to `2_tr100.mp4` in folder
`22df4661-d1db-40b8-a429-9203bdffdbd2` and `6_tr10.mp4` in folder
`dca5cd74-fbb9-4226-aaac-3021d67d8a6f`. The class-2 folder listed **129 files, 1.37 GB**,
and the class-6 folder **19 files, 0.16 GB**. Those counts match Table 2 of the paper
exactly, which gives 129 and 19 training clips for these two classes. Every file was checked
against the sha256 the Mendeley API publishes; class 2 finished 129 of 129 verified with 0
failures.

**Choice.** Treat Table 2 as describing the published folders, and use the folder contents
as the population for selection and for the classifier's training and validation splits.

**Rationale.** Two independent sources now agree on the counts, and each file's hash matches
the publisher's. The earlier class-0 and class-4 folders matched Table 2 the same way, so
this is the third and fourth confirmation of the same correspondence.

**What was not downloaded.** The class-2 and class-6 **test** folders. The official test
split stays untouched, so no clip in it can reach a composite, a donor, a training split or
a validation split.

## P5. What the panel actually is, identified from the footage

**Evidence.** The audit measured where class-2 and class-6 medians differ, and the strongest
differences on this camera were workers, not fixtures: for `2_tr89` against `6_tr16` the
largest difference box held a worker in a hi-vis vest. Looking at the same two medians at
full resolution rather than at 480x270 settled it. Crop `x=150..650, y=700..1080` of both
shows the front of an eccentric press labelled "EKSANTRİK PRES". In `6_tr16` a
**yellow-framed wire-mesh guard** covers the press opening. In `2_tr89` the same yellow frame
is there and **the mesh is swung away**, with a worker reaching into the opening.

**Choice.** Take that guard as the panel the class-2/class-6 labels describe on this camera,
and freeze its box as `[265, 770, 180, 210]` in 1920x1080 coordinates in
`configs/panel_cover_regions.json`, with a static alignment patch at `[900, 150, 500, 400]`
on the far wall and coil stock.

**Rationale.** The paper defines the pair as a cover left open against a cover closed, and
describes Figure 2(d) as "a panel connected to a machine". A hinged mesh guard on a press is
exactly that. Nothing else in the frame changes state between the two classes.

**Uncertainty, and why it matters for selection.** The identification is from two clips.
Other class-2 clips may show a different panel, or this machine from a different pose, and
the frozen box would then point at the wrong pixels. That is why `scripts/pc_select.py`
measures an **alignment correlation** on a static patch for every clip and drops the ones
below 0.90 rather than scoring their region anyway.

**A consequence for source selection that was not obvious beforehand.** In `2_tr89` the
guard is open *because a worker is working through the opening*, and that worker occupies the
region. The task requires sources "with visible panels and no worker or equipment blocking
closure". So the obvious class-2 clips, the ones where the open guard is most visible, may be
exactly the ones disqualified by occlusion. `region_activity` measures this per clip, and a
source is only frozen after its region has also been looked at.

**Correction to an assumption in the audit.** The audit's camera-view clustering, which
correlates whole median backgrounds, put 146 of 148 clips in one group at threshold 0.95 and
split them into 15 groups at 0.98 that do not correspond to cameras: `2_tr89` and `2_tr1`
landed in different groups while showing the same press from the same pose. Whole-frame
correlation measures how much *moved*, not which camera filmed it. Camera identity for this
experiment is therefore decided by the alignment patch in `pc_select.py`, not by those
groups, and the audit's view assignment is not used for selection.

## P6. Correction: the press guard is not the panel the class-2 label refers to

**What P5 claimed.** That the yellow-framed mesh guard on the eccentric press is the object
the class-2 and class-6 labels describe, identified from `2_tr89` against `6_tr16`.

**The measurement that overturned it.** `scripts/pc_select.py` scored all 148 clips against
that region. Among the 120 class-2 clips aligned to this camera, **96 have a region
correlating above 0.80 with the closed reference**, which is the same range the class-6
donors occupy (0.83 to 0.94). Only three class-2 clips read as open at all: `2_tr89` (0.43),
`2_tr86` (0.45) and `2_tr85` (0.55). Region activity tells the same story: the class-2 and
class-6 activity distributions are nearly identical, with medians of 5.56 and 5.49.

**So the press guard is shut in the large majority of Opened Panel Cover clips.** It cannot
be the thing those clips are labelled for. P5 generalised from one pair of clips to a folder
of 129, and the generalisation is wrong.

**What P5 got right, and what it did not.** The guard genuinely is open in `2_tr89` and shut
in `6_tr16`; that observation was accurate. The error was treating one true observation as
the folder's rule without measuring it. The measurement cost 0.1817 SU and was worth it: had
three sources been frozen on P5's reading, every composite would have replaced a region that
was already closed in the source, and the experiment would have produced three
indistinguishable before-and-after pairs with no way to tell the failure from a subtle one.

**What replaces it.** The panel must be found from the **populations** rather than from a
pair, by averaging each class's median backgrounds over all aligned clips and differencing
those two averages. A per-clip difference is dominated by whichever worker happened to be in
that clip; an average over scores of clips suppresses the workers, which differ from clip to
clip, and keeps whatever differs consistently with the label. `scripts/pc_population.py`
does exactly that and nothing else.

**Uncertainty carried forward.** If the population difference also comes out dominated by
people rather than by a fixture, then the class-2/class-6 distinction on this camera is not a
localised object change that compositing can perform, and that is the finding to report
rather than a case to force.

## P7. The panel is an electrical cabinet door on the press column

**Evidence.** `scripts/pc_population.py` (job 50282294, 00:03:41, 0.0614 SU) built one
temporal median per clip at 960x540 for every clip aligned to this camera at ncc >= 0.95,
averaged those medians within each class (100 class-2, 12 class-6), and differenced the two
averages. Two regions survive the threshold of `max(8.0, p99.5)` = 46.15:

| region (full res x,y,w,h) | area px | mean difference | per-clip balanced accuracy |
|---|---|---|---|
| [218, 624, 110, 144] | 2081 | 59.2 of 255 | **0.935** |
| [150, 932, 52, 76] | 454 | 41.5 of 255 | 0.767 |

The per-clip statistic is `ncc(region, class-6 mean) - ncc(region, class-2 mean)`: class 2
scores -0.530 +- 0.486, class 6 scores +0.754 +- 0.007.

**What the first region is.** Looked at in `outputs/panel_cover/population/mean_class2.png`
against `mean_class6.png`, it is an electrical cabinet door low on the press column, below
the blue ram housing and left of the blue parameter plate. In the class-2 average the door
stands open on a dark cavity with internal gear visible and a red indicator lit. In the
class-6 average the same opening is a flat cream metal panel with its latch screws, with the
green-display controller mounted on its face. This is a panel cover, it is open across the
*Opened Panel Cover* population and closed across the *Closed Panel Cover* population, and it
is a fixture rather than a person. The second region is the scrap bin at the foot of the
press; it is not used.

**Why the class-6 spread is not circular.** Twelve clips with a standard deviation of 0.007
would be worthless if they were one recording counted twelve times. They are not: by the
audit's recording-group assignment those 12 clips fall into **11 distinct groups**
(`6_tr13` contributes two; the other ten groups contribute one each). The consistency is
across recordings.

**Known weaknesses, carried forward rather than smoothed over.**
1. The class-2 standard deviation of 0.486 is large. A balanced accuracy of 0.935 means
   roughly one aligned class-2 clip in eight does not read as open in this region, so some
   class-2 clips are labelled for something this door does not explain, or the door is
   occluded in them. Sources are chosen from the clearly-open end and checked by eye.
2. Each clip contributes to the mean it is scored against: 1% of the class-2 mean and 8% of
   the class-6 mean. The separation is therefore optimistic and is evidence for a hand-frozen
   plan, not a selection rule.
3. A region that differs with the label is not proof that the region holds the labelled
   object. The reading above is still a reading of two pictures; it is recorded so it can be
   disputed against the images, which are in the review package.
4. Only 12 class-6 clips align to this camera, and only 8 of them are long enough for 49
   frames at stride 2. Donor choice is close to unconstrained by the data.

**Decision.** `configs/panel_cover_regions.json` is rewritten: region
`views.press_front.panel_box_full_res` becomes `[208, 614, 130, 164]`, the population
bounding box padded by 10 px on each side so the door edge and latch fall inside the region
rather than on its boundary. The superseded file is kept at
`outputs/panel_cover/regions_superseded_P5.json` and the measurements that disproved it at
`outputs/panel_cover/selection_superseded_50282082.json`.

**Decision.** `scripts/pc_select.py` is rewritten to score each clip against the two class
means instead of against one closed and one open reference clip. Scoring against a single
reference clip is what produced P5, so the single reference is removed from the method; one
reference clip remains, for the alignment patch only, where a single static background is
exactly what is wanted.

## P8. The frozen region is a polygon, and two corrections were made before compositing

**Correction 1: the first box was too small.** `configs/panel_cover_regions.json` froze
`[208, 614, 130, 164]` from the population bounding box plus 10 px. Looking at the selection
contact sheets showed why that is not enough: the cabinet door is hinged on its left edge and
swings out towards the camera, so in an open clip the door leaf itself occupies pixels to the
left of the opening. Part of that leaf fell outside the box. Compositing with it would have
produced a closed panel with a sliver of the open door still standing beside it, which is the
kind of artefact a reader would reasonably mistake for a generation failure.

**Correction 2: the region is a connected component, not a rectangle.** The region now comes
from the absolute difference of the two class means, smoothed over 5 px, thresholded, and
flooded from a seed inside the cabinet opening; the convex hull of that component is expanded
2 px and mapped into preprocessed coordinates. 16 points, 7667 px, 2.22% of the frame.

**Why the threshold is 20 and not lower.** Three thresholds were computed and rendered over
both class means before one was chosen.

| threshold | component | what the rendering shows |
|---|---|---|
| 25 | 2604 px | clips the lower left of the door |
| **20** | **2955 px** | **follows the door in both states, excludes the blue parameter plate, stops above the mesh guard** |
| 15 | 4026 px | the door's component merges with the mesh-guard blob below it, and the convex hull then fills the notch between them |

The merge at 15 matters more than it looks. The mesh guard is the object of the disproved P5.
It does differ slightly between the two class populations, so a permissive threshold pulls it
back in through a connected path, and a convex hull turns that thin path into a filled area.
The experiment would then have been editing both the door and the guard while claiming to
edit a panel cover. 20 is the widest threshold that still separates them.

**Landmarks.** Six patches on fixed structure, each checked against the feathered mask for
overlap by exact mask arithmetic before freezing, not by eye. The first patch was moved 4 px
right because the polygon reaches x = 235 and the patch began at x = 234.

**Result of stage 3a**, job 50284349, 00:00:44, 0.0122 SU. All three cases built.

| case | landmarks | residual rms | donor coverage | px changed outside | region movement, source to composite |
|---|---|---|---|---|---|
| 1 | 6/6 | 0.0 px | 1.0000 | 0 | 4.175 to 0.156 |
| 2 | 6/6 | 0.0 px | 1.0000 | 0 | 3.709 to 0.130 |
| 3 | 6/6 | 0.0 px | 1.0000 | 0 | 4.937 to 0.146 |

Every landmark offset came back `[0, 0]`: source and donor are the same fixed camera, and the
translation fit confirms it rather than assuming it.

**Correction 3: acceptance criterion 5 was the wrong test, and it fails as written.** The
criterion asked that the seam step across the region boundary be "of the same order as a
control boundary in untouched scene". Measured, the composite seams are 22.2, 12.7 and 22.3
against controls of 1.1, 1.2 and 1.3. That is a factor of 10 to 20, so the criterion fails.
It was a badly chosen test: the region boundary runs along the edge of a cabinet door, which
is a high-contrast structural edge, while the control boundary sits in flat scene. The
*source* clips have seam steps of 17.8, 18.3 and 17.9 across the same boundary before
anything is composited. The meaningful comparison is composite against source at the same
boundary: +25%, -30% and +25%. The criterion is recorded as failed and the comparison it
should have specified is reported beside it; it is not quietly redefined into a pass.

## P9. The classifier reads the panel, and the composite still does not convince it

**Two measurements that have to be read together.**

*The region-masking diagnostic says the classifier depends on the panel.* Masking the panel
box, 1.03% of the classifier's 240x320 frame, against three equal-area controls:

| class | n | panel | mirror_x | mirror_y | centre |
|---|---|---|---|---|---|
| 2, Opened | 26 | **0.1311** | 0.0123 | 0.0132 | 0.0127 |
| 6, Closed | 3 | **0.1318** | 0.0174 | 0.0056 | 0.0188 |

About ten times every control, with the correct sign in each direction: masking an open panel
moves P(closed) up, masking a closed panel moves it down. Two class-2 clips reverse, `2_tr124`
and `2_tr5`, whose base P(closed) is already 0.455 and 0.714; both are among the 12 of 100
class-2 clips whose panel region reads closed, so the reversal is consistent.

*The evaluation says the composite barely moves the classifier.* P(closed) over the three
cases: source 0.133 to 0.149, composite 0.309 to 0.342. Real class-6 clips score 0.76 to 0.87.
Only one arm of fifteen crosses 0.5.

**Why that is not a contradiction, and what is still unknown.** The classifier does use the
region; it is not indifferent to it. One candidate explanation is that the donor patch is a
temporal median and therefore perfectly still, while a real closed panel in this dataset
carries sensor noise and lighting flicker, and `region_stability` measures exactly that
difference: 4.175 in the source against 0.156 in the composite, a region 27 times stiller
than its surroundings. **This is a hypothesis, not a measurement.** Testing it would need a
`per_frame` donor arm, which the compositing script supports and which was not run.

**The control arms move as much as the real change, so no flip may be attributed to the
panel.**

| arm | P(closed), 3 cases | panel actually closed? |
|---|---|---|
| source | 0.133, 0.142, 0.149 | no |
| `vae_source` | 0.223, 0.250, 0.207 | **no** |
| `base_edit` | 0.297, 0.412, 0.360 | **no** |
| composite | 0.321, 0.342, 0.309 | yes |
| `refined` | 0.470, **0.543**, 0.453 | yes |
| `refined_restricted` | 0.337, 0.353, 0.327 | yes |

A VAE round trip that changes no panel moves P(closed) by +0.07 to +0.11. A prompt-only edit
that leaves the panel open moves it by +0.16 to +0.27, which is more than the composite's
+0.17 to +0.20 while genuinely closing the panel. And `refined` falls from 0.470 to 0.337
when its pixels outside the region are restored, as `refined_restricted`: most of that arm's
rise comes from outside the panel. The single crossing of 0.5, case 2's `refined` at 0.543,
sits inside that control band and is not evidence of a panel effect.

**The prompt alone never closed the panel.** `base_edit` leaves the region correlating 0.853
to 0.863 with the open reference and 0.063 to 0.091 with the closed one, while dropping
whole-frame fidelity to about 20.4 dB. It changed the picture everywhere except the thing it
was asked to change. This is the prompt-guided baseline behaving as a baseline, and it is
reported as a negative result rather than tuned until it passes.

## P10. Correction: the region-masking diagnostic silently measured nothing

`pc_diagnostics.py` looked the panel box up by the audit's camera-view label, under
`regions["views"][view]["panel_box"]`. Two things were wrong at once: the region config keys
its one view `press_front` rather than by a clip label, and it stores
`panel_box_full_res` rather than `panel_box`. Every one of the 37 validation clips was
therefore skipped with `no frozen panel box for this view`, and job 50284823 spent 0.0603 SU
producing a file whose main section contained nothing but skip records. The VAE control in
the same job was unaffected and valid.

It would have been easy to miss: the job exited 0, wrote its output, and printed its section
heading. What caught it was reading the output file rather than the exit status.

The fix drops the view lookup entirely. The box is frozen once in source coordinates, the
classifier sees the whole frame scaled with no crop, so the box scales by the same factors;
a clip is included on its measured `alignment_ncc` instead, which is the test that actually
governs whether the box means anything for that clip. Keying on the audit's view assignment
was doubly wrong, because DECISIONS P5 had already rejected that clustering as not being
camera identity. The superseded output is kept at
`outputs/panel_cover/diagnostics_superseded_50284823.json`.

## P11. Correction: the gradient probe's NaN was the probe's step, not the gradient

Job 50286103 reported `log_p_target_after_one_step: NaN` while its gate block reported a
finite, non-zero gradient that fits in memory. Read together those two lines look like a
gradient that exists but destroys the decode, which would have been a reason not to spend
1.0 SU on guidance.

The cause was the probe's own default. It took one step of
`step_size * grad / ||grad|| * ||lat||` with `step_size = 50.0`, which moves the latent fifty
times its own norm. The fp16 VAE decode overflows at that magnitude. The number said nothing
about the gradient.

Reading only the `gate` block would have passed this through as a clean result, and reading
only the NaN would have abandoned a working stage. Both lines had to be read against each
other.

**The fix replaces the single step with a sweep**, because one step at one arbitrary
magnitude cannot show either whether the objective moves or where the decode breaks. Job
50286262, case 1, starting from `log p(target) = -0.798`:

| step, as a fraction of the latent norm | log p(target) | P(closed) |
|---|---|---|
| 0.01 | -0.618 | 0.539 |
| **0.02** | **-0.446** | **0.640** |
| 0.05 | -0.622 | 0.537 |
| 0.1 | -1.198 | 0.302 |
| 0.2 | -1.612 | 0.199 |
| 0.5 | -1.614 | 0.199 |

Every row is finite: there is no NaN anywhere in the usable range, and the earlier one was
entirely an artefact of the 50x step. The objective rises to an optimum near 0.02 and falls
away after it, which is the shape a useful gradient with a limited trust region has.

`CLASSIFIER_SCALE` in `slurm/pc_guided.sbatch` was already 0.02 before this sweep ran, so
the frozen setting is not being tuned to this result; the sweep happens to confirm it.

**All three prerequisites for stage 5 now hold**: the classifier reaches usable validation
performance on a group-separated split, region masking shows a panel effect about ten times
its equal-area controls with the correct sign in both classes, and the gradient is finite,
non-zero and fits in memory. A rise in `log p(target)` along the gradient is still only a
statement about the classifier's objective. It is not evidence that a panel closed, and the
guided output is judged on the same panel-state proxy and the same controls as every other
arm.

## P12. Classifier guidance flips the label without closing the panel

Job 50286287, case 1. Guided and unguided runs share seed 20261105, strength 0.6, 30
executed steps, guidance scale 6.0, the same prompt and the same initial noise. The
classifier term is the only difference: scale 0.02 against 0.0.

| | input | unguided control | guided |
|---|---|---|---|
| P(closed) | 0.1325 | 0.2835 | **0.9999** |
| prediction | opened | opened | **closed** |
| ncc of the panel region against the open reference | 0.988 | 0.858 | **0.821** |
| ncc against the closed reference | 0.113 | 0.061 | **0.109** |
| PSNR vs input | | 20.29 dB | 20.06 dB |
| generation | | 47.3 s | 659.5 s |

**The panel is still open in the guided output.** Its region correlates 0.821 with the open
reference and 0.109 with the closed one, which is where the source itself sits. The
classifier is 99.99% confident of the opposite.

**The attribution rule is satisfied and it does not rescue the result.** The unguided control
did not flip, so the flip belongs to the classifier term rather than to the denoising loop.
That makes it a sound statement about the classifier and no statement at all about the panel.

**Set against the composite, the classifier's confidence runs backwards.**

| arm | panel, ncc vs closed | P(closed) |
|---|---|---|
| composite | **0.9997**, genuinely closed | 0.3205 |
| guided | **0.1093**, still open | **0.9999** |

An arm that actually closes the panel is given 0.32. An arm that leaves it open is given
0.9999. This is an adversarial counterfactual: the optimisation found a cheaper route to the
label than the physical change the label is supposed to track. The whole-frame fidelity of
20.06 dB puts it in the same band as `base_edit`, the arm that demonstrably never closed the
panel, rather than near the composite's 35 dB.

**This is the result the control structure existed to catch**, and it is reported as the
main finding rather than as a footnote to a successful flip. A changed prediction is not
evidence of a changed scene, and here the two point in opposite directions in the same
experiment, measured on the same proxy, within one case.

**Replication.** One case can be luck, so the pair was repeated on cases 2 and 3, which share
no recording group with case 1 or with each other.

## P13. Two gaps that would have left the guided arm unmeasured

`pc_evaluate.py` looked for arms only in `<case>/videos/`, and its `ARM_FILES` tuple listed
neither `guided` nor `unguided_control`. `pc_build_review.py` had the matching problem: it
copied the guidance clips under `guided_<name>.mp4` while its page looked for
`<case>_<arm>.mp4`, so they would have been staged and never shown.

Left alone, the guided output's only number would have been the classifier's own
P(closed) = 0.9999, reported by the script that produced it. The panel-state proxy that
contradicts it would never have run, and the reviewer would have been handed a flip with no
way to see that the panel never moved. Both are fixed: the guided arm is now scored on
exactly the same proxy, preservation, seam and stability tests as every other arm, and both
guidance arms appear on the review page with the control named beside them.
