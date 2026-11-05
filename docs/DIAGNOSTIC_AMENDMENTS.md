# Amendments to the shortcut diagnostics, made after the fact

**These are not preregistered criteria.** `docs/ACCEPTANCE_CRITERIA.md` was written before
any training and its classifier acceptance criteria — B1, B2, B3 and the Stage C reasoning
— are unchanged and must stay unchanged. Everything in this file was decided *after* seeing
results, because the diagnostic that Stage C depends on was found to be broken. Changes made
after seeing data are weaker evidence than changes made before, and they are recorded
separately so that nobody can mistake one for the other later.

Each amendment below states what was observed, what changed, and what the change cannot
claim.

---

## Amendment 1 — the 30% motion-energy floor

**Date:** 2026-11-04, after job 50218860.

**Observed.** The worker mask changed validation balanced accuracy by −0.0107, exactly as
much as the control mask, with 2 of 79 predictions flipped. That reading is compatible with
two opposite conclusions: the classifier ignores the worker, or the mask never covered the
worker.

**Changed.** Each mask now reports the fraction of the clip's motion saliency it covers, and
Gate C fails as *inconclusive* when the worker mask covers less than 30% of it.

**What it cannot claim.** The 30% figure is a judgement, not a calibrated threshold. No
measurement establishes that 30% is the point where the diagnostic becomes trustworthy. It
is set to catch the failure that was observed — a mask covering almost nothing — and nothing
more. Motion-energy coverage is **supporting evidence about where a mask sits, not proof
that it covers the worker.** The visual check is what settles that.

---

## Amendment 2 — per-frame mask boxes

**Date:** 2026-11-04, after job 50218860.

**Observed.** One box per clip, placed at the motion centroid of the whole clip, cannot
follow a worker who walks across the frame. The centroid sits between his start and end
positions.

**Changed.** A box is now computed for every frame.

**What it cannot claim.** Per-frame placement fixes the tracking problem only. It does not
make the placement rule correct, as Amendment 3 shows.

---

## Amendment 3 — box placement by sliding-window maximum

**Date:** 2026-11-04, after rendering `outputs/classifier/mask_overlays/`.

**Observed.** The overlay images showed all three masks converging on the same central
region, a workpiece glow in the machine bank. None covered the worker. None covered the
walkway. The marked walkway in these clips is an L shape whose centroid lies off the
walkway entirely, so a centroid-placed box sat in open floor.

**Changed.** A box is placed at the sliding-window maximum of the score map, computed with
an integral image, so the box lies inside the mass it is masking by construction.

**Result.** The lane mask now sits on the green walkway in every sampled frame, confirmed
visually. The control mask sits on ceiling and open floor.

---

## Amendment 4 — flicker suppression in the motion map

**Date:** 2026-11-04, same inspection.

**Observed.** Raw deviation from the clip median was dominated by a workpiece glow pulsing
in every frame.

**Changed.** Motion saliency is divided by each pixel's temporal standard deviation, which
suppresses anything varying constantly in place and keeps what passes through a pixel once.

**What it cannot claim.** This did not fix worker localisation. See the open problem below.

---

## Open problem — motion does not isolate the worker

**Date:** 2026-11-04.

After Amendments 2, 3 and 4, the overlay images still show the worker mask on the machine
bank rather than on the worker. The reason is a false premise written into the original
code and into my own description of it:

> "In a fixed-camera factory clip the only thing that moves is the worker."

That is wrong. These are clips of a working production line. Presses actuate, arms swing,
workpieces glow and move. Motion is not specific to people, so no motion-based rule can
locate the worker here, however the box is placed.

**Status (superseded by Amendment 5): Stage C was blocked.** The worker-occlusion diagnostic is not valid, so Gate C
cannot be judged, so Stage D and Stage E do not run. This is a blocker, not a result.

The candidate fix under evaluation is a pretrained COCO person detector to locate the worker
directly, instead of inferring the worker's position from motion. If that is adopted it
becomes Amendment 5 and carries the same caveat as everything in this file: it is a change
made after seeing the data.

The lane and control masks are unaffected by this problem and are now visually confirmed.


---

## Amendment 5 — worker located by a COCO person detector

**Date:** 2026-11-04, authorised by the user after the blocker was reported.

**Observed.** Motion-based localisation could not find the worker, because presses, arms
and workpieces move in these clips. The overlay images showed every motion-placed mask on
machinery.

**Changed.** The worker mask is placed from the highest-scoring `person` detection of
`fasterrcnn_mobilenet_v3_large_fpn` with COCO weights, at the same area as the lane and
control masks. A frame with no detection carries the previous box forward and is counted
as a miss. The detector is used only to place this mask; it takes no part in training or
in guidance.

**Validated visually.** `outputs/classifier/mask_overlays/` shows the red worker box on the
walking worker across sampled times in both classes. Missed detections: 0 of 16 frames in
five of six inspected clips, 1 of 16 in `4_tr10`. Aggregate miss rate over the validation
split: see `person_detection_miss_rate` in `eval_report.json`.

**Replaced.** The 30% motion-energy floor of Amendment 1 no longer decides validity. Once
the mask sits on a detected person, coverage of motion energy is beside the point, so
validity is judged on the detection miss rate instead, with the same 30% figure and the
same caveat that it is a judgement rather than a calibrated threshold.

---

## Amendment 6 — gate B3 evaluated on the full validation split

**Date:** 2026-11-04, authorised by the user after the 16-clip result failed.

**Observed.** Agreement of 0.8750 on 16 clips, below the preregistered 0.90. Fourteen of
sixteen carries a wide interval.

**Changed.** The sample was expanded to the whole fixed validation split, evaluated once.
**The 0.90 threshold was not changed.** Reconstructions are cached, so the original clips
are reused rather than recomputed.

**Both results are preserved and reported:** 14/16 = 0.8750 from the first run, and the
full-split figure from the second. Neither replaces the other. The expansion was decided
after seeing a failure, which is the weakest moment at which to decide such a thing, and
that is exactly why both numbers stay on the record.

**Also added.** Per-clip probabilities, predictions and true labels, plus accuracy and
balanced accuracy against the true labels before and after reconstruction. Agreement alone
cannot show whether a flip moved a clip from right to wrong or the other way.
