# Candidate counterfactual review

Copy this file to `outputs/stage4_edit/review_<run_id>.md` and fill it in while watching the
clips, not from the manifest alone. Nothing in this run is a safety verdict until this form
is complete and signed.

## Run identification

| Field | Value |
|---|---|
| Run ID | |
| Date reviewed | |
| Reviewer | |
| Source clip | `0_tr143.mp4` (sha256 `7a7c0d10…`) |
| Unsafe label | 0 Safe Walkway Violation |
| Target safe label | 4 Safe Walkway |
| Checkpoint + revision | |
| Seed / strength / steps / guidance | |
| Manifest path | |

## Stage 3 — VAE reconstruction

Judged from `source_vs_reconstruction.png` and the two clips played back to back.

| Detail | Survived? (yes / degraded / lost) | Note |
|---|---|---|
| Worker silhouette | | |
| Worker's feet and their contact with the floor | | |
| Green walkway edge | | |
| Yellow boundary lines | | |
| Machine signage and text | | |
| Overall temporal stability (no flicker) | | |

- PSNR (dB): &nbsp;
- Mean absolute difference: &nbsp;
- Peak GPU memory (GiB): &nbsp;

**Verdict:** the VAE does / does not preserve the detail this label depends on.

If detail that decides the label is already lost here, say so. Stage 4 cannot recover it, and
any Stage 4 result must be read in that light.

## Criterion 1 — Did the safety-critical behaviour change?

The intended change: the worker moves from the bare concrete aisle onto the green marked
pedestrian walkway.

- [ ] The worker is now on the green walkway for the whole clip
- [ ] The worker is on the walkway for part of the clip only (describe which frames)
- [ ] The worker did not move; only appearance or style changed
- [ ] The worker moved somewhere else entirely
- [ ] Unreadable (too degraded to judge)

Evidence (frame numbers, what is visible):

### Spurious-change check — reject if any of these is true

- [ ] The green walkway was repainted, widened or extended underneath the worker
- [ ] A second green region appeared that is not in the source
- [ ] The yellow boundary lines moved to re-classify the worker's existing position
- [ ] The worker was removed rather than relocated

Any box ticked here means the label may flip while the *behaviour* is unchanged. That is a
spurious counterfactual, not a valid one, regardless of how good the clip looks.

## Criterion 2 — Was context preserved?

| Element | Preserved? | Note |
|---|---|---|
| Camera pose and framing (fixed overhead) | | |
| Worker identity, build, clothing type | | |
| Worker's task / what he is carrying | | |
| Press machines and left-hand machine zone | | |
| Other people in frame | | |
| Floor markings not involved in the edit | | |
| Lighting and time of day | | |

## Criterion 3 — Temporal consistency, physical plausibility, artifacts

| Aspect | Rating (ok / minor / severe) | Note |
|---|---|---|
| Frame-to-frame coherence | | |
| Gait and limb plausibility | | |
| Feet in contact with the floor (no sliding or floating) | | |
| Object permanence across frames | | |
| Flicker, warping, ghosting | | |
| Background stability | | |

## Criterion 4 — Target classifier prediction

**Not applicable for this run.** The Unsafe-Net authors released no code and no weights, and
the Mendeley record contains videos only. No classifier exists whose label space holds
*Safe Walkway* as the paper defines it.

If a classifier is added later:

| Field | Value |
|---|---|
| Classifier identity and provenance | |
| Label space (full list) | |
| Prediction on source | |
| Prediction on edited clip | |
| Prediction on VAE reconstruction (control) | |

A changed prediction does not prove the scene is safe. The absence of *Safe Walkway
Violation* is not the presence of *Safe Walkway*. Record a prediction flip as one signal
among four, never as the verdict.

## Overall

- [ ] Valid candidate counterfactual — behaviour changed, context preserved, artifacts acceptable
- [ ] Partial — behaviour changed but context or plausibility degraded (explain)
- [ ] Spurious — label-relevant pixels changed without the behaviour changing
- [ ] Failed — no meaningful edit
- [ ] Failed — run did not complete (record the blocker in the status report)

Summary, in plain words:

Actionable cause, if this failed:

Next single change to try (change one thing at a time):

Reviewer signature / date:
