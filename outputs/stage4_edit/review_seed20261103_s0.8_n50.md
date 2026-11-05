# Candidate counterfactual review

**Status: DRAFT, prepared by Claude from the contact sheet and the manifest. Not signed.**
The reviewer must watch `edit_seed20261103_s0.8_n50.mp4` and
`outputs/stage3_vae/reconstruction.mp4` end to end, correct every line below, and sign.
Entries marked *(needs playback)* cannot be judged from still frames at all.

Nothing in this run is a safety verdict until this form is complete and signed.

## Run identification

| Field | Value |
|---|---|
| Run ID | `seed20261103_s0.8_n50`, Slurm job 50198591 |
| Date reviewed | *(unsigned draft, prepared 2026-11-04)* |
| Reviewer | *(unsigned)* |
| Source clip | `0_tr143.mp4` (sha256 `7a7c0d10…`) |
| Unsafe label | 0 Safe Walkway Violation |
| Target safe label | 4 Safe Walkway |
| Checkpoint + revision | `THUDM/CogVideoX-2b`, `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01` |
| Seed / strength / steps / guidance | 20261103 / 0.8 / 50 requested, 40 executed / 6.0 |
| Manifest path | `outputs/stage4_edit/manifest_seed20261103_s0.8_n50.json` |

The step count needs explaining, because it has been read as a configuration error before.
The pipeline truncates its timestep schedule by `strength`, so a request for 50 steps at
strength 0.8 runs 40. Both numbers are now recorded in the manifest as
`num_inference_steps` and `denoising_steps_executed`.

## Stage 3 — VAE reconstruction

Judged from `outputs/stage3_vae/source_vs_reconstruction.png`. Playback comparison is
outstanding.

| Detail | Survived? (yes / degraded / lost) | Note |
|---|---|---|
| Worker silhouette | yes | Outline and limb separation readable at all four sampled frames |
| Worker's feet and their contact with the floor | yes | Contact point stays on the concrete, no visible float |
| Green walkway edge | yes | Boundary stays sharp; this is the pixel evidence the label depends on |
| Yellow boundary lines | yes | Unbroken, no colour bleed into the concrete |
| Machine signage and text | degraded | Lettering softens to unreadable; does not bear on this label |
| Overall temporal stability (no flicker) | *(needs playback)* | Per-frame PSNR varies only 31.43 to 33.14 dB, which suggests stability but does not prove the absence of flicker |

- PSNR (dB): 32.60
- Mean absolute difference: 3.52 / 255
- Peak GPU memory (GiB): 21.611

**Verdict:** the VAE **does** preserve the detail this label depends on. The walkway edge,
the boundary lines and the worker's foot contact all survive the round trip. Any Stage 4
failure is therefore a failure of the edit, not of the autoencoder.

This number is also the ceiling. No edited clip can score higher than 32.60 dB against the
source, because every edited frame is decoded through this same VAE.

## Criterion 1 — Did the safety-critical behaviour change?

The intended change: the worker moves from the bare concrete aisle onto the green marked
pedestrian walkway.

- [ ] The worker is now on the green walkway for the whole clip
- [ ] The worker is on the walkway for part of the clip only (describe which frames)
- [ ] The worker did not move; only appearance or style changed
- [x] The worker moved somewhere else entirely
- [ ] Unreadable (too degraded to judge)

Evidence (frame numbers, what is visible):

Sampled frames 0, 16, 32 and 48. In every one the worker walks down the centre of a bare
concrete aisle, not on the green strip. The green strip in the edited clip runs along the
right of frame and the worker never enters it. The worker also faces the camera and walks
toward it, where in the source the camera looks down on him from behind and above. So the
worker's position relative to the walkway is unchanged in the only sense that matters: he
is still off it.

### Spurious-change check — reject if any of these is true

- [ ] The green walkway was repainted, widened or extended underneath the worker
- [x] A second green region appeared that is not in the source
- [ ] The yellow boundary lines moved to re-classify the worker's existing position
- [ ] The worker was removed rather than relocated

The green walkway in the edited clip is not the source's walkway moved or repainted. It is
a different walkway in a different room, because the whole scene was regenerated. That is
worse than a spurious counterfactual, not better: there is no correspondence to the source
left to evaluate.

## Criterion 2 — Was context preserved?

| Element | Preserved? | Note |
|---|---|---|
| Camera pose and framing (fixed overhead) | no | The viewpoint drops from a high overhead security angle to a near-eye-level view down the aisle |
| Worker identity, build, clothing type | partly | Still one worker in dark navy overalls; face, build and gait are not the same person's |
| Worker's task / what he is carrying | *(needs playback)* | Nothing carried is visible in the sampled frames in either clip |
| Press machines and left-hand machine zone | no | The source's blue press bank is replaced by unrelated yellow racking and a different machine in the near foreground |
| Other people in frame | *(needs playback)* | Second figure appears beside the worker at the sampled frame 16 of the edited clip; check whether anyone else enters across the full 49 frames |
| Floor markings not involved in the edit | no | The source's floor arrows and lane markings are replaced with different ones |
| Lighting and time of day | no | Source is dim indoor artificial light; the edit is bright, near-daylight, with different shadow direction |

## Criterion 3 — Temporal consistency, physical plausibility, artifacts

| Aspect | Rating (ok / minor / severe) | Note |
|---|---|---|
| Frame-to-frame coherence | *(needs playback)* | Sampled frames are individually coherent; drift between them is unmeasured |
| Gait and limb plausibility | ok (stills only) | Limbs articulate plausibly in each sampled frame |
| Feet in contact with the floor (no sliding or floating) | *(needs playback)* | Cannot be judged from four stills |
| Object permanence across frames | minor | The near-foreground machine changes shape between sampled frames 0 and 48 |
| Flicker, warping, ghosting | *(needs playback)* | |
| Background stability | severe | The background differs from the source throughout; stability *within* the edited clip still needs playback |

The artifacts are not the problem with this run. The clip looks like a plausible factory. It
is a plausible factory that is not the one in the source.

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
- [x] Failed — no meaningful edit
- [ ] Failed — run did not complete (record the blocker in the status report)

Summary, in plain words:

The job completed cleanly and the model generated a good-looking factory video. It is not an
edit of the source clip. The camera moved, the machines changed, the lighting changed and
the walkway is a different walkway. The one thing that was supposed to change, the worker's
path relative to the marked walkway, did not. PSNR of 15.89 dB against a 32.60 dB VAE
ceiling says the same thing numerically: about 17 dB of the loss is the edit discarding the
source, not the autoencoder.

This counts as a completed feasibility run with a failed outcome, which `CLAUDE.md` treats
as an honest result and not as a successful generation. No safety claim is made.

Actionable cause:

`strength=0.8` keeps only 20 percent of the source latent trajectory, which is not enough to
hold the scene. Combined with `guidance_scale=6.0` and a prompt that re-describes the whole
room — machines, floor, lighting — the model regenerates the room from the text instead of
editing the video. The prompt invites exactly the failure that was observed.

Next single change to try (change one thing at a time):

Lower `strength`. Slurm job 50213456 sweeps 0.5, 0.6 and 0.7 at guidance 6.0, plus 0.6 at
guidance 3.5 to separate the effect of guidance from the effect of strength. Read the result
as a trade-off curve: expect low strength to preserve the scene and change nothing, and high
strength to change everything. The open question is whether any point does both.

Reviewer signature / date: *(unsigned draft)*
