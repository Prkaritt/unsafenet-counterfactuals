# How recording identity is decided, and why the first attempt was wrong

Written 2026-11-04 from measurements on the 228 downloaded training clips.

## The first attempt failed, measurably

Recording groups were first found with a 64-bit average hash of sampled frames: two clips
sharing a frame within 4 bits were taken to share a recording. On this dataset that test
is meaningless.

Measured minimum Hamming distance between clip pairs:

| Pair type | 0th | 1st | 5th | 25th | 50th percentile |
|---|---|---|---|---|---|
| `0_tr143` against `0_tr140`–`0_tr145` | 1 | 3 | 1 | 1 | 2 |
| **class 0 against class 4** | **0** | **0** | **0** | 2 | 19 |

A class-0 clip and a class-4 clip carry different behaviour labels, so they cannot be
segments of one recording. Yet their minimum distance reached **0**, and **36%** of sampled
cross-class pairs fell within the 4-bit threshold.

The cause is the subject matter. These are fixed-camera clips of one factory. An 8x8
average hash encodes the room, which every clip from that camera shares, and the worker
occupies a handful of pixels at that resolution. The hash answers "same room?" when the
question was "same moment?".

The consequence was not a subtle bias. Union-find over 9,964 spurious pairs merged nearly
every clip into one group, that group went to validation, the training split came out
empty, and the job failed one stage later at the training probe.

## What replaced it

Recording identity is now decided on the **motion residual**: each sampled frame minus the
clip's own median background, downsampled to 36x64, mean-centred and unit-normalised.
Subtracting the median cancels the room, which is the part every clip from that camera
shares, and keeps the part that differs: a specific person, in a specific pose, in a
specific place.

Two frames from one continuous recording have near-identical residuals. Two frames from the
same camera an hour apart do not.

Similarity between two clips is the maximum cosine similarity over all pairs of their
sampled frames.

## Calibration

The class-0 against class-4 pairs are a clean negative set: different behaviour labels
means different recordings, with no judgement call involved.

| Statistic over 400 sampled class-0 / class-4 pairs | Residual similarity |
|---|---|
| 50th percentile | 0.1349 |
| 90th | 0.5239 |
| 99th | 0.7969 |
| 99.9th | 0.8728 |
| **maximum** | **0.8875** |
| fraction at or above 0.90 | **0.0000** |

The threshold is set at **0.90**, above the entire measured negative distribution with
margin. Under it, 71 pairs among 228 clips are judged to share a recording, against 9,964
under the old test.

A threshold chosen this way is a data-integrity decision calibrated on a negative set whose
labels make the answer known in advance. It is not a model hyperparameter and it was not
tuned against any held-out performance number.

## What this changed about the source clip

Under the average hash, `0_tr143` appeared to share a recording with `0_tr140`, `0_tr141`,
`0_tr142`, `0_tr144` and `0_tr145`, at Hamming distances of 1 to 3. **That finding was an
artefact of the broken metric and is withdrawn.**

Under the residual metric, `0_tr143`'s nearest neighbours are:

| Clip | Residual similarity |
|---|---|
| `0_tr125` | 0.8629 |
| `4_tr44` | 0.7790 |
| `0_tr145` | 0.7341 |
| `0_tr138` | 0.7260 |

Every one sits below the maximum of the negative distribution, 0.8875. **No clip in the
dataset shares a recording with `0_tr143`.** Only `0_tr143` itself is excluded from the
splits.

## Scene sharing is reported, not excluded

46 clips share the source clip's camera position without sharing its recording. An earlier
version excluded those too, as a precaution. That cost **26 of the 50 safe-walkway clips** —
over half the positive class, and specifically the clips showing the same aisle traversed
correctly, which are the most informative examples of the behaviour the classifier must
learn.

Exclusion is therefore decided on recording overlap alone, which is what actually leaks
frames between splits. The 46 scene-sharing clips stay in training and are listed under
`share_source_scene_but_not_recording` in `outputs/classifier/groups.json`.

The residual risk is stated rather than hidden: the classifier will have seen this camera
position, with both behaviours, during training. It has not seen `0_tr143` or any frame
from its recording.

## Resulting splits

| Split | class 0 | class 4 |
|---|---|---|
| train | 117 | 30 |
| val | 60 | 20 |
| excluded | 1 (`0_tr143`) | 0 |

Official test clips are not downloaded; test evaluation is deferred by user decision.

Validation groups are taken whole until a target share of each label's **clips** is
reached. Taking a fraction of the *groups* instead gave validation only 4 clips of one
label, too few for a per-class recall gate to mean anything.

## Checks now enforced by Gate A

- Zero exact duplicate sha256 across real splits.
- Zero recording-overlap pairs across real splits. A pair with one foot in `excluded` is
  not a leak; it is the audit working, and it no longer counts as a failure.
- Neither train nor val empty, and both labels present in each.
- No undecodable clips.
