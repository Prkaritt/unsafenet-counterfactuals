# Stage A — data integrity audit

Clips inventoried: 228

| Check | Result | Gate |
|---|---|---|
| A1 exact duplicate sha256 groups | 0 | — |
| A1 crossing a split boundary | 0 | must be 0 |
| A2 recording-overlap pairs found | 61 | — |
| A2 **detected** overlaps crossing a split boundary | 0 | must be 0 |
| A3 clips excluded in total | 9 | — |
| of which quarantined as label-ambiguous | 8 | — |
| A3 any excluded clip leaked into train or val | False | must be False |
| Clips that could not be decoded | 0 | must be 0 |
| Binary labels present | [0, 1] | must be [0, 1] |

**Gate A: PASS**

## Recording groups versus scene groups

Pairs sharing frames outright, minimum mean absolute pixel difference at or below 1.0: 61. These are certain evidence of one recording.
Pairs probably from one recording, difference above 1.0 and at or below 2.0: 302. Reported as uncertain and kept together when splitting.
Positive control: overlapping halves of one clip were DETECTED.
Scene-similarity pairs, meaning a shared camera position and nothing more: 5101.
Pairs where recording identity is **uncertain** — same scene and adjacent filename index, but no shared frame: 302.

A similar factory background does not establish a shared recording. Exclusion from the splits is therefore decided on recording overlap alone, which is what leaks. Clips that merely share the source clip's camera position are kept in training and listed under share_source_scene_but_not_recording in groups.json, 46 of them. Excluding those as well was tried and removed half of the safe-walkway class, including the examples most useful for learning behaviour rather than room identity.
