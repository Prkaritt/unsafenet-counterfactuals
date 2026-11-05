# Status report

Date: 2026-11-04 (last updated). Earlier entries dated 2026-11-03 are kept as written.

Four categories, kept strictly separate: implemented, locally checked, actually GPU-executed,
and blocked.

---

## 1. Implemented

| Item | Path |
|---|---|
| Checkpoint-constraint constants, crop maths, segment validation, ffmpeg command builder, video probing | `src/unsafenet_cf/preprocess.py` |
| Environment capture, GPU/VRAM snapshot, peak-memory helpers, run-manifest schema | `src/unsafenet_cf/manifest.py` |
| PSNR, per-frame PSNR, mean absolute difference, comparison contact sheets | `src/unsafenet_cf/compare.py` |
| Colab notebook, 22 cells: hardware probe → pins → clip fetch → preprocess → VAE check → edit → manifest → Stage 5 gate | `notebooks/unsafenet_cogvideox_feasibility.ipynb` |
| Clip spec and generation settings, including what must not change and the spurious-change definition | `configs/clip_0_tr143.json`, `configs/experiment.json` |
| Dataset provenance: DOI, licence, 8-class table, per-file SHA-256 | `data/raw/PROVENANCE.json` |
| Human review form covering all four success criteria | `templates/REVIEW_TEMPLATE.md` |
| Local and Colab run instructions, limitations | `README.md` |
| Gitignore keeping clips, outputs and weights out of version control | `.gitignore` |

---

## 2. Locally checked (no GPU)

| Check | Result |
|---|---|
| Unsafe-Net paper read end to end (27 pages, text extracted) | 8-class label space and dataset description confirmed from Table 1 and §4 |
| Dataset identity verified against the paper | Same authors, same institution, 691 clips = 566 train + 125 test, CC BY 4.0 |
| Three class-0 clips downloaded from Mendeley | All three SHA-256 match the published hashes |
| Clip metadata probed | All three: 1920×1080, 25 fps, 262 frames, 10.44 s, h264, `yuvj420p` |
| Frames inspected visually | `0_tr143.mp4` selected; worker large, unoccluded, walkway visible |
| CogVideoX-2b constraints read from the live checkpoint configs | VAE and transformer configs fetched and recorded |
| `CogVideoXVideoToVideoPipeline` signature read from current diffusers docs | Confirmed: no `num_frames` argument; frame count is `len(video)` |
| Test suite | **67 tests pass** (49, plus 18 covering the generation-settings maths added 2026-11-04) |
| Preprocessing executed end to end on the real clip | Produced 720×480, 49 frames, 8 fps, 6.12 s |
| Crop verified by rendering preprocessed frames | Worker and full green walkway both present at `x=300` |
| Notebook validated | Valid nbformat 4 JSON; all 13 code cells parse as Python |

One real bug was found and fixed by the tests: the ffmpeg report parser captured
`progressive` as the pixel format instead of `yuvj420p`, because a regex split on commas
inside parentheses. Replaced with a parenthesis-aware field splitter.

### What local validation does **not** cover

Syntax validation is not GPU validation. Nothing below has been exercised: model loading,
dtype behaviour on an actual GPU, VAE encode/decode, tiling, CPU offloading, the denoising
loop, memory headroom, or runtime.

---

## 3. Actually GPU-executed

### 3a. Bridges-2, L40S, 2026-11-04 — the current record

Two jobs on the GPU-shared partition:

| Job | What ran | Elapsed |
|---|---|---|
| 50196825 `cfprobe` | hardware probe, no weights loaded | about 2 min |
| 50198591 `cfbase` | preprocess, VAE reconstruction, one prompt-guided edit | 9 min 11 s |

Assigned hardware: NVIDIA L40S, sm_89, 44.42 GiB VRAM, 1006 GiB system RAM. The Colab
runtime's 12 GiB of system RAM was the binding constraint there; it is gone here. Stack:
Python 3.12.13, torch 2.9.1+cu128, diffusers 0.36.0, transformers 4.57.6. Checkpoint
staged locally at `checkpoints/CogVideoX-2b`, revision
`1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01`, loaded with `HF_HUB_OFFLINE=1`.

| Stage | Measurement |
|---|---|
| VAE load | 24.9 s |
| VAE round trip (49x480x720) | 42.8 s, peak 21.611 GiB, latent `(1, 16, 13, 60, 90)` |
| VAE fidelity | PSNR **32.60 dB**, mean abs diff 3.52/255 |
| Pipeline load, whole checkpoint resident, no offload | 198.4 s |
| Generation, 50 requested steps at strength 0.8, so 40 executed | **93.0 s**, peak **36.205 GiB** |
| Edit vs source | PSNR **15.89 dB**, mean abs diff 29.47/255 |

The VAE numbers reproduce the T4 run closely: 32.60 dB against 32.56 dB, 3.52 against 3.56.
That is the expected agreement for the same deterministic `latent_dist.mode()` encode on
different fp16 kernels.

The 36.205 GiB generation peak is worth recording on its own. The T4's 15 GB card could
never have held this configuration without offloading, so the two runs' generation timings
are not like for like even where the settings match.

**The edit failed, and failed worse than the T4 note suggested.** The output is a
plausible factory video that is not an edit of the source: the camera drops from the
overhead security angle to near eye level, the press bank is replaced, the lighting
changes from dim artificial to near-daylight, and the worker still is not on the green
walkway. 15.89 dB against a 32.60 dB VAE ceiling says the same numerically — about 17 dB
of the loss is the edit discarding the source, not the autoencoder. Drafted review:
`outputs/stage4_edit/review_seed20261103_s0.8_n50.md`, unsigned pending clip playback.

Actionable cause recorded there: `strength=0.8` retains only 20 percent of the source
latent trajectory, and a `guidance_scale` of 6.0 with a prompt that re-describes the whole
room invites the model to regenerate the room from text rather than edit the video.

One real bug was found by reading this job's logs. `run_baseline.py` divided elapsed
generation time by the *requested* step count, not the executed one, so every
`s_per_step` it had written was understated. Fixed; the manifest now records
`num_inference_steps` and `denoising_steps_executed` separately, and the timing field is
`s_per_denoising_step`. The same arithmetic explains the long-standing "40 versus 50"
discrepancy, which was never a configuration error: see `bridges2/MISSING_FILES.md`.

### 3b. Free Colab T4, 2026-11-03 — superseded, outputs lost

**Stages 1 through 4 ran end to end on free Colab, 2026-11-03.** One real source clip
produced an edited clip on an assigned T4.

**That clip did not produce a valid walkway counterfactual.** The run completed; the
intended behaviour change did not occur. This is a failed feasibility test with an
actionable cause, which CLAUDE.md treats as an honest result and not as a successful
generation.

Two caveats on everything in this section. The output clip and its manifest were excluded
from the transfer to Bridges-2 and no longer exist, so the numbers below are transcriptions
that cannot be checked against the manifest. And the generation step count reported here
contradicts `configs/experiment.json`: see the discrepancy note in
`bridges2/MISSING_FILES.md`.

### Measured results

| Stage | Measurement |
|---|---|
| VAE load | 17.5 s, 862 MB download at 116 MB/s |
| VAE round trip (49x480x720) | 54.3 s, peak 6.729 GiB, latent `(1, 16, 13, 60, 90)` |
| VAE fidelity | PSNR 32.56 dB, mean abs diff 3.56/255 |
| Prompt encode (T5-XXL, fp16) | 37.0 s, weights resident 10.75 GiB, peak 10.83 GiB |
| Prompt embeddings | `(1, 226, 4096)` for both positive and negative |
| Pipeline load without text encoder | 53.7 s, 3.39 GB download at 47.2 MB/s |
| System RAM during generation | 3 GiB used, 8 GiB available of 12 |
| Generation (40 steps, strength 0.8) | **820.5 s (13.7 min), 18.78 s/step, peak 6.234 GiB** |

Stage 3 visual check passed. The yellow boundary lines and green walkway edge stay sharp
through the VAE round trip, and the worker's silhouette and foot contact survive. Machine
signage softens and floor micro-texture smooths; neither affects the label.

### Environment

| Item | Result |
|---|---|
| GPU assigned | Tesla T4, sm_75, 15360 MiB, 14.46 GiB free, no competing processes |
| System RAM | 12 GiB total, 11 GiB available |
| Disk | 113 GB free |
| Runtime | Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0 |
| Dependency install | All 7 pins resolved exactly as requested, pip exit 0 |

Four notebook bugs were found by this run and fixed:

1. `%pip install -q {" ".join(PINS)}` passed the pins through a shell, which read `<1.0` in
   `huggingface_hub>=0.34,<1.0` as an input redirection. The install never ran while the cell
   still printed a success message. Replaced with a `subprocess.run([...])` argv call that
   raises on a non-zero exit.
2. The measured 12 GiB system RAM is smaller than the 13.78 GB checkpoint
   (text_encoder 9.52, transformer 3.39, vae 0.86). `enable_model_cpu_offload()` holds every
   weight in system RAM, so the original single-phase load would have exhausted RAM. Replaced
   with two-phase loading: the T5-XXL encoder goes to the GPU alone, encodes the prompts, and
   is released before the pipeline loads with `text_encoder=None`.

3. `encode_prompt()` is a plain method, so unlike the pipeline's `__call__` it does not
   disable autograd. Calling it directly made T5-XXL retain a backward graph nobody needed,
   pushing allocation from 10.83 GiB to 14.28 GiB and exhausting the card. Wrapped in
   `torch.no_grad()`.
4. After that failure, the leaked allocation could not be released from inside the kernel:
   IPython output history and retained traceback frames still referenced the tensors, so
   `gc.collect()` and `empty_cache()` freed nothing and the next attempt OOMed identically.
   Added a pre-load free-VRAM guard that aborts with an explicit instruction to restart the
   session, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in the first cell
   before CUDA initialises.

`torch.cuda.is_bf16_supported()` returns `True` on this T4 because recent torch counts
emulation. sm_75 has no native bfloat16, so `float16` remains correct.

Still outstanding: a completed `templates/REVIEW_TEMPLATE.md`. It was never filled in for
the T4 run, and the clip it would have assessed is gone, so it cannot be filled in
retrospectively. The review attaches to the Bridges-2 run instead.

To change this: run `slurm/baseline.sbatch` as described in `bridges2/README.md`. Outputs
land in Ocean, so they survive the job.

---

## 4. Blocked

### Stage 5, classifier-guided feasibility gate — blocked, no workaround taken

The Unsafe-Net paper publishes **no code repository and no trained weights**. Its
data-availability statement covers the dataset only, and the Mendeley record
(doi:10.17632/xjmtb22pff.1) contains 691 `.mp4` files and no model artefacts. There is
therefore no classifier whose label space contains *Safe Walkway* and *Safe Walkway
Violation* as the paper defines them.

Per CLAUDE.md, no unrelated classifier has been substituted and no claim is made that a full
classifier-guided implementation was tested. The notebook ships a guarded gradient-path probe
(`RUN_GRADIENT_PROBE = False`) that has never been executed.

Unblocking requires, in order: training an 8-class classifier on the Mendeley dataset;
verifying its label space actually contains class 4; verifying gradients reach the optimised
latent through classifier preprocessing, the classifier and VAE decoding; and replacing the
stock pipeline loop, which runs under `torch.no_grad()`. Dataset-scale training is outside
this feasibility milestone.

### Deferred by user decision

- **GitHub remote.** The Colab sync route needs one. The user chose to set it up when the
  work is ready to run. `.gitignore` is in place; `git init` has not been run.

### Dependency pins

`requirements-colab.txt` was installed and verified on free Colab on 2026-11-03; every pin
resolved exactly as requested. `transformers==4.57.6` was chosen over the 5.x line to avoid a
major-version break in the T5 encoder and tokenizer that CogVideoX depends on.

That file is **not** a Bridges-2 specification: it deliberately omits `torch`, because Colab
shipped its own CUDA-matched build. Bridges-2 needs an explicit torch pin, which lives in
`bridges2/requirements-bridges2.lock.txt`.

---

## Honest summary, as of 2026-11-04

**The initial success criterion is met. The experiment then failed at the next question,
twice, on two different GPUs.** One real source clip produced an edited clip, first on a
free Colab T4 and then on a Bridges-2 L40S, with reproducible settings and measured
resource use. Neither clip is a valid walkway counterfactual. On the L40S run the whole
scene was regenerated rather than edited. Per `CLAUDE.md`, a failed feasibility test with a
documented cause is an honest result and is not a successful generation.

The two weaknesses in the earlier record are now repaired:

1. The lost T4 outputs are not recovered and will not be. What they were evidence *for* has
   been re-established instead: the Bridges-2 run at the same seed, strength, steps and
   guidance fails the same way, and is reviewed in writing rather than informally.
2. The "40 versus 50" step discrepancy is resolved. It was never a configuration mismatch;
   the pipeline truncates its schedule by `strength`. The real defect was a derived timing
   figure, now fixed and covered by tests.

Both are written up in `bridges2/MISSING_FILES.md`.

What is still outstanding, in order of how much it costs the result:

- **The review is a draft, not a verdict.** Several rows in
  `outputs/stage4_edit/review_seed20261103_s0.8_n50.md` need the clips played back and
  cannot be settled from still frames. It is unsigned.
- **No valid counterfactual has been produced.** Slurm job 50213456 sweeps strength 0.5,
  0.6 and 0.7 at guidance 6.0, plus 0.6 at guidance 3.5, to measure the preservation
  against change trade-off instead of guessing one more point on it.
- **Stage 5 remains blocked** by a missing artefact that no implementation work here can
  supply.

One expectation worth stating before the sweep returns, so it is on the record either way.
A pretrained CogVideoX-2b, prompt-guided and never fine-tuned on this dataset, has no
handle on "this worker, that walkway" — it has a text prompt describing a whole frame. The
strength high enough to move the worker may well be the strength that redraws the room, in
which case no point on the curve satisfies both criteria. That gap is what LD-ViCE closes
with per-dataset fine-tuning plus classifier gradients, and both are out of scope for this
milestone. If the sweep shows no viable point, the deliverable is the trade-off curve and
its cause, not a counterfactual.

---

## 6. Classifier-guided experiment, started 2026-11-04

Approved by the user on 2026-11-04 with a hard budget of **10 SU**. Tracked in
`docs/SU_LEDGER.md`. **0.00 SU spent so far**: everything below is login-node and network
work, and no job has been submitted under this budget.

Gates are pre-registered in `docs/ACCEPTANCE_CRITERIA.md`, written before any training so
thresholds cannot be chosen after seeing results.

### Implemented and locally checked

| Item | Path | Checked |
|---|---|---|
| Clip inventory, label map, media metadata | `src/unsafenet_cf/clipset.py` | metadata read for a real clip |
| Perceptual hashing, duplicate detection, recording and scene grouping | `src/unsafenet_cf/grouping.py` | compiles; exercised by the audit |
| r3d_18 model and the differentiable preprocessing shared by training and guidance | `src/unsafenet_cf/classifier.py` | gradient verified finite and non-zero on CPU |
| Stage A audit and split construction | `scripts/audit_splits.py` | run on a 12-clip subset |
| Stage B probe and training | `scripts/train_classifier.py` | compiles; not yet run |
| Dataset folder downloader with sha256 verification | `scripts/fetch_class_folder.py` | downloaded 178 clips, all verified |

### Data in hand

178 class-0 training clips, 2.80 GB, every file verified against the publisher's sha256.
Class 4 is **blocked**; see below.

### Blocker: the Mendeley API will not list the dataset root

`bridges2/DATASET_ACCESS.md` records every route tried. A folder can be listed once any
file inside it is known, which is how the class-0 training folder was reached, but the root
itself returns an empty array and the bulk-archive route returns 404. Folder ids are random
UUIDs and cannot be derived.

Unblocking needs one file UUID from each of three folders, taken from a browser, or a
browser download transferred with rsync. The two are not equivalent: the UUID route keeps
checksum verification against the publisher's hashes, the transfer route does not, and the
manifests say which was used.

Without class-4 clips there is nothing to train, so Gates B, C, D and E are all blocked.

### Bugs found and fixed during implementation, none of them on a GPU

1. Both decode paths originally read a whole clip into memory before sampling frames. A
   262-frame 1080p clip is about 1.6 GB that way, multiplied by hundreds of clips. Both now
   stream and keep only the wanted frames, downscaled.
2. Neither `ffmpeg` nor `ffprobe` is on PATH on a Bridges-2 login node. Decoding now uses
   the binary bundled with the installed `imageio_ffmpeg` wheel, and the `pyav` plugin,
   which is also absent, is no longer used.
3. Reading container metadata discarded it when closing the reader raised. Closing makes
   ffmpeg exit and that is expected; the metadata is kept regardless.
4. Gate A reported PASS on a subset where every clip had failed to decode, so the
   cross-split overlap check had not run at all. An undecodable clip now fails Gate A.

---

## 5. Platform change: Colab to PSC Bridges-2, 2026-11-03

By user decision, the GPU stages move from free Colab to PSC Bridges-2. This **overrides**
the free-Colab-only requirement in `CLAUDE.md`. The scientific scope is unchanged: still one
clip, still a prompt-guided editing baseline, still no safety claim, still no LD-ViCE
reproduction.

What changed, and what did not:

| | Colab T4 | Bridges-2 |
|---|---|---|
| Interface | notebook in a browser | `sbatch` on the GPU-shared partition |
| GPU | Tesla T4, 15 GB, sm_75, fp16 only | L40S-48, 48 GB, sm_89 (requested; the probe job confirms) |
| System RAM | 12 GiB, the binding constraint | 500 GiB or more |
| Text encoding | two-phase, forced by the 12 GiB RAM limit | single-phase, whole checkpoint resident |
| Offloading | `enable_model_cpu_offload()` required | not required; kept as a flag |
| torch | whatever Colab shipped | pinned `2.9.1+cu128` |
| Weights | re-downloaded every session | staged once in Ocean |
| Outputs | ephemeral, lost on disconnect | written to Ocean, which is why they survive now |

Status of the Bridges-2 work is tracked in `bridges2/README.md`. The hardware probe and
the full baseline have both executed; see section 3a for the measurements.

---

## 6. Multi-source walkway comparison, 2026-11-05

Three *new* walkway-violation sources edited with and without the existing LoRA adapter.
No retraining, no classifier-gradient guidance, adapter scale 1.0, seed 20261103,
strength 0.6, text guidance 6.0, 50 requested steps.

**Outcome: no source improved.** `0_tr16` unchanged, `0_tr125` unchanged, `0_tr48`
inconclusive. The worker reaches the marked lane in 0 of 49 frames in every one of the six
edits.

| Source | Role | Source gap | base | lora | Verdict |
|---|---|---|---|---|---|
| `0_tr16` | near the lane, camera view absent from LoRA training | 262.4 px | -15.1 px | -6.7 px | unchanged |
| `0_tr125` | far from the lane, same view as `0_tr143` | 394.6 px | -1.4 px | -9.8 px | unchanged |
| `0_tr48` | different walking direction | 107.7 px | +36.4 px | +29.1 px | inconclusive, 24/49 source detector misses |

The stronger and more consistent effect is on the environment, not the worker. Lane IoU
against each source's own marking: base 0.0000, 0.0017, 0.0120; LoRA 0.2443, 0.0477, 0.4207.
The base edit erases or relocates the marked lane in all three sources. That is the outcome
`configs/experiment.json` names as the one to reject.

Only **two** camera views exist among eligible sources, not three; `0_tr16` and `0_tr48`
share a view at r 0.929. The third source supplies a different walking direction instead.
Three clips support no claim about generalisation.

Full write-up `outputs/multi_walkway/FINDINGS.md`. Package
`/ocean/projects/cis260288p/puprety/transfer/multi-walkway-review/`, 54 files, 99 MB,
checksums verified. Session narrative in `SESSION_LOG_2026-11-04.md`.
