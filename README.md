# UnsafeNet video counterfactual feasibility experiment

Edit one unsafe workplace-safety video clip into a candidate safe counterfactual using
CogVideoX. The GPU stages now run on **PSC Bridges-2** under Slurm; they previously ran on
free Google Colab.

**Status, 2026-11-03.** Stages 1 to 4 ran end to end on a free Colab T4 and **the run
completed without producing a valid walkway counterfactual.** The intended behaviour change
did not occur. That is a documented feasibility failure, not a successful generation.

The move to Bridges-2 excluded `data/` and `outputs/`, so that run's edited clip and
manifest no longer exist and its result cannot be re-inspected. The baseline is being
reproduced on Bridges-2 before anything else is attempted.

- [`bridges2/README.md`](bridges2/README.md) — how to run on Bridges-2
- [`bridges2/MISSING_FILES.md`](bridges2/MISSING_FILES.md) — what the transfer lost and what
  is recoverable
- [`STATUS.md`](STATUS.md) — the line between implemented, locally checked, GPU-executed and
  blocked

---

## What this is, and what it is not

| | |
|---|---|
| **Is** | A prompt-guided video-to-video editing baseline on a pretrained `THUDM/CogVideoX-2b` |
| **Is not** | A reproduction of LD-ViCE |
| **Is not** | Evidence that any generated clip depicts safe behaviour |
| **Is not** | Classifier-evaluated — no Unsafe-Net classifier exists publicly |

LD-ViCE ([arXiv:2509.08422](https://arxiv.org/abs/2509.08422)) fine-tunes CogVideoX
independently for each evaluated dataset and steers denoising with target-model gradients,
SmoothGrad, and an artifact refinement stage. This project does none of those. Calling its
output an LD-ViCE result would be false.

Outputs are **candidate counterfactuals** until a human completes
[`templates/REVIEW_TEMPLATE.md`](templates/REVIEW_TEMPLATE.md).

---

## Source data

**Dataset:** *Video Dataset for Safe and Unsafe Behaviours*,
[doi:10.17632/xjmtb22pff.1](https://data.mendeley.com/datasets/xjmtb22pff/1),
Önal & Dandıl, Bilecik Şeyh Edebali University. **CC BY 4.0.** 691 clips, 10.0 GB.

**Paper:** Önal, O., Dandıl, E. *Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system
for real-time detection of unsafe behaviours in workplace.* Multimedia Tools and Applications
(2025) 84:34967–34993. [doi:10.1007/s11042-024-19276-8](https://doi.org/10.1007/s11042-024-19276-8)

Dataset identity was verified against the paper: same authors, same institution, same 8-class
label space, and 691 clips matching the paper's 566 training + 125 test split.

Filenames encode the label: `<classID>_<tr|te><index>.mp4`.

| ID | Class | Type | Pair |
|---|---|---|---|
| 0 | Safe Walkway Violation | unsafe | ↔ 4 |
| 1 | Unauthorized Intervention | unsafe | ↔ 5 |
| 2 | Opened Panel Cover | unsafe | ↔ 6 |
| 3 | Carrying Overload with Forklift | unsafe | ↔ 7 |
| 4 | Safe Walkway | safe | |
| 5 | Authorized Intervention | safe | |
| 6 | Closed Panel Cover | safe | |
| 7 | Safe Carrying | safe | |

### Privacy

The clips show identifiable factory workers who consented to the authors' research study,
not to redistribution by third parties. CC BY 4.0 permits redistribution legally; this
project chooses not to. `data/` and `outputs/` are gitignored, and the notebook downloads the
clip directly from Mendeley rather than carrying it in the repository.

---

## The selected task

| Field | Value |
|---|---|
| Source clip | `0_tr143.mp4` (3,120,796 bytes, sha256 `7a7c0d10204bca0e…`) |
| Source format | 1920×1080, 25 fps, 262 frames, 10.44 s, h264 |
| Unsafe label | **0 Safe Walkway Violation** |
| Target safe label | **4 Safe Walkway** |
| Desired change | The worker moves from the bare concrete aisle onto the green marked walkway |
| Segment used | source frames 0–144, every 3rd frame → 49 frames, 5.88 s of real time |
| Crop | 1620×1080 at `x=300`, then scaled to 720×480 |

**Why this clip.** Of the three smallest class-0 clips inspected, only `0_tr143.mp4` has the
violating worker large (about 40% of frame height), unoccluded and centre-frame, with the
empty green walkway visible in the same shot.

**Why `x=300` and not a centred crop.** The source is 16:9 and the checkpoint wants 3:2, so
1620 of 1920 columns survive. A centred crop (`x=150`) clips the green walkway at the right
edge — which is exactly the thing the counterfactual has to move the worker onto.

**Expected difficulty.** This counterfactual requires relocating a person roughly 700 source
pixels sideways. A pretrained prompt-guided model may well fail and simply restyle the clip.
That is a real feasibility result and is reported as such, not hidden.

**The failure mode to watch for.** The model can "succeed" by repainting or extending the
green walkway *underneath* the worker. That flips the visual label by changing the
environment rather than the behaviour. The review template rejects it explicitly.

---

## Checkpoint constraints

Read from the published configs of `THUDM/CogVideoX-2b`, not from memory:

| Source | Values |
|---|---|
| `vae/config.json` | `sample_height=480`, `sample_width=720`, `temporal_compression_ratio=4`, `latent_channels=16`, `scaling_factor=1.15258426`, `invert_scale_latents=false` |
| `transformer/config.json` | `sample_frames=49`, `sample_height=60`, `sample_width=90`, `patch_size=2` |
| `scheduler/scheduler_config.json` | `CogVideoXDDIMScheduler`, `prediction_type=v_prediction`, `rescale_betas_zero_snr=true` |

So: 720×480 pixels → 90×60 latents, 49 frames → 13 latent frames, and the pipeline input
tensor is `(1, 16, 13, 60, 90)`.

`CogVideoXVideoToVideoPipeline.__call__` takes `video`, `prompt`, `negative_prompt`,
`height`, `width`, `num_inference_steps`, `strength`, `guidance_scale`. It has **no
`num_frames` argument** — the frame count is `len(video)`.

---

## Local preparation (MacBook M1)

No model weights and no GPU stack are needed locally.

```bash
cd ~/workspace/unsafenet-counterfactuals

python3 -m venv .venv
./.venv/bin/pip install -r requirements-dev.txt

# Run the checks
./.venv/bin/python -m pytest tests -q

# Reproduce the preprocessed clip (needs data/raw/0_tr143.mp4)
./.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from unsafenet_cf import preprocess as pp
spec = pp.load_spec('configs/clip_0_tr143.json')
print(pp.preprocess_clip(spec, 'outputs/preprocessed/0_tr143_720x480x49.mp4')['output_probe'])
"
```

`ffmpeg` is not required on `PATH`: `resolve_ffmpeg()` falls back to the binary bundled with
the `imageio-ffmpeg` wheel.

To fetch the source clip locally:

```bash
mkdir -p data/raw
curl -sSL -o data/raw/0_tr143.mp4 \
  "https://data.mendeley.com/public-files/datasets/xjmtb22pff/files/f48e2cbc-bd0e-42b4-bb7e-9ca3980143be/file_downloaded"
shasum -a 256 data/raw/0_tr143.mp4
# must print 7a7c0d10204bca0e4cfdf7bdb699ab16383b37800471fd6f3a03b7a1780903f3
```

---

## Running on free Colab

1. Open [`notebooks/unsafenet_cogvideox_feasibility.ipynb`](notebooks/unsafenet_cogvideox_feasibility.ipynb)
   in Colab.
2. **Runtime → Change runtime type → T4 GPU.** Run cell 1 first: it records the GPU, VRAM,
   RAM, disk and versions *before* any weights load. If `nvidia-smi` is missing, no GPU was
   assigned; stop and change the runtime.
3. Run the install cell. It pins everything except `torch`, which stays as Colab shipped it.
4. Give the notebook the helpers. Either set `REPO_URL` to your GitHub remote, or upload
   `src/`, `configs/` and `templates/` to `/content/unsafenet-counterfactuals`.
5. Run the remaining cells in order. The clip downloads itself from Mendeley and its SHA-256
   is verified before use. **No upload and no Drive mount is required.**
6. Download the output archive after each stage. Free Colab wipes the runtime on disconnect.

### Syncing Colab with local work

Use GitHub for code, Mendeley for data, and neither for weights.

- **Code**: edit on the M1, push, and `git pull` in cell 4. One source of truth, no
  copy-paste drift.
- **Data**: the notebook `curl`s the 3 MB clip. Never upload 10 GB and never commit a clip.
- **Outputs**: written to `outputs/`, zipped, and downloaded. Set `MOUNT_DRIVE = True` in the
  last cell if you prefer Drive.
- **Weights**: `~/.cache/huggingface` dies with the runtime. CogVideoX-2b is a multi-GB
  download each session. Caching it in Drive is optional and off by default.

### Memory

The T5-XXL text encoder, not the 1.7B transformer, dominates memory. `enable_model_cpu_offload()`
keeps one component resident at a time; VAE tiling and slicing handle the 49×480×720 decode.
If a stage still runs out of memory, the documented fallback is `enable_sequential_cpu_offload()`
(under 4 GB, much slower). Change one thing at a time and record it. Do not loop on identical
failures.

---

## Repository layout

```
configs/clip_0_tr143.json     selected clip, crop, segment, labels
configs/experiment.json       checkpoint, seed, prompt, strength, what must not change
data/raw/PROVENANCE.json      DOI, licence, class table, per-file SHA-256  (clips gitignored)
notebooks/…feasibility.ipynb  the Colab notebook, 22 cells
src/unsafenet_cf/preprocess.py  checkpoint constraints, crop maths, ffmpeg command, probing
src/unsafenet_cf/manifest.py    environment capture, GPU/memory snapshots, run manifests
src/unsafenet_cf/compare.py     PSNR, mean abs diff, comparison contact sheets
templates/REVIEW_TEMPLATE.md    human review against all four success criteria
tests/                          49 local checks, no GPU required
```

`src/` imports `torch`, `numpy` and `PIL` lazily, so the preprocessing and manifest logic is
testable on a machine with no GPU stack.

---

## Required user inputs

Everything needed for stages 1–4 is already in the repository. What remains:

1. **Run the notebook on Colab** and report back the cell-1 hardware block, any errors, and
   the output archive. Execution is unverified until then.
2. **Decide on GitHub.** The sync route above needs a remote. Deferred by request.
3. **Complete the review template** after the first successful generation. No safety claim is
   made before that.

---

## Limitations

- **One GPU run has happened, on a Colab T4, and it failed its objective.** It produced an
  edited clip but not a valid walkway counterfactual. Nothing has yet run on a Bridges-2 GPU.
- **That run's outputs are gone.** Excluded from the transfer. Its reported numbers are
  transcriptions with no surviving manifest behind them, and one of them is internally
  inconsistent. See [`bridges2/MISSING_FILES.md`](bridges2/MISSING_FILES.md).
- **No classifier exists.** The Unsafe-Net authors released no code and no weights; the
  Mendeley record contains videos only. Stage 5 is blocked, documented, and not faked with a
  substitute classifier.
- **Dependency pins differ per platform.** `requirements-colab.txt` omits `torch` on
  purpose and was verified on Colab. Bridges-2 pins `torch==2.9.1+cu128` explicitly;
  `bridges2/requirements-bridges2.lock.txt` is the frozen set that actually installed.
- **One clip, one configuration.** No benchmark, no statistics, no generalisation claim.
- **A prediction flip would not prove safety** even once a classifier exists. Absence of an
  unsafe label is not presence of a safe one.
- **The paper's own system is not reimplemented.** Unsafe-Net's reported 95.81% accuracy is
  the authors' result on their own data and is cited, not reproduced.

---

## Attribution

Dataset: Önal, O.; Dandıl, E. *Video Dataset for Safe and Unsafe Behaviours.* Mendeley Data,
V1 (2024). doi:10.17632/xjmtb22pff.1. Licensed CC BY 4.0.
