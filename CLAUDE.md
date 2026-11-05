# UnsafeNet video counterfactual feasibility experiment

## Platform override, 2026-11-03

The GPU stages run on **PSC Bridges-2** under Slurm, not on free Google Colab. This
supersedes every Colab-only requirement below, including the "free Google Colab" instruction
in the Assignment section and all of Stage 2's Colab-specific wording. Read
`bridges2/README.md` before running anything.

What carries over unchanged: no paid compute or paid APIs, no local model execution on the
M1, no hosted web UI or background service, and every scientific constraint in this file. The
free-Colab notebook stays in the repository as the record of the earlier T4 run.

What changes: `sbatch` replaces notebook cells, the allocation is finite and charged in SUs,
model workloads must never run on a login node, and outputs live in Ocean rather than an
ephemeral runtime.

## Assignment
Build a reproducible, staged experiment that edits one unsafe source video into a candidate safe counterfactual using CogVideoX. Prepare code locally and run GPU stages in **free Google Colab**. The user's local machine is a MacBook M1; its memory is unknown. Do not require local model execution, paid compute, or paid APIs.

Start by reading this file. Inspect the project and relevant official documentation before implementation. Work through the stages below in order. Implement what can be completed without the user's dataset or live Colab session. Ask for missing inputs only when they block execution; never fabricate dataset labels, clips, classifier weights, results, or benchmark claims.

## Research context and evidence
- LD-ViCE: Latent Diffusion Model for Video Counterfactual Explanations: https://arxiv.org/abs/2509.08422
- Full method: https://arxiv.org/html/2509.08422v1
- CogVideoX pipelines: https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox
- CogVideoX VAE: https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl_cogvideox
- Colab limitations: https://research.google.com/colaboratory/faq.html

LD-ViCE encodes the factual video, adds noise to its latents, and denoises using text conditioning plus target-model gradients. It uses SmoothGrad and an artifact refinement stage. The paper reports independently fine-tuning CogVideoX for each evaluated dataset. A pretrained prompt-guided baseline is therefore not a reproduction of LD-ViCE. Verify the paper's exact implementation, checkpoint, and any available author code before claiming fidelity.

CogVideoX's matching pretrained 3D VAE (AutoencoderKLCogVideoX) translates video pixels to and from latents. The diffusion transformer performs the edit. The VAE alone reconstructs video; it does not understand or enforce safety. Use the VAE and latent scaling conventions that match the chosen checkpoint.

## Scope and success criteria
Initial success means one real source clip produces an edited clip on an actually assigned free Colab GPU, with reproducible settings and measured resource use. A failed feasibility test with an actionable, documented cause is also an honest result; it is not successful generation.

Evaluate separately:
1. Whether the intended safety-critical behavior or equipment changed.
2. Whether worker identity, task, objects, camera, and scene remain sufficiently preserved.
3. Temporal consistency, physical plausibility, and artifacts.
4. Target classifier prediction, /cavemanonly when a suitable classifier exists.

A changed classifier prediction does not prove that a scene is safe. Human review against explicit safety criteria is necessary. Describe outputs as candidate counterfactuals until reviewed. No claims of causal validity from a prediction flip alone.

## Stage 1: Verify inputs and prepare locally
- Locate or request the actual UnsafeNet source, label definitions, and one short authorized clip. Verify the dataset identity and access conditions.
- There may be related work in two of the user's other local directories; ask for their paths when it is relevant. Inspect only relevant files and their instructions. Do not modify those projects or copy private material without need.
- Select one label with a concrete visible safe alternative. Helmet addition was a hypothetical example, not a verified UnsafeNet label or selected task.
- Record source clip, temporal segment, unsafe label, desired safe behavior, prompt, and context that must remain unchanged.
- Inspect clip metadata and create a reproducible preprocessing path. Follow verified checkpoint frame-count and resolution constraints. Do not arbitrarily shorten tensors to unsupported shapes.
- Keep source data and large weights out of version control.

## Stage 2: Create the free Colab notebook
- Use a notebook with visible, sequential cells. Do not require a hosted web UI, background service, or paid subscription.
- Record GPU model, free/total VRAM, system RAM, Python, PyTorch, CUDA, and package versions before loading weights.
- Start with THUDM/CogVideoX-2b if verified compatible with the selected video-to-video implementation.
- Use a supported native video-to-video pipeline where it provides the needed behavior. Verify its current API and source-video initialization rather than inventing a custom loop prematurely.
- Pin a tested coherent dependency set; record model revision, seed, prompt, scheduler, strength, steps, frame count, resolution, dtype, and optimizations.
- Use supported CPU offloading and VAE tiling as needed. Avoid assuming quantization methods work on every assigned GPU or preserve gradient support.
- Upload a single clip without embedding account tokens. Make persistent output storage optional and save downloadable results after each stage.

## Stage 3: VAE reconstruction check
- Load only the matching VAE where possible. Encode and decode the preprocessed source without diffusion editing.
- Save the preprocessed source and reconstruction side by side.
- Inspect whether small safety-relevant details survive. Record reconstruction artifacts and peak GPU memory.
- Release unused components before loading later stages.

## Stage 4: Prompt-guided source-video editing baseline
- Initialize from the actual source video's encoded latents, not random noise or only its first frame.
- Generate one candidate using an explicit safe-behavior prompt and fixed seed.
- Start with one supported configuration and a small number of trials; inspect the effect of edit strength only as needed.
- Save original, reconstruction, edited clip, comparison frames, settings, runtime, and measured peak GPU memory.
- If out of memory, release stale tensors and make one documented change at a time. Preserve valid model input constraints. Stop repeated identical failures.
- Clearly label this as a prompt-guided editing baseline, not LD-ViCE or verified safety.

## Stage 5: Classifier-guided feasibility gate
Only proceed after baseline execution fits and a suitable differentiable classifier is available.
- Verify the classifier's label space includes the intended safe target. Absence of one unsafe label is not equivalent to safety.
- Verify gradients through classifier preprocessing, classifier, and VAE decoding reach the optimized latent. Frozen model weights must not block input gradients.
- Standard inference pipelines may disable autograd. Inspect and adapt only the responsible loop; do not assume calling a stock pipeline enables classifier guidance.
- Run one guided denoising-step probe first. Record finite/nonzero gradient checks, memory, and timing before a full loop.
- Keep inference-only optimizations separate from operations requiring autograd. Verify offloading/quantization compatibility.
- If feasible, implement target-loss guidance, then assess whether SmoothGrad and refinement fit. Re-evaluate the classifier after refinement.
- If no classifier or insufficient memory exists, document the exact blocker. Do not substitute an unrelated classifier or claim a full implementation was tested.

## Deliverables
1. README.md with exact local preparation and Colab run instructions, required user inputs, and limitations.
2. One clear Colab notebook for hardware inspection, preprocessing, VAE reconstruction, and source-video editing.
3. Small reusable Python helpers only where they reduce notebook complexity.
4. A run manifest and lightweight results/review template covering all success criteria above.
5. A status report separating implemented, locally checked, actually GPU-executed, and blocked work.

Keep the system minimal. Do not build a dashboard, service, training framework, or large benchmark. Full dataset fine-tuning is outside this first feasibility milestone. Do not download full datasets or models onto the M1 merely to prepare the notebook.

## Validation and handoff
Validate notebook syntax and helper logic locally with small meaningful checks. Do not label syntax validation as GPU validation. If browser access or a Colab account is unavailable, deliver a ready-to-upload notebook and precise instructions for the user; state that execution remains unverified. Preserve outputs outside ephemeral runtime storage, and report failures as well as successes.

## First action
Inspect relevant existing assets and official implementation support. Then build the smallest notebook that can test VAE reconstruction and one source-video edit on free Colab. Request the chosen dataset clip and label if unavailable, while completing independent setup work.
