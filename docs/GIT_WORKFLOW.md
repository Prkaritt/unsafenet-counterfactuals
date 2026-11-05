# Git workflow

The repository was initialised on 2026-11-05, after four experiments had already run. The
initial commit imports the project as it stood at that point, so the history starts at the
multi-source walkway comparison rather than at the beginning of the work.

## What is tracked and what is not

Tracked: code, Slurm scripts, configs, acceptance criteria, manifests, measurements, audit
reports and findings.

Not tracked: every clip and every frame. The source footage shows identifiable factory
workers who consented to the original study and not to redistribution, so no media goes into
version control. Also excluded are model weights and checkpoints, the decode cache under
`outputs/classifier/clip_cache/`, Slurm job logs, and the review HTML pages, which reference
media the repository does not contain.

`.gitignore` excludes `data/**` and `outputs/**` and then re-admits only `.json`, `.md`,
`.csv` and `.txt` from inside them. It also carries a blanket media guard on `*.mp4`, `*.png`
and similar extensions, anywhere in the tree. A diagram that genuinely belongs in the
repository therefore needs `git add -f`, which is deliberate: the guard should be harder to
defeat by accident than a clip is to drop in the wrong folder.

Check before every commit that nothing media-shaped slipped in:

```bash
git diff --cached --name-only | grep -Ei '\.(mp4|png|jpg|jpeg|safetensors|ckpt|pt|bin)$'
```

No output means nothing media-shaped is staged.

## Branch per major change

`main` holds the record of what has actually run. Each major change gets its own branch, so
an experiment that fails can be left intact rather than unwound.

Name branches `exp/NN-slug`, continuing the numbering the write-ups already use: attempt 01
was the prompt-guided baseline, 02 the classifier-guided run, 03 the LoRA fine-tune, 04 the
multi-source comparison. The next one is 05.

```bash
git switch -c exp/05-classifier-repair main
```

Work, then commit in steps small enough to read later:

```bash
git add scripts/train_classifier.py docs/ACCEPTANCE_CRITERIA.md
git commit
```

Put the Slurm job ID and the SU cost in the commit message of any commit that records a run.
That is what ties a commit to a row in `docs/SU_LEDGER.md`:

```
Stage C re-run with the repaired worker-dependence check

Slurm job 50251xxx, 0.08 SU. Balanced accuracy drops 0.0431 when the worker is
masked, against +0.0278 for the control region, so Gate C now passes.
```

When the branch reaches a result, merge it into `main` with `--no-ff` so the branch stays
visible as a unit in the history, and tag it:

```bash
git switch main
git merge --no-ff exp/05-classifier-repair
git tag -a exp-05 -m "Classifier repair: Gate C passes"
```

Keep the branch after merging. It costs nothing and it is the thing you extend later.

A negative result is merged the same way as a positive one. The point of the history is what
ran, not what worked.

## Extending a branch you finished earlier

Two situations, and they need different commands.

**The branch is still the right base and `main` has not moved in a way that matters.** Switch
back to it and keep committing:

```bash
git switch exp/03-lora
git commit -m "Second LoRA configuration: rank 32, lr 5e-5"
```

Then merge into `main` again when it is done. The second merge brings only the new commits.

**`main` has moved on and the branch needs what landed since.** Bring `main` into the branch
before continuing, so the branch stays a readable sequence:

```bash
git switch exp/03-lora
git merge main          # or: git rebase main, only if the branch is unpushed
git commit -m "..."
```

Use `merge` once the branch has been shared anywhere, and `rebase` only while it is still
local to this machine. Rebasing rewrites commit IDs, which breaks anyone who already has
them.

**You want a variant rather than a continuation.** Branch off the old branch instead of
extending it, so the two stay comparable:

```bash
git switch -c exp/06-lora-rank32 exp/03-lora
```

That is the right shape for a sweep: one branch per configuration, all sharing a parent, each
one a thing you can check out and reproduce.

## Finding your way back

```bash
git log --oneline --graph --all --decorate   # the whole shape, branches and tags
git branch -v                                # every branch and its tip
git tag -l -n1                               # completed experiments with their one-liners
git log --oneline exp/03-lora ^main          # what a branch has that main does not
git switch --detach exp-03                   # the tree exactly as it was at that tag
```

## What git does not cover here

The clips are not in the repository, so a commit does not capture what a run produced. A
commit pins the code, the settings and the measurements; the media lives in Ocean beside it
and in the transfer packages under
`/ocean/projects/cis260288p/puprety/transfer/`. When a commit refers to a clip, name the
manifest that describes it, so the two can be matched later.
