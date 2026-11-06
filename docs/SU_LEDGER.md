# Service-unit ledger

Approved budget for the classifier-guided experiment: **10 SU maximum**, covering training,
validation, memory and gradient probes, generation and retries. Bridges-2 GPU-shared bills
elapsed time at 1 SU per L40S GPU-hour. Slurm bills what a job uses, not what it requests,
but a job's **requested wall time is its maximum possible charge**, so concurrency is bounded
by requested time rather than by expected time.

## Rule followed when submitting

Before any `sbatch`, the sum of (requested wall hours x GPUs) over every job already running
or queued, plus the new job, must stay at or below the remaining budget. No two jobs are in
flight whose combined worst case could exceed what is left. Execution stops and reports
before 10 SU is reached.

## Spent so far

Read from `sacct` on 2026-11-04, not from memory. Every job is 1 x `l40s-48` on
`GPU-shared`, which bills 1 SU per GPU-hour of **elapsed** time, so SU = elapsed / 3600.
A FAILED job is charged for the time it ran; failures are listed because they cost
allocation.

| Job | Name | Elapsed | SU | State | What it bought |
|---|---|---|---|---|---|
| 50217313 | `cfclf` | 00:03:02 | 0.0506 | FAILED | classifier training, first attempt |
| 50217460 | `cfaudit` | 00:00:56 | 0.0156 | COMPLETED | Stage A audit |
| 50217528 | `cftrain` | 00:02:23 | 0.0397 | FAILED | classifier training |
| 50217572 | `cftrain` | 00:00:00 | 0.0000 | CANCELLED | cancelled before start |
| 50218519 | `cfaudit` | 00:01:43 | 0.0286 | COMPLETED | Stage A audit, re-run |
| 50218609 | `cftrain` | 00:04:27 | 0.0742 | COMPLETED | classifier trained |
| 50218713 | `cftrain` | 00:03:40 | 0.0611 | COMPLETED | classifier re-trained |
| 50218860 | `cfeval` | 00:01:22 | 0.0228 | FAILED | Stage C diagnostics |
| 50218934 | `cfeval` | 00:08:03 | 0.1342 | FAILED | Stage C diagnostics |
| 50219213 | `cfeval` | 00:35:18 | 0.5883 | FAILED | Stage B3 + C, full validation split |
| 50220880 | `cfguide` | 00:00:28 | 0.0078 | FAILED | Stage D probe |
| 50223164 | `cfguide` | 00:05:45 | 0.0958 | FAILED | Stage D probe |
| 50223413 | `cfguide` | 00:03:13 | 0.0536 | FAILED | Stage D probe |
| 50223481 | `cfguide` | 00:03:23 | 0.0564 | FAILED | Stage D probe |
| 50223610 | `cfguide` | 00:03:07 | 0.0519 | FAILED | Stage D probe |
| 50224249 | `cfguide` | 00:03:35 | 0.0597 | FAILED | Stage D probe |
| 50225049 | `cfguide` | 00:14:20 | 0.2389 | COMPLETED | Stage D + E, guided candidate |
| 50226774 | `cflora-probe` | 00:04:10 | 0.0694 | FAILED | interval verification done; trainer died, no protobuf |
| 50226994 | `cflora-probe` | 00:30:04 | 0.5011 | TIMEOUT | nothing: trainer piped into `tail`, output lost when Slurm killed it |
| 50229034 | `cflora-probe` | 00:11:19 | 0.1886 | FAILED | 4.14 s/step, 20.16 GiB peak; training worked, job died in upstream's post-save block |
| 50229842 | `cflora-train` | 00:02:42 | 0.0450 | FAILED | nothing: `--dataloader_num_workers 2` forks workers around CUDA latents |
| 50230058 | `cflora-train` | 01:04:46 | 1.0794 | COMPLETED | the LoRA adapter, 1000 steps |
| 50232171 | `cflora-gen` | 00:08:22 | 0.1394 | COMPLETED | matched base and LoRA edits of `0_tr143` |
| 50232395 | `cflora-verify` | 00:00:18 | 0.0050 | COMPLETED | worker position in source, reconstruction, base, LoRA |

| 50247622 | `cfsel` | 00:02:20 | 0.0389 | COMPLETED | characterised 15 candidate sources |
| 50248886 | `cfsel3` | 00:00:24 | 0.0067 | COMPLETED | characterised 6 more candidates |
| 50249193 | `cfmulti` | 00:11:51 | 0.1975 | COMPLETED | six base/LoRA edits on three new sources, plus first evaluation |
| 50250149 | `cfeval2` | 00:02:17 | 0.0381 | COMPLETED | re-evaluation with a tracked worker |

**Cumulative under this budget: 3.8883 SU. Remaining: 6.1117 SU.** Project total since
2026-11-01, including the three pre-budget jobs: 4.2533 SU.

The multi-walkway stage cost 0.2811 SU in total, under the 1.00 SU reported as its maximum
before submission. Machine-readable: `outputs/multi_walkway/su_actuals.json`.

The LoRA stage cost 2.0279 SU. Of that, **0.5461 SU bought nothing**: jobs 50226994 and
50229842 failed on mistakes in the job scripts, not in the model or the cluster. They are
listed rather than folded into the training line, because a ledger that hides its waste
cannot be used to plan the next run. Machine-readable copy: `outputs/lora/su_actuals.json`.

All dataset download, hashing, inventory, preprocessing and code work runs on a login
node and is not charged against an allocation.

## Earlier charges, outside this budget

These predate the classifier experiment and are recorded for the project total, not against
the 10 SU.

| Job | Name | Elapsed | Approx SU |
|---|---|---|---|
| 50196825 | `cfprobe` hardware probe | ~2 min | ~0.03 |
| 50198591 | `cfbase` baseline, stages preprocess/vae/edit | 9 min 11 s | ~0.15 |
| 50213456 | `cfsweep` four-run strength sweep | 10 min 26 s | ~0.17 |

Allocation balance at the time of approval: 223 of 300 SU, expiring 2027-09-08.

## Planned allocation of the 10 SU

| Stage | Work | Planned | Notes |
|---|---|---|---|
| A | Inventory, hashing, duplicate and group audit | 0.0 | CPU, login node, free |
| B0 | Training memory and time probe | 0.5 | Shape compatibility is not memory fit; measure first |
| B | Train and validate the binary classifier | 1.0 | |
| C | Shortcut diagnostics and VAE consistency | 1.5 | VAE round trip is 42.8 s per clip and dominates |
| D | Guided-step gradient probe | 1.0 | |
| E | One guided generation plus matched unguided control | 1.5 | |
| — | Retries and the autograd/memory unknown | 4.5 | |
| | **Total** | **10.0** | |

The contingency is deliberately the largest line. The unguided generation already peaked at
36.205 GiB of 44.42 GiB, and a backward pass through a 49-frame 720x480 VAE decode has to
fit in what is left. That is the one number in this experiment that cannot be predicted from
anything measured so far.

If substantial CPU work moves to a compute node — decoding several hundred clips, for
instance — it is charged here at the partition's rate and deducted from the same 10 SU.

## Agent-specified relocation experiment, 2026-11-05

Branch `experiment/manual-path-refinement`, uncommitted. Read from `sacct`, not from
requested wall time. Every job is 1 x `l40s-48` on `GPU-shared`, billed at 1 SU per GPU-hour
of elapsed time.

| Job | Name | Elapsed | SU | State | What it bought |
|---|---|---|---|---|---|
| 50275552 | `mprseg` | 00:03:40 | 0.0611 | COMPLETED | SAM 2.1 masks and composites; **superseded**, wrote colour-inverted video |
| 50276648 | `mprseg` | 00:02:52 | 0.0478 | COMPLETED | same stage re-run after the video-writer fix |
| 50276861 | `mprref` | 00:05:48 | 0.0967 | COMPLETED | single-clip pilot refinement, `0_tr143` |
| 50276977 | `mprref` | 00:01:06 | 0.0183 | COMPLETED | refinement, `0_tr125` |
| 50276958 | `mprref` | 00:01:04 | 0.0178 | COMPLETED | refinement, `0_tr16` only; `--export` split on the comma and dropped `0_tr125` |
| 50277016 | `mprev` | 00:00:29 | 0.0081 | COMPLETED | evaluation of all nine clips |
| | **Experiment total** | | **0.2497** | | |

Two jobs bought less than they should have. 50275552 at 0.0611 SU produced correct masks and
composites but inverted video, because `export_to_video` rescales numpy frames by 255 and
uint8 wraps modulo 256. 50276958 at 0.0178 SU refined one clip instead of two, because
`sbatch --export=ALL,CLIPS=a,b` treats the comma as a separator between exported variables.
Neither is hidden; both are written up in `DECISIONS.md` D7 and in this table.

**Budget position.** 3.8883 SU had been spent of the approved 10 before this experiment.
With 0.2497 SU added, the total is **4.1380 SU spent, 5.8620 remaining**. Project total since
2026-11-01 is 4.5030 SU.

## Panel-cover counterfactual experiment, 2026-11-05

Branch `experiment/panel-cover-counterfactual`, uncommitted. Every job is 1 x `l40s-48` on
`GPU-shared`, billed at 1 SU per GPU-hour of elapsed time. Elapsed times are read from
`sacct`, never from requested wall time.

The grant carries one resource only, Bridges-2 GPU, with 223 of 300 SU remaining on
2026-11-05. There is no CPU-only RM allocation, so CPU-bound media work is charged at the
same rate as a generation, which is why every stage asks for the shortest wall time that
fits its measured throughput. Downloading and hash-verifying the two dataset folders was
done on the login node and cost nothing.

| Job | Name | Elapsed | SU | State | What it bought |
|---|---|---|---|---|---|
| 50280101 | `pcaudit` | 00:16:04 | 0.2678 | COMPLETED | audit of 148 class-2 and class-6 clips: camera views, recording groups, measured panel-candidate regions, contact sheets |
| 50282082 | `pcsel` | 00:10:54 | 0.1817 | COMPLETED | scored all 148 clips against the press-guard box of DECISIONS P5 and **disproved it**: 96 of 120 aligned class-2 clips had that guard closed |
| 50282294 | `pcpop` | 00:03:41 | 0.0614 | COMPLETED | population difference between the two classes; found the cabinet door that separates them clip by clip at balanced accuracy 0.935 (DECISIONS P7) |
| 50283251 | `pcsel` | 00:13:20 | 0.2222 | COMPLETED | rescore against the corrected cabinet-door box: class 2 reads open 0.81 / closed 0.09, class 6 reads closed 0.998, and only 12 of 100 class-2 clips read closed |
| 50284349 | `pccomp` | 00:00:44 | 0.0122 | COMPLETED | all three composites: donor coverage 1.0000, 0 px changed outside the region, landmark residual 0.0 px |
| 50284466 | `pcgen` | 00:13:07 | 0.2185 | COMPLETED | 15 CogVideoX arms, 5 per case; LoRA verified absent before and after every generation |
| 50284467 | `pctrain` | 00:01:57 | 0.0325 | COMPLETED | the class-2/class-6 classifier, group-separated split, validation balanced accuracy 0.9849 on only 4 independent safe clips |
| 50284823 | `pcdiag` | 00:03:37 | 0.0603 | COMPLETED | **region masking measured nothing**: a stale view lookup skipped all 37 clips. The VAE control in the same job was valid, 8 of 8 clips kept their prediction |
| 50284824 | `pceval` | 00:01:10 | 0.0194 | COMPLETED | per-arm evaluation: the composite closes the panel (ncc 0.9997) while the classifier moves only 0.13 to 0.32 |
| 50285088 | `pcdiag` | 00:06:27 | 0.1075 | COMPLETED | region masking rerun after the fix: panel 0.131 against controls 0.006 to 0.019, correct sign in both classes |
| 50285379 | `pcprobe` | 00:00:03 | 0.0008 | FAILED | submitted without `CASE` exported; my error, not a code fault |
| 50286103 | `pcprobe` | 00:09:25 | 0.1569 | COMPLETED | gradient gate passes: finite, non-zero, 36.9 GiB peak. Its one-step score was NaN from a 50x-norm step, a fault in the probe and not in the gradient (P11) |
| 50286262 | `pcprobe` | 00:02:28 | 0.0411 | COMPLETED | step sweep replacing that single step: log p rises from -0.798 to -0.446 at 0.02 of the latent norm, no NaN anywhere |
| 50286287 | `pcguided` | 00:13:11 | 0.2197 | COMPLETED | case 1 guided pair: guided P(closed) 0.99989 flips, unguided control 0.28348 does not |
| 50286405 | `pceval` | 00:01:19 | 0.0219 | COMPLETED | rescored with the guidance arms: the guided output's panel is **still open**, ncc 0.109 against the closed reference |
| 50286464 | `pcguided` | 00:13:05 | 0.2181 | COMPLETED | case 2 guided pair: 0.99992 flips, control 0.36108 does not |
| 50286465 | `pcguided` | 00:18:12 | 0.3033 | COMPLETED | case 3 guided pair: 0.99990 flips, control 0.32542 does not |
| 50287094 | `pceval` | 00:01:24 | 0.0233 | COMPLETED | final scoring, 10 arms per case: guided panel open in all three |

**Three jobs that bought negative or void results, recorded rather than written off.**
50282082 at 0.1817 SU disproved the panel identification of DECISIONS P5 and stopped three
composites being built over a region already closed in the source. 50284823 at 0.0603 SU
produced an empty main section through a code bug (P10) and had to be rerun. 50285379 at
0.0008 SU failed on a missing environment variable. Together 0.2428 SU, 11% of what this
experiment spent.

**Nothing further is outstanding.** All 18 jobs have finished.

**Budget position, final.** 4.1380 SU had been spent before this experiment; **2.1686 SU**
was spent inside it across 18 jobs, so **6.3066 SU spent and 3.6934 remaining** of the
approved 10. Nothing is outstanding, so spent equals committed. The ceiling was never
approached: the highest the running total of spent plus outstanding worst case ever reached
was 9.900 SU, before the early stages reported their real elapsed times and the worst-case
figures were replaced by actuals.
