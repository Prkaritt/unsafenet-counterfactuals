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
