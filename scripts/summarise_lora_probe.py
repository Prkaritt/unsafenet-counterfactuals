#!/usr/bin/env python
"""Turn a LoRA probe/training Slurm log into a recorded measurement.

Reads the job's log and `sacct`, and writes a JSON record of what one training
step actually cost, so the training schedule is chosen from a measurement rather
than an estimate. Nothing here infers a number that is not in the log; a field
that cannot be read is written as null.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sacct(job_id: str) -> dict:
    out = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-n", "--format=JobID,JobName,Elapsed,State,AllocTRES%60"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not out:
        return {}
    f = out.split(None, 4)
    elapsed = f[2] if len(f) > 2 else None
    secs = None
    if elapsed and elapsed.count(":") == 2:
        h, m, s = elapsed.split(":")
        secs = int(h) * 3600 + int(m) * 60 + int(s)
    return {
        "job_id": f[0], "job_name": f[1], "elapsed": elapsed,
        "elapsed_s": secs, "state": f[3] if len(f) > 3 else None,
        "su_charged": round(secs / 3600.0, 4) if secs else None,
        "alloc_tres": f[4].strip() if len(f) > 4 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default="outputs/lora/probe_result.json")
    ap.add_argument("--label", default="memory and timing probe")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else next(
        iter(sorted((REPO / "logs").glob(f"*-{args.job_id}.out"))), None)
    text = log_path.read_text(errors="replace") if log_path and log_path.exists() else ""

    # tqdm writes "Steps: 100%|...| 12/12 [01:23<00:00,  6.95s/it]" or "1.23it/s".
    rate_s_per_step = None
    steps_done = None
    train_wall_s = None
    for m in re.finditer(
        r"Steps:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(\d+):(\d\d)<[^,]*,\s*([0-9.]+)(s/it|it/s)", text
    ):
        steps_done = int(m.group(1))
        mins, secs = int(m.group(3)), int(m.group(4))
        train_wall_s = mins * 60 + secs
        v = float(m.group(5))
        rate_s_per_step = v if m.group(6) == "s/it" else (1.0 / v if v else None)

    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    record = {
        "label": args.label,
        "log": str(log_path) if log_path else None,
        "sacct": sacct(args.job_id),
        "gpu": grab(r"NVIDIA ([A-Za-z0-9 ]+),", str),
        "driver": grab(r"NVIDIA [A-Za-z0-9 ]+,\s*([0-9.]+),", str),
        "gpu_total_mib": grab(r"NVIDIA [^,]+,\s*[0-9.]+,\s*(\d+) MiB", int),
        "steps_completed": steps_done,
        "train_wall_s": train_wall_s,
        "seconds_per_step": round(rate_s_per_step, 3) if rate_s_per_step else None,
        "max_rss_kb": grab(r"Maximum resident set size \(kbytes\): (\d+)", int),
        "time_v_wall": grab(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)", str),
        "trainable_params": grab(r"trainable params[: ]+([0-9,]+)", str),
        "_note": (
            "seconds_per_step is tqdm's rate over the optimiser steps only. It excludes "
            "pipeline load and the one-off VAE pre-encode of the whole dataset, which the "
            "job pays once; budget a real run as load + pre-encode + steps x rate."
        ),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
