#!/bin/bash
# Status of every job this experiment has submitted, from sacct and squeue.
#
# Job ids are read from outputs/panel_cover/JOBS.tsv, which each submission appends to, so
# this survives a closed session: the record is on disk, not in a chat transcript.
#
# Usage: bash scripts/pc_jobs.sh
set -uo pipefail
REPO="/ocean/projects/cis260288p/puprety/unsafenet-counterfactuals"
LEDGER="${REPO}/outputs/panel_cover/JOBS.tsv"

if [[ ! -f "${LEDGER}" ]]; then
  echo "no job ledger at ${LEDGER}"
  exit 1
fi

IDS=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ {printf "%s%s", sep, $1; sep=","}' "${LEDGER}")
if [[ -z "${IDS}" ]]; then
  echo "job ledger has no job ids yet"
  exit 0
fi

echo "=== ledger ==="
column -t -s $'\t' "${LEDGER}"

echo
echo "=== sacct (elapsed is what is charged: 1 SU per L40S GPU-hour) ==="
sacct -X -j "${IDS}" \
  --format=JobID,JobName%14,State%12,Start%20,Elapsed,Timelimit,ExitCode

echo
echo "=== still in the queue ==="
squeue -j "${IDS}" -o "%.10i %.14j %.9T %.8M %.20S %R" 2>/dev/null \
  || echo "none of these jobs are queued or running"

echo
echo "=== SU charged per job, computed from elapsed ==="
sacct -X -n -j "${IDS}" --format=JobID,JobName,Elapsed,State -P | \
awk -F'|' '{
  split($3, t, ":");
  n = length(t);
  secs = 0;
  if (n == 3) { split(t[1], d, "-"); h = (length(d) == 2) ? d[2] : t[1];
                days = (length(d) == 2) ? d[1] : 0;
                secs = days*86400 + h*3600 + t[2]*60 + t[3]; }
  else if (n == 2) { secs = t[1]*60 + t[2]; }
  su = secs / 3600.0;
  total += su;
  printf "%-12s %-14s %-10s %8.4f SU  %s\n", $1, $2, $3, su, $4;
}
END { printf "%-48s %8.4f SU\n", "TOTAL for this experiment", total; }'
