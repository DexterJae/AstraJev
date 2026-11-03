#!/usr/bin/env bash
# Run both pipelines across every task shape and print the comparison matrix.
#
# This is the generality check. The pipelines contain no per-task code: the
# decomposition comes from the planner and the routing reads only each
# subgoal's tolerance, pixel-dependence and path constraint. If a task needed
# a code change to work, that claim would be false.
#
#   bash scripts/run_all_tasks.sh              # toy world, 20 episodes each
#   bash scripts/run_all_tasks.sh -n 50
#   bash scripts/run_all_tasks.sh --world sim  # the real benchmark

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3
N=20
WORLD=fake
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N="$2"; shift 2 ;;
    --world) WORLD="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

TASKS=(pick_to_basket board_to_board pick_from_floor open_fridge)
mkdir -p results

printf '\n%-17s %-10s %8s %9s %13s %13s %8s %7s\n' \
  task pipeline success progress 'usd/episode' 'usd/success' astra/ep jev/ep
printf '%s\n' "$(printf '=%.0s' {1..92})"

for t in "${TASKS[@]}"; do
  for pl in astra astra_jev; do
    out="results/${WORLD}_${t}_${pl}.json"
    "$PY" scripts/run_eval.py --pipeline "$pl" --world "$WORLD" \
      --fake-task "$t" -n "$N" --quiet -o "$out" ${EXTRA[@]+"${EXTRA[@]}"} >/dev/null
    "$PY" - "$out" "$t" "$pl" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
n = max(1, d["episodes"])
astra = d["by_backend"].get("astra", {}).get("calls", 0) / n
jev = d["by_backend"].get("jev", {}).get("calls", 0) / n
print(f'{sys.argv[2]:<17} {sys.argv[3]:<10} {d["success_rate"]:8.3f} '
      f'{d["mean_plan_progress"]:9.3f} {d["usd_per_episode"]:13.4e} '
      f'{d["usd_per_success"]:13.4e} {astra:8.1f} {jev:7.1f}')
PYEOF
  done
done

echo
"$PY" - "$WORLD" "${TASKS[@]}" <<'PYEOF'
import json, sys
world, tasks = sys.argv[1], sys.argv[2:]
print("saving in cost per successful episode, by task")
print("-" * 56)
for t in tasks:
    try:
        a = json.load(open(f"results/{world}_{t}_astra.json"))
        b = json.load(open(f"results/{world}_{t}_astra_jev.json"))
    except FileNotFoundError:
        continue
    ca, cb = a["usd_per_success"], b["usd_per_success"]
    delta = (cb - ca) / ca * 100 if ca and ca == ca else float("nan")
    note = ""
    if b["success_rate"] < a["success_rate"] - 0.02:
        note = f"  (hybrid {a['success_rate'] - b['success_rate']:.0%} less reliable)"
    print(f"  {t:<17} {delta:+7.1f}%{note}")
print()
print("The pipelines contain no per-task code. Every row above ran the same")
print("routing rule over a plan the planner generated for that instruction.")
PYEOF
