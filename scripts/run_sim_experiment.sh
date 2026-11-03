#!/usr/bin/env bash
# Run both pipelines on the real RoboBenchMart PickToBasket task and print the
# comparison. Assumes scripts/setup_linux_gpu.sh has completed.
#
#   bash scripts/run_sim_experiment.sh                # 30 episodes each
#   bash scripts/run_sim_experiment.sh --smoke        # 2 episodes, verbose, video
#   bash scripts/run_sim_experiment.sh -n 50 --video
#   bash scripts/run_sim_experiment.sh --sweep        # also re-run the parameter grid
#
# Env overrides:
#   VENV=...         python environment (default .venv-sim)
#   ENV_ID=...       default PickToBasketContNiveaEnv (train item)
#                    PickToBasketContNestleEnv for the OOD / zero-shot item
#   SCENE_DIR=...    default demo_envs/test_unseen_items_pick_to_basket
#   ASTRA_URL=...    real Astra endpoint; without it the mock backend is used
#   JEV_URL=...      real Jev endpoint
#
# With ASTRA_URL/JEV_URL set the run uses --backend http and the success rates
# are a benchmark result. Without them it uses --backend mock: the *costs* are
# still exact, because they are computed from the prompts actually built, but
# the success rates are a property of the mock capability model and must not be
# reported as benchmark numbers.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv-sim}"
PY="$VENV/bin/python"
ENV_ID="${ENV_ID:-PickToBasketContNiveaEnv}"
SCENE_DIR="${SCENE_DIR:-demo_envs/test_unseen_items_pick_to_basket}"
RBM_DIR="${RBM_DIR:-$ROOT/RoboBenchMart}"

N=30
VIDEO=""
SWEEP=""
SMOKE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)  SMOKE=1; N=2; VIDEO="--save-video"; shift ;;
    --video)  VIDEO="--save-video"; shift ;;
    --sweep)  SWEEP=1; shift ;;
    -n)       N="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFAILED\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "$PY" ]] || die "no python at $PY -- run scripts/setup_linux_gpu.sh first"
[[ -d "$RBM_DIR/dsynth" ]] || die "no benchmark checkout at $RBM_DIR"

BACKEND_ARGS=(--backend mock)
BACKEND_NOTE="mock backends: costs are exact, success rates are NOT benchmark results"
if [[ -n "${ASTRA_URL:-}" ]]; then
  BACKEND_ARGS=(--backend http --astra-url "$ASTRA_URL")
  [[ -n "${JEV_URL:-}" ]] && BACKEND_ARGS+=(--jev-url "$JEV_URL")
  BACKEND_NOTE="real endpoints: astra=$ASTRA_URL jev=${JEV_URL:-<unset>}"
fi

# ----------------------------------------------------------------- preflight
say "preflight"
"$PY" scripts/check_sim_env.py --env-id "$ENV_ID" --scene-dir "$SCENE_DIR" \
  || die "preflight failed; fix the named checks before running episodes"

mkdir -p results

COMMON=(--world sim --env-id "$ENV_ID" --scene-dir "$SCENE_DIR" -n "$N")
[[ -n "$VIDEO" ]] && COMMON+=("$VIDEO")
[[ -z "$SMOKE" ]] && COMMON+=(--quiet)

# ------------------------------------------------------------------- run A
say "pipeline A: astra only  ($N episodes, $ENV_ID)"
echo "   $BACKEND_NOTE"
"$PY" scripts/run_eval.py --pipeline astra "${COMMON[@]}" "${BACKEND_ARGS[@]}" \
  -o results/sim_astra.json || die "astra pipeline run failed"

# ------------------------------------------------------------------- run B
say "pipeline B: astra + jev  ($N episodes, $ENV_ID)"
"$PY" scripts/run_eval.py --pipeline astra_jev "${COMMON[@]}" "${BACKEND_ARGS[@]}" \
  -o results/sim_astra_jev.json || die "astra_jev pipeline run failed"

# ---------------------------------------------------------------- compare
say "comparison"
"$PY" scripts/compare_cost.py results/sim_astra.json results/sim_astra_jev.json \
  --markdown | tee results/sim_comparison.txt

# ------------------------------------------------------------------ sweep
say "sensitivity"
SWEEP_ARGS=(results/sim_astra.json results/sim_astra_jev.json)
if [[ -n "$SWEEP" ]]; then
  # Re-runs both pipelines across their cost knobs. Costs ~8x a single run.
  SWEEP_ARGS+=(--grid --world sim -n "$N")
fi
"$PY" scripts/sweep_pricing.py "${SWEEP_ARGS[@]}" | tee results/sim_sensitivity.txt

say "done"
cat <<EOF
   results/sim_astra.json
   results/sim_astra_jev.json
   results/sim_comparison.txt
   results/sim_sensitivity.txt
EOF
[[ -n "$VIDEO" ]] && echo "   results/videos/{astra,astra_jev}/"
if [[ -z "$SWEEP" ]]; then
  cat <<'EOF'

   The headline percentage is not the result on its own. Re-run with --sweep to
   get the break-even re-localisation rate, which is the number the claim
   actually turns on. See README.md.
EOF
fi
