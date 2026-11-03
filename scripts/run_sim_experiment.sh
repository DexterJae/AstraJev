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
#   OPENROUTER_API_KEY=...  switches to --backend api: real jev-1.14 and a real
#                           vision LLM standing in for Astra. Set this and the
#                           dollar figures become provider-reported rather
#                           than computed.
#   VLM_MODEL=gpt-6-astra   which model fills the Astra slot
#   LOCAL_POLICY_PORT=8000  a RoboBenchMart policy server (octo/pi0) to use for
#                           fine motion instead of the mock
#
# Without an API key the run uses --backend mock. The *costs* are still
# computed from the prompts actually built, but the success rates are a
# property of the mock capability model and must not be reported as benchmark
# numbers.

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
BACKEND_NOTE="mock backends: token counts are real, success rates are NOT benchmark results"
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  BACKEND_ARGS=(--backend api --vlm-model "${VLM_MODEL:-gpt-6-astra}")
  [[ -n "${LOCAL_POLICY_PORT:-}" ]] &&     BACKEND_ARGS+=(--local-policy-port "$LOCAL_POLICY_PORT")
  BACKEND_NOTE="real APIs: jev-1.14 + ${VLM_MODEL:-gpt-6-astra}"
  if [[ -z "${LOCAL_POLICY_PORT:-}" ]]; then
    BACKEND_NOTE="$BACKEND_NOTE (fine motion still mocked; set LOCAL_POLICY_PORT)"
  fi
  echo "   NOTE: a real run bills your account. gpt-6-astra is \$10/\$50 per"
  echo "         million input/output tokens and this experiment makes a few"
  echo "         frontier calls per episode -- budget roughly \$0.10 per"
  echo "         Astra-only episode and \$0.04 per hybrid episode."
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
