#!/usr/bin/env bash
# One-shot setup for a fresh Linux + NVIDIA GPU box (RunPod / Lambda / vast.ai).
#
# Idempotent: every stage checks whether it is already done and skips if so, so
# re-running after a failure costs only the stage that failed.
#
#   bash scripts/setup_linux_gpu.sh
#
# Env overrides:
#   PY=python3.10            interpreter to build the venv from
#   SKIP_APT=1               you have no sudo (Vulkan must already work)
#   SKIP_ASSETS=1            assets already downloaded
#   RBM_DIR=/path/to/repo    existing RoboBenchMart checkout to reuse
#
# Disk: ~25 GB (torch ~3 GB, RoboCasa assets ~15 GB, RBM assets ~3 GB).
# Time: 20-40 min, dominated by asset downloads.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3.10}"
RBM_DIR="${RBM_DIR:-$ROOT/RoboBenchMart}"
VENV="$ROOT/.venv-sim"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '   \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[31mFAILED\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. sanity
say "0/7  environment"
[[ "$(uname -s)" == "Linux" ]] || die "this script is for Linux; SAPIEN has no
  working Vulkan device on macOS and mplib has no macOS wheel. See README."
uname -m | grep -q x86_64 || warn "not x86_64: mplib only ships manylinux
  x86_64 wheels, so it will try to build from source and probably fail."
command -v "$PY" >/dev/null || die "$PY not found. ManiSkill wants 3.9-3.12;
  set PY=python3.11 (or similar) if your box has a different one."
ok "$($PY -V), $(uname -m)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || warn "no nvidia-smi: a GPU is not strictly required (sim-backend=cpu
  works) but rendering still needs a Vulkan device."

# ------------------------------------------------------------- 1. vulkan
say "1/7  Vulkan"
if [[ -z "${SKIP_APT:-}" ]]; then
  if command -v sudo >/dev/null && [[ $(id -u) -eq 0 || -n "${FORCE_APT:-}" ]] \
     || [[ $(id -u) -eq 0 ]]; then
    apt-get update -qq || sudo apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      libvulkan1 vulkan-tools libglvnd0 libgl1 libegl1 git git-lfs \
      || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
         libvulkan1 vulkan-tools libglvnd0 libgl1 libegl1 git git-lfs
  else
    warn "no root/sudo; skipping apt. Set SKIP_APT=1 to silence this."
  fi
fi
if command -v vulkaninfo >/dev/null; then
  if vulkaninfo --summary >/dev/null 2>&1; then
    ok "vulkaninfo reports a device"
  else
    warn "vulkaninfo failed. On headless cloud boxes the usual fix is
  NVIDIA's ICD file: ls /usr/share/vulkan/icd.d/  (expect nvidia_icd.json).
  If it is missing, reinstall the NVIDIA driver with --no-opengl-files omitted."
  fi
else
  warn "vulkaninfo not installed; continuing (check_sim_env.py will catch it)"
fi

# --------------------------------------------------------- 2. benchmark repo
say "2/7  RoboBenchMart checkout"
if [[ -d "$RBM_DIR/.git" || -d "$RBM_DIR/dsynth" ]]; then
  ok "reusing $RBM_DIR"
else
  git clone --depth 1 https://github.com/emb-ai/RoboBenchMart "$RBM_DIR"
  ok "cloned to $RBM_DIR"
fi

# ------------------------------------------------------------- 3. venv
say "3/7  python environment"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY" -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
"$PIP" install -q --upgrade pip wheel setuptools
ok "$("$VENV/bin/python" -V) at $VENV"

# -------------------------------------------------------- 4. dependencies
say "4/7  dependencies (this is the slow one)"
"$PIP" install -q -r "$ROOT/requirements.txt"
"$PIP" install -q -r "$RBM_DIR/requirements.txt"
# mplib is pinned by the benchmark and is what the motion planner needs.
"$PIP" install -q "mplib==0.2.1" \
  || die "mplib install failed. It only ships manylinux x86_64 wheels.
  On x86_64 check your glibc (needs >= 2.17); on arm64 there is no wheel."
# Optional but worth it: exact BPE token counts instead of the estimator.
"$PIP" install -q tiktoken pillow || warn "tiktoken/pillow optional install failed"
ok "installed"

"$VENV/bin/python" -c "
import sapien, mani_skill, mplib, torch
print('   sapien', sapien.__version__, '| mani_skill', mani_skill.__version__,
      '| mplib', mplib.__version__, '| torch', torch.__version__)
" || die "imports failed after install"

say "4b/7  ManiSkill smoke test"
( cd "$RBM_DIR" && "$VENV/bin/python" -m mani_skill.examples.demo_random_action >/dev/null 2>&1 ) \
  && ok "demo_random_action ran" \
  || warn "demo_random_action failed -- almost always Vulkan. Fix that before
  going further; check_sim_env.py will report it precisely."

# ------------------------------------------------------------- 5. assets
say "5/7  assets"
if [[ -n "${SKIP_ASSETS:-}" ]]; then
  ok "skipped (SKIP_ASSETS set)"
else
  cd "$RBM_DIR"
  if [[ -d "$HOME/.maniskill/data/scene_datasets/robocasa_dataset" ]]; then
    ok "RoboCasa assets already present"
  else
    "$VENV/bin/python" -m mani_skill.utils.download_asset -y RoboCasa \
      || die "RoboCasa download failed"
  fi
  if [[ -d "$RBM_DIR/assets" ]] && [[ -n "$(ls -A "$RBM_DIR/assets" 2>/dev/null)" ]]; then
    ok "RoboBenchMart assets already present"
  else
    "$VENV/bin/hf" download emb-ai/RoboBenchMart_assets \
      --repo-type dataset --local-dir assets \
      || die "RoboBenchMart_assets download failed"
  fi
  cd "$ROOT"
fi

# ------------------------------------------------------- 6. test scenes
say "6/7  test scenes"
SCENES="$RBM_DIR/demo_envs/test_unseen_items_pick_to_basket"
if [[ -d "$SCENES" ]] && [[ -n "$(ls -A "$SCENES" 2>/dev/null)" ]]; then
  ok "already generated: $SCENES"
else
  # Generated locally rather than downloaded: these two configs write into
  # demo_envs/test_unseen_items_pick_to_basket and both set rewrite: true, so
  # the HF demo_envs download is not needed for the PickToBasket evaluation.
  cd "$RBM_DIR"
  for cfg in test_unseen_items_pick_to_basket_1 test_unseen_items_pick_to_basket_2; do
    "$VENV/bin/python" scripts/generate_scene_continuous.py "ds_continuous=$cfg" \
      || die "scene generation failed for $cfg"
  done
  cd "$ROOT"
  ok "generated $SCENES"
fi

# --------------------------------------------------------- 7. preflight
say "7/7  preflight"
"$VENV/bin/python" scripts/check_sim_env.py \
  --env-id PickToBasketContNiveaEnv \
  --scene-dir demo_envs/test_unseen_items_pick_to_basket \
  || die "preflight failed -- see the named checks above. Nothing below this
  point will work until they pass."

say "done"
cat <<EOF
   venv:    $VENV
   scenes:  $SCENES

   next:    bash scripts/run_sim_experiment.sh --smoke     # 2 episodes first
            bash scripts/run_sim_experiment.sh
            bash scripts/run_all_tasks.sh --world sim      # generality matrix

   real models (bills your account):
            export OPENROUTER_API_KEY=...
            bash scripts/run_sim_experiment.sh --smoke
EOF
