# Astra vs Astra + Jev on a RoboBenchMart gripper task

Two pipelines solve the **same** RoboBenchMart `PickToBasket` episode with the
mobile base locked, so the task is arm + torso-lift + gripper only — no mobile
manipulation. They share the phase machine, the geometry and the low-level
executor. The only thing that differs is **which model makes each decision and
how often an expensive one is invoked**, which is what makes the cost
difference attributable.

| | pipeline A | pipeline B |
|---|---|---|
| name | `astra` | `astra_jev` |
| localisation | Astra-Global | Astra-Global |
| coarse motion | Astra-Local every tick | **Jev** |
| precise grasp | Astra-Local | Astra-Local |
| "re-localise now?" | answered by paying for a re-localisation | **answered by a Jev noul** |

## Result, and the condition it depends on

Toy world, 30 episodes, mock backends, default configuration:

| metric | `astra` | `astra_jev` | delta |
|---|---|---|---|
| success rate | 0.967 | 0.967 | +0.0% |
| USD / episode | 6.040e-04 | 3.698e-04 | **-38.8%** |
| USD / successful episode | 6.248e-04 | 3.826e-04 | **-38.8%** |
| GFLOP / episode | 27,366 | 20,482 | **-25.2%** |
| Astra-Global calls / episode | 3.17 | **1.00** | -68.4% |
| Astra-Local calls / episode | 6.47 | 3.47 | -46.4% |
| Jev calls / episode | 0 | 15.27 | — |
| detector calls / episode | 0 | 13.20 | — |

**The headline number is not the result. This is:**

> On this task, adding Jev pays for itself only above **≈1.6 Astra-Global
> re-localisations per episode**. Below that rate, plain Astra is cheaper — the
> Jev and detector ticks cost more than the Astra-Global calls they avoid.

The baseline's cost is almost entirely set by how often it re-localises, and
the sweep makes that explicit:

| Astra-only cadence | Astra-Global calls/ep | USD / success |
|---|---|---|
| every tick | 11.07 | 1.902e-03 |
| every 2 ticks | 5.87 | 1.053e-03 |
| every 4 ticks (default) | 3.17 | 6.248e-04 |
| every 8 ticks | 1.63 | 3.784e-04 |
| once per episode | 1.00 | **2.768e-04** |

`astra_jev` sits at 3.73e-04–5.31e-04 depending on its own coarse-step knob. So
against an Astra baseline that localises **once** and never again, the hybrid
**loses**. The claim is real but conditional, and the condition is empirical:
*how often must Astra-Global re-localise under real occlusion and clutter?*
That is the number to measure on the real benchmark, and this repo is set up to
measure it.

Two further sensitivities, both reported by `scripts/sweep_pricing.py`:

* break-even Jev **price**: 0.134 USD/Mtok, i.e. 2.7x the assumed 0.05.
* break-even Jev **size**: 3.2 B params, i.e. 1.6x the assumed 2 B — because
  the hybrid replaces cheap 0.15 B Astra-Local ticks with larger Jev ticks, so
  it can be cheaper in dollars and dearer in FLOPs.

## How the two models are fused

Phase tolerances are declared in `astraxjev/types.py:PHASE_SPEC` and the
assignment is *derived* from them, not hard-coded:

| phase | tolerance | needs pixels | owner |
|---|---|---|---|
| LOCATE | 50 mm | yes | Astra-Global |
| APPROACH | 40 mm | no | **Jev** |
| ALIGN | 6 mm | yes | Astra-Local |
| GRASP | 6 mm | yes | Astra-Local |
| RETRACT | 25 mm | no | **Jev**, risk-gated |
| PLACE | 60 mm | no | **Jev** |
| SETTLE | — | no | free |

Jev plays three distinct roles, matching the three uses its paper describes:

1. **Coarse policy.** A `choice` question over seven coarse directions. Jev
   reads a centimetre-quantised text description (`astraxjev/verbalize.py`), so
   it *structurally cannot* drive a 6 mm grasp off that interface — the
   division of labour is enforced by the interface, not promised in prose.
2. **Gate on the expensive model.** A `noul` question, "is the stored target
   position too stale to grasp from?". This is where most of the saving comes
   from. Astra alone must re-localise on a cadence because *asking* the
   question costs the same MLLM forward pass as *answering* it; Jev answers it
   for roughly a tenth of the price, so Astra-Global becomes need-based rather
   than scheduled. A second noul, "is the gripper close enough to hand off?",
   triggers the APPROACH → ALIGN transition.
3. **Ordinal Q-function over actions.** A `score` question rating Astra-Local's
   proposed grasp pose. A proposal below threshold is rejected and Astra is
   asked again — bounded by `max_align_resamples`, because an unbounded
   accept/reject loop would spend more on Astra than it saves.

All questions on a tick go in **one** Jev call. That single-forward-pass
property is the whole basis of the cost argument, so `JevBackend.query` takes a
batch by design and `tests/test_pipelines.py` asserts that one call answers
several questions.

## Honest accounting

* **Costs are measured, not estimated.** Every prompt both pipelines send is
  really constructed (`astraxjev/prompts.py`, `astraxjev/verbalize.py`) and its
  tokens counted (`astraxjev/tokens.py`). Only the *unit prices* are
  assumptions, and each carries a `provenance` string in
  `config/pricing.yaml` that is copied into every results JSON.
* **Re-pricing needs no re-run.** The work done is fixed by the run, so
  `CostLedger.recompute` and `scripts/sweep_pricing.py` let a reader re-derive
  the dollar figure under their own assumptions.
* **The hybrid pays for its own perception.** Jev reads text, not pixels, so a
  cheap open-vocabulary detector is charged on every hybrid tick. Leaving it
  out would be the easiest way to overstate the advantage.
* **Proprioception is free for both.** Forward kinematics, gripper width and
  the pose of a basket bolted to the robot cost nothing on a real robot.
* **Lead with cost per *successful* episode.** A pipeline can always be made
  cheaper by failing sooner. `scripts/compare_cost.py` prints the per-success
  figure and flags any success-rate gap.
* **Both cost knobs are swept.** The baseline's `global_every_n_ticks` and the
  hybrid's `coarse_step_scale` each set how expensive their pipeline is;
  reporting one hand-picked value of either would be choosing the answer.

## What is real and what is a stand-in

Read this before using any number from here in a paper.

| | status |
|---|---|
| RoboBenchMart task, success criterion, action layout, base locking | **real**, read from the benchmark source |
| prompt construction and token counting | **real** |
| cost ledger and the comparison | **real**, given the price assumptions |
| Astra weights | **not public** (arXiv:2506.06205, in-house deployment) → `MockAstraBackend`, or plug an endpoint into `HttpAstraBackend` |
| Jev weights | **closed model** (arXiv:2610.11692) → `MockJevBackend`, or `HttpJevBackend` |
| success rates in the table above | **toy world** (`tests/fake_world.py`), not a benchmark result |
| the `--world sim` code path | written against the benchmark source, **never executed** — macOS has no Vulkan device and no `mplib` wheel. `scripts/check_sim_env.py` exists to make its first run on a Linux box debuggable. |

Two adaptations this project makes, both of which belong in any write-up:

1. **Astra-Local is used in end-effector space.** It was published as a *local
   path planner for a mobile base*; here it is the local planner for the
   gripper. Astra is a navigation system, not a manipulation policy, and
   nothing in arXiv:2506.06205 claims millimetre grasp regression.
2. **The mock capability model is an assumption, not a measurement.**
   `astraxjev/backends/astra_mock.py` encodes "Astra-Global is semantically
   strong but spatially coarse, with error that shrinks as the gripper gets
   closer; Astra-Local is millimetre-precise and servos on current pixels".
   That division is what makes the fusion sensible, and it is exactly what the
   real-backend runs have to confirm.

The cost half of the comparison transfers to real backends unchanged, because
it is computed from the prompts actually built. The success-rate half does not
and must be re-run with `--world sim --backend http`.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# toy world, no simulator needed
.venv/bin/python scripts/run_eval.py --pipeline astra     --world fake -n 30 -o results/astra.json
.venv/bin/python scripts/run_eval.py --pipeline astra_jev --world fake -n 30 -o results/astra_jev.json
.venv/bin/python scripts/compare_cost.py results/astra.json results/astra_jev.json --markdown

# sensitivity: prices, model sizes, and both pipelines' cost knobs
.venv/bin/python scripts/sweep_pricing.py results/astra.json results/astra_jev.json --grid -n 30

.venv/bin/python -m pytest tests/ -q
```

### The sim path

**macOS cannot run it.** Two independent blockers, both verified: SAPIEN
installs on arm64 but raises `failed to find a rendering device` (Apple GPUs
expose Metal, not Vulkan), and `mplib`, which the motion planner needs, ships
only manylinux x86_64 wheels — its source build fails at `libclang==11.0.1`.
Use a Linux box with an NVIDIA GPU.

```bash
bash scripts/setup_linux_gpu.sh            # vulkan, repo, venv, assets, scenes, preflight
bash scripts/run_sim_experiment.sh --smoke # 2 episodes, verbose, with video
bash scripts/run_sim_experiment.sh         # 30 episodes each, then compare
bash scripts/run_sim_experiment.sh --sweep # ...plus the parameter grid
```

`setup_linux_gpu.sh` is idempotent — re-running after a failure costs only the
stage that failed. It generates the PickToBasket test scenes locally rather
than downloading `demo_envs`, since the two relevant configs write into
`demo_envs/test_unseen_items_pick_to_basket` with `rewrite: true`.

**Run `scripts/check_sim_env.py` first** (both scripts above do). It exercises
every environment, planner and executor API the sim path touches — 33 named
checks — because **the sim path was written by reading the RoboBenchMart source
and has never been executed**: it could not be, on the machine where it was
written. A version mismatch therefore surfaces in seconds with the attribute
named, rather than thirty minutes into a run. One of its checks asserts the
base qpos never changes, i.e. no mobile manipulation, against the real
simulator.

With real model endpoints:

```bash
ASTRA_URL=http://localhost:9001 JEV_URL=http://localhost:9002 \
    bash scripts/run_sim_experiment.sh

# or directly
python scripts/run_eval.py --pipeline astra_jev --world sim -n 30 \
    --env-id PickToBasketContNiveaEnv \
    --scene-dir demo_envs/test_unseen_items_pick_to_basket \
    --backend http --astra-url http://localhost:9001 --jev-url http://localhost:9002
```

`PickToBasket` is the right task for this: "the robot is spawned in close
proximity to the shelf", so it is solvable without driving. On top of that, the
executor goes through RoboBenchMart's own `static_manipulation`, which holds the
body joints and writes zeros into the base action dims — so the base provably
never moves. `PickToBasketContNestleEnv` is the out-of-distribution target item
if you want the zero-shot generalisation setting as well.

## Layout

```
astraxjev/
  types.py        phases, tolerances, action layout, waypoints
  cost.py         CostLedger, PriceTable, per-success summaries, re-pricing
  tokens.py       text and image token counting
  prompts.py      Astra-Global / Astra-Local / detector prompt construction
  verbalize.py    coarse state -> the terse text Jev reads
  geometry.py     standoff, grasp, retract and place poses
  senses.py       the proprioception both pipelines get free
  oracle.py       privileged state, used only by the mock backends
  executor.py     shared waypoint executor (ManiSkill + protocol)
  backends/       Astra and Jev: protocols, mocks, HTTP adapters
  pipelines/      base phase machine, astra_only.py, astra_jev.py
scripts/
  run_eval.py            run one pipeline (--world fake | sim)
  compare_cost.py        the table a paper would quote
  sweep_pricing.py       price / size / parameter-grid sensitivity
  check_sim_env.py       preflight for the sim path -- run this first
  setup_linux_gpu.sh     one-shot setup on a Linux GPU box
  run_sim_experiment.sh  both pipelines on the real sim, then compare
config/           default.yaml (task + pipeline knobs), pricing.yaml (assumptions)
tests/            fake_world.py plus the integrity tests
```

## References

* RoboBenchMart — arXiv:2511.10276, <https://github.com/emb-ai/RoboBenchMart>
* Astra — arXiv:2506.06205, <https://astra-mobility.github.io/>
* Jev as a Q-function or policy — arXiv:2610.11692
