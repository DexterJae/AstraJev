# Astra + Jev: routing a frontier model's work to a cheap decision model

Two pipelines solve the **same** RoboBenchMart gripper task with the mobile
base locked — arm, torso lift and gripper only, no mobile manipulation. They
execute the same generated plan through the same low-level executor. The only
difference is **which model makes each decision, and how often the expensive
one runs**.

| | pipeline A (`astra`) | pipeline B (`astra_jev`) |
|---|---|---|
| plan the task | `gpt-6-astra` | `gpt-6-astra` |
| ground the references | `gpt-6-astra`, on a schedule | `gpt-6-astra`, **on demand** |
| coarse motion | `gpt-6-astra`, every tick | **`jev-1.14`** |
| fine motion | `gpt-6-astra` | `gpt-6-astra` |
| "re-ground now?" | answered by paying for a re-grounding | **answered by a Jev noul** |

Both models are real, hosted APIs, and that ratio is the whole point:

| | input | output |
|---|---|---|
| `gpt-6-astra` | $10 /Mtok | **$50 /Mtok** (cached input $1) |
| `jev-1.14` | $0.042 /Mtok | **free** |

**One Astra response costs about as much as two thousand Jev queries.** So the
bill tracks the number of Astra *responses*, and the hybrid is an attempt to
spend them only where nothing cheaper will do.

## Results

Toy world, 20 episodes per cell, mock capability model, published prices.
`astra/ep` is frontier calls per episode — the quantity that is the bill.

| task | pipeline | success | USD/episode | USD/success | astra/ep | jev/ep |
|---|---|---|---|---|---|---|
| pick_to_basket | `astra` | 0.600 | 9.57e-02 | 1.595e-01 | 9.0 | 0 |
| pick_to_basket | `astra_jev` | **0.750** | 4.01e-02 | **5.35e-02** | **4.1** | 9.4 |
| board_to_board | `astra` | 0.600 | 9.56e-02 | 1.594e-01 | 9.0 | 0 |
| board_to_board | `astra_jev` | **0.750** | 4.01e-02 | **5.35e-02** | **4.1** | 9.2 |
| pick_from_floor | `astra` | 0.750 | 1.01e-01 | 1.349e-01 | 9.7 | 0 |
| pick_from_floor | `astra_jev` | 0.750 | 4.30e-02 | **5.73e-02** | **4.5** | 11.7 |
| open_fridge | `astra` | **1.000** | 1.79e-01 | 1.789e-01 | 18.0 | 0 |
| open_fridge | `astra_jev` | 0.950 | 9.97e-02 | **1.050e-01** | **12.9** | 7.5 |

Saving in cost per successful episode: **−66%, −67%, −58%, −41%**.

Jev accounts for **under 0.5% of the hybrid's bill**. The entire saving is
avoided Astra calls — 9.0 per episode down to 4.1. That is worth saying
plainly: this is not a story about a clever cheap model, it is a story about
not calling the expensive one.

Three sensitivities, all from `scripts/sweep_pricing.py`:

* **Jev's price** could be 16× higher and the hybrid still wins at every point
  swept. (In the first version of this work, built against a much cheaper
  assumed frontier model, break-even was only 2.7× — the margin is wide now
  because `gpt-6-astra` output is $50/Mtok.)
* **Jev's size** likewise: no break-even inside the swept range.
* **The baseline's re-grounding cadence** is what sets its cost. Even against
  the cheapest possible baseline — ground once per episode, never again — the
  dearest hybrid configuration is still 17.8% cheaper per success.

## How the two are joined

They are not chained. There is no "Astra output → Jev input → Astra" pipeline.
A plan cursor routes each subgoal, and the two models share exactly two
datatypes: a `Frame` (where something is) and a `Waypoint` (where the gripper
should go). Whoever decides, the same executor runs it, so low-level control
quality is identical and the cost delta is attributable.

Jev → Astra carries only two signals: *call it now* and *that pose is good
enough*.

### The plan is data, not code

`astraxjev/plan.py` defines a small, deliberately task-agnostic vocabulary:

```
standoff(ref, distance)   contact(ref)        withdraw(distance)
over(ref, height)         arc(ref, angle)     set_gripper / hold
```

`gpt-6-astra` is asked once per instruction for an ordered list of subgoals
over that vocabulary, each annotated with a tolerance and whether it needs
current pixels. The plan is cached per instruction — the decomposition for
"pick X to basket" does not change between episodes, only the grounding does,
and at $0.027 a planning call paying it 30 times would be waste, not rigour.

### Tolerances are measured, not chosen

A language model proposing a plan cannot know the gripper stroke or the task's
success threshold, so whatever tolerance it suggests is a guess.
`plan.calibrate` overwrites it where the environment supplies the real number,
and records that it did (`tolerance_source`):

* **place** ← the task's own `TARGET_POS_THRESH`, which `PickToBasketContEnv`
  literally sets to 0.2 m. The first version of this code guessed 0.06.
* **grasp** ← whichever binds first, the finger stroke against the item width
  or half the gap to the neighbouring item. On a dense retail board the
  neighbour binds, at 10 mm.
* **withdraw** ← half the neighbour gap.

### The routing rule

Three conditions, all read off the generated plan, none of which know what
task is being solved:

```python
jev_owns = (tolerance_mm >= threshold      # coarse enough
            and not needs_pixels            # pose follows from stored coords
            and not path_constrained)       # motion is free in space
```

The third condition was **not** in the original design. It was added after
running the fridge task: routing on precision alone sent the door swings to
the coarse controller, which chose among seven fixed directions, cut the chord
of the hinge arc, tore the grip off on the first step, and left the hybrid both
less reliable *and* 3.6× dearer than the baseline. Precision requirement is
not the only thing that decides who can drive a subgoal. Running a second,
structurally different task is what surfaced that; nothing about the shelf
tasks would have.

### Jev's three roles

Matching the three its model card describes:

1. **Coarse policy.** A `choice` over seven generic directions. Jev reads a
   centimetre-quantised text state (`astraxjev/verbalize.py`), so it
   *structurally cannot* drive a millimetre subgoal off that interface — the
   division of labour is enforced by the interface, not promised in prose.
2. **Gate on the expensive model.** A `noul`: "is the stored reference too
   stale to act on?" This is where the saving comes from. Astra alone must
   re-ground on a schedule, because *asking* the question costs the same
   frontier call as *answering* it. A second noul decides when to hand over to
   fine motion.
3. **Ordinal Q-function over actions.** A `score` rating Astra's proposed
   pose; below threshold it is rejected and Astra asked again, bounded by
   `max_align_resamples`.

All questions on a tick go in **one** call. Jev evaluates them against a
single shared read of the state, which is the property the cost argument rests
on, so `JevBackend.query` takes a batch by design.

The hybrid has **no per-tick perception of its own**. Jev's text is built from
forward kinematics (free on any real robot) plus the frames and obstacle list
from the last grounding call. The coarse controller dead-reckons — which is
precisely why "is my belief stale?" is the question worth paying for.

## Honest accounting

* **With `--backend api` the dollar figure is not an assumption.** Both
  providers return their own usage and cost; the ledger records theirs as
  authoritative and keeps the computed figure alongside, so a divergence is
  visible rather than silent. `cost_provenance` in every results JSON reports
  what fraction was measured.
* **Costs are computed from prompts actually built**, so a mock run reports
  the token counts a real run would.
* **Cached input is priced.** These pipelines re-send a long system prompt;
  ignoring the $1/Mtok cached rate would overstate both arms.
* **Both pipelines pay for planning.** Generating the decomposition is the
  price of generality, not a freebie.
* **Proprioception is free for both.** Forward kinematics and gripper width
  cost nothing on a real robot.
* **Lead with cost per *successful* episode.** A pipeline can always be made
  cheaper by failing sooner.
* **Both cost knobs are swept.** The baseline's re-grounding cadence and the
  hybrid's coarse step scale each set how expensive their pipeline is.

## What is real and what is a stand-in

Read this before using any number from here in a paper.

| | status |
|---|---|
| `gpt-6-astra` and `jev-1.14` prices | **published**, read 2026-11 |
| Jev's API, question primitives, single-shared-read property | **real**, `astraxjev/backends/jev_api.py` |
| RoboBenchMart task, success criterion, action layout, base locking | **real**, read from the benchmark source |
| prompt construction and token counting | **real** |
| the planner | **stand-in.** `synthesise_plan` keys off the instruction text to produce the plan a real `gpt-6-astra` call would be asked for. The biggest simulation left. Lives in the mock backend, never in the pipelines. |
| the capability model | **assumption.** "Astra localises semantically but coarsely; it corrects a pose to millimetres when looking at the thing." The second half is the most optimistic claim here and the one a real run is most likely to contradict. |
| success rates above | **toy world** (`tests/fake_world.py`), not benchmark results |
| the `--world sim` path | written against the benchmark source, **never executed** — macOS exposes no Vulkan device to SAPIEN and `mplib` has no macOS wheel. `scripts/check_sim_env.py` exists to make its first run on a Linux box debuggable. |
| fine motion under `--backend api` | a frontier VLM asked for a corrected pose. Whether a chat model can regress a grasp pose to millimetres is exactly what needs testing. `--local-policy-port` substitutes a real VLA policy instead. |

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# toy world, no simulator, no API key
.venv/bin/python scripts/run_eval.py --pipeline astra     --world fake -n 30 -o results/astra.json
.venv/bin/python scripts/run_eval.py --pipeline astra_jev --world fake -n 30 -o results/astra_jev.json
.venv/bin/python scripts/compare_cost.py results/astra.json results/astra_jev.json --markdown

# the generality check: same code, four task shapes
bash scripts/run_all_tasks.sh -n 20

# sensitivity: prices, model sizes, and both pipelines' cost knobs
.venv/bin/python scripts/sweep_pricing.py results/astra.json results/astra_jev.json --grid -n 30

.venv/bin/python -m pytest tests/ -q
```

### With the real models

```bash
export OPENROUTER_API_KEY=...        # covers jev-1.14 and gpt-6-astra
.venv/bin/python scripts/run_eval.py --pipeline astra_jev --world fake -n 5 \
    --backend api --vlm-model gpt-6-astra
```

This bills your account. Budget roughly **$0.10 per Astra-only episode** and
**$0.04 per hybrid episode**; 30 episodes of each is about $4. Jev's share is
under a cent. `--jev-provider typesafe` uses `api.typesafe.ai` instead of
OpenRouter.

### The sim path

**macOS cannot run it.** Two blockers, both verified: SAPIEN installs on arm64
but raises `failed to find a rendering device` (Apple GPUs expose Metal, not
Vulkan), and `mplib`, which the motion planner needs, ships only manylinux
x86_64 wheels — its source build fails at `libclang==11.0.1`. Use Linux with
an NVIDIA GPU.

```bash
bash scripts/setup_linux_gpu.sh             # vulkan, repo, venv, assets, scenes, preflight
bash scripts/run_sim_experiment.sh --smoke  # 2 episodes, verbose, with video
bash scripts/run_sim_experiment.sh          # 30 episodes each, then compare
bash scripts/run_all_tasks.sh --world sim   # generality matrix on the benchmark
```

`setup_linux_gpu.sh` is idempotent — re-running after a failure costs only the
failed stage. It generates the PickToBasket test scenes locally rather than
downloading `demo_envs`.

**Run `scripts/check_sim_env.py` first** (both scripts above do). It exercises
all 35 environment, planner, oracle and executor APIs the sim path touches,
because that path was written by reading the RoboBenchMart source and has
never been executed. One check asserts the base qpos never changes — no mobile
manipulation — against the real simulator. Another checks that every reference
a generated plan names can actually be grounded in the chosen environment,
which is the first thing to break on a task this was not developed against.

`PickToBasket` is the right task to lead with: "the robot is spawned in close
proximity to the shelf", so it is solvable without driving, and the executor
goes through RoboBenchMart's own `static_manipulation`, which holds the body
joints and writes zeros into the base action dims.

## Layout

```
astraxjev/
  plan.py         the subgoal vocabulary, plans as data, tolerance calibration
  types.py        action layout, waypoints, the coarse symbolic state
  cost.py         CostLedger, PriceTable, provider-reported cost, re-pricing
  tokens.py       text and image token counting
  prompts.py      planning / grounding / fine-motion prompts
  verbalize.py    coarse state -> the terse text Jev reads
  senses.py       the proprioception both pipelines get free
  oracle.py       privileged state, used only by the mock backends
  executor.py     shared waypoint executor (ManiSkill + protocol)
  backends/
    base.py       protocols, question and answer types
    jev_api.py    the real hosted Jev
    astra_api.py  a vision LLM in the Astra slot; a VLA policy for fine motion
    *_mock.py     simulated from privileged state, with real token counts
  pipelines/
    base.py       the plan cursor and the shared episode driver
    astra_only.py pipeline A, plus the plan cache both pipelines share
    astra_jev.py  pipeline B: the routing rule and Jev's three roles
scripts/
  run_eval.py            one pipeline (--world fake|sim, --backend mock|api|http)
  run_all_tasks.sh       the generality matrix
  compare_cost.py        the table a paper would quote
  sweep_pricing.py       price / size / cadence sensitivity
  check_sim_env.py       preflight for the sim path -- run this first
  setup_linux_gpu.sh     one-shot setup on a Linux GPU box
  run_sim_experiment.sh  both pipelines on the real sim, then compare
config/
  default.yaml    evaluation setup and controller parameters
  pricing.yaml    published rates, with provenance
tests/            fake_world.py (four task shapes) plus 80 tests
```

Note what `config/default.yaml` does **not** contain any more: standoff
distances, retract distances, basket offsets, torso lifts. Those are
properties of a task, and tasks are decomposed by the planner now.

## References

* RoboBenchMart — arXiv:2511.10276, <https://github.com/emb-ai/RoboBenchMart>
* GPT-6 Astra — model id `gpt-6-astra`, <https://openrouter.ai/openai/gpt-6-astra>
* Jev — `typesafe/jev-1.14`, <https://openrouter.ai/api/alpha/decisions>;
  "Can Jev be Your Q or Policy in Reinforcement Learning?", arXiv:2610.11692
