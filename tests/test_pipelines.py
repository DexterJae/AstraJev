"""End-to-end pipeline behaviour across all four task shapes."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from fake_world import TASKS, FakeExecutor, FakeOracle, FakeWorld

from astraxjev.backends import MockAstraBackend, MockJevBackend
from astraxjev.cost import CostLedger, PriceTable
from astraxjev.pipelines import (
    AstraJevPipeline,
    AstraOnlyPipeline,
    EpisodeContext,
    Geometry,
    PlanCache,
)
from astraxjev.senses import ProprioSenses

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
PRICES = PriceTable.from_yaml(ROOT / "config" / "pricing.yaml")
N = 8


def run(kind, task="pick_to_basket", n=N, **overrides):
    ledger = CostLedger(PRICES)
    cache = PlanCache(enabled=overrides.pop("plan_cache_enabled", True))
    outcomes = []
    geom = Geometry(**CFG["geometry"])
    m = CFG["mock"]

    for i in range(n):
        rng = np.random.default_rng(m["seed"] + i)
        world = FakeWorld(rng=rng, task=task)
        world.target_position = world.target_position + rng.normal(0, 0.03, 3)
        world.item_position = world.target_position.copy()
        world.neighbours = world._neighbours()
        oracle = FakeOracle(world)
        astra = MockAstraBackend(
            oracle,
            localize_sigma_m=m["astra_localize_sigma_m"],
            act_sigma_m=m["astra_act_sigma_m"],
            rng=rng,
        )
        jev = MockJevBackend(
            oracle,
            choice_accuracy=m["jev_choice_accuracy"],
            noul_noise=m["jev_noul_noise"],
            rng=rng,
        )
        if kind == "astra":
            a = {**CFG["astra_only"], **overrides}
            pipeline = AstraOnlyPipeline(
                astra, geom,
                global_every_n_ticks=a["global_every_n_ticks"],
                max_ticks=a["max_ticks"],
                plan_cache=cache, capabilities=oracle.capabilities(),
            )
        else:
            h = {**CFG["astra_jev"], **overrides}
            pipeline = AstraJevPipeline(
                astra, jev, geom,
                jev_precision_threshold_mm=h["jev_precision_threshold_mm"],
                handoff_prob=h["handoff_prob"],
                relocalize_prob=h["relocalize_prob"],
                max_align_resamples=h["max_align_resamples"],
                align_accept_score=h["align_accept_score"],
                obstacle_risk_escalate=h["obstacle_risk_escalate"],
                obstacle_attention_m=h["obstacle_attention_m"],
                max_relocalizes_per_subgoal=h["max_relocalizes_per_subgoal"],
                max_ticks=h["max_ticks"],
                plan_cache=cache, capabilities=oracle.capabilities(),
            )
        ctx = EpisodeContext(
            observe=lambda w=world, t=task: w.observation(TASKS[t]),
            senses=ProprioSenses(oracle),
            executor=FakeExecutor(world, max_env_steps=CFG["task"]["max_horizon"]),
            ledger=ledger, seed=i, episode=i,
        )
        outcomes.append(pipeline.run_episode(ctx))
    return ledger, outcomes, cache


def per_ep(ledger, outcomes):
    return ledger.total_usd() / len(outcomes)


# -- both pipelines solve every task shape ------------------------------------
@pytest.mark.parametrize("task", sorted(TASKS))
@pytest.mark.parametrize("kind", ["astra", "astra_jev"])
def test_pipeline_executes_the_generated_plan(task, kind):
    """The generality claim: one pipeline, four structurally different plans.

    If this needed a per-task branch anywhere in the pipelines, the routing
    would be a phase list in disguise.
    """
    _, outcomes, _ = run(kind, task)
    assert all(o.subgoals_total >= 3 for o in outcomes)
    progress = sum(
        o.subgoals_completed / o.subgoals_total for o in outcomes
    ) / len(outcomes)
    assert progress > 0.6, (task, kind, progress)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_hybrid_is_cheaper_on_every_task(task):
    base_l, base_o, _ = run("astra", task)
    hyb_l, hyb_o, _ = run("astra_jev", task)
    assert per_ep(hyb_l, hyb_o) < per_ep(base_l, base_o), task


@pytest.mark.parametrize("task", sorted(TASKS))
def test_hybrid_makes_fewer_frontier_calls(task):
    """The mechanism, stated as a test."""
    base_l, _, _ = run("astra", task)
    hyb_l, _, _ = run("astra_jev", task)
    assert (
        hyb_l.by("backend")["astra"]["calls"] < base_l.by("backend")["astra"]["calls"]
    ), task


def test_episodes_are_deterministic_given_a_seed():
    """Reproducibility is a precondition for reporting any of these numbers."""
    a, _, _ = run("astra_jev", n=4)
    b, _, _ = run("astra_jev", n=4)
    assert a.total_usd() == pytest.approx(b.total_usd())
    assert [r.tokens_in for r in a.records] == [r.tokens_in for r in b.records]


# -- the fusion does what it is supposed to do --------------------------------
def test_jev_is_a_rounding_error_in_the_hybrid_bill():
    """Where the saving comes from: avoided Astra calls, not cheap Jev calls.

    If Jev ever became a material share of the bill, the prompt or the tick
    count would be the thing to fix, not the routing.
    """
    led, _, _ = run("astra_jev")
    by_b = led.by("backend")
    assert by_b["jev"]["usd"] / led.total_usd() < 0.05


def test_every_jev_query_is_a_single_shared_read():
    led, _, _ = run("astra_jev")
    jev = [r for r in led.records if r.backend == "jev"]
    assert jev
    assert all(r.forward_passes == 1 for r in jev)
    assert all(r.tokens_out == 0 for r in jev)


def test_jev_batches_multiple_questions_into_one_call():
    """A per-question call would destroy the property being measured."""
    led, _, _ = run("astra_jev")
    ops = {r.op for r in led.records if r.backend == "jev"}
    assert any(op.count("+") >= 1 for op in ops), ops


def test_fine_motion_is_always_billed_to_astra():
    led, _, _ = run("astra_jev")
    actors = {r.backend for r in led.records if r.op == "act_fine"}
    assert actors == {"astra"}


def test_the_hybrid_needs_no_per_tick_perception():
    """Jev's state comes from free proprioception plus the last grounding.

    An invented per-tick detector would be the easiest way to make the hybrid
    look cheap on paper and fail in deployment.
    """
    led, _, _ = run("astra_jev")
    assert set(led.by("backend")) <= {"astra", "jev"}


def test_relocalization_is_bounded():
    """The gate needs a gate.

    A gripper stuck for some unrelated reason keeps reporting a stale belief;
    without a cap the pipeline buys frontier calls until the episode ends.
    """
    led, outcomes, _ = run("astra_jev", task="open_fridge")
    per_subgoal: dict[tuple, int] = {}
    for r in led.records:
        if r.op == "localize":
            key = (r.episode, r.subgoal)
            per_subgoal[key] = per_subgoal.get(key, 0) + 1
    cap = CFG["astra_jev"]["max_relocalizes_per_subgoal"]
    # One grounding can also be forced by a missing reference, so allow +1.
    assert per_subgoal and max(per_subgoal.values()) <= cap + 1


def test_both_pipelines_pay_for_planning():
    """Generating the decomposition is the price of generality, not a freebie."""
    for kind in ("astra", "astra_jev"):
        led, _, _ = run(kind)
        assert [r for r in led.records if r.op == "plan_task"], kind


def test_plan_is_cached_across_episodes_of_a_task():
    led, outcomes, cache = run("astra_jev", n=5)
    plans = [r for r in led.records if r.op == "plan_task"]
    assert len(plans) == 1
    assert cache.hits == 4 and cache.misses == 1


def test_disabling_the_plan_cache_costs_a_call_per_episode():
    led, outcomes, _ = run("astra_jev", n=4, plan_cache_enabled=False)
    assert len([r for r in led.records if r.op == "plan_task"]) == 4


# -- cost comparison and its limits -------------------------------------------
def test_cost_advantage_shrinks_when_astra_need_not_reground():
    """The limitation, stated as a test.

    With `global_every_n_ticks=0` Astra grounds once per episode. That is the
    cheapest the baseline can be, and any write-up quoting the headline saving
    has to quote this alongside it.
    """
    cheap_l, cheap_o, _ = run("astra", global_every_n_ticks=0)
    dear_l, dear_o, _ = run("astra", global_every_n_ticks=1)
    assert per_ep(cheap_l, cheap_o) < per_ep(dear_l, dear_o)


def test_every_model_call_is_recorded():
    """An unrecorded call is an understated cost."""
    led, outcomes, _ = run("astra_jev", n=1)
    claimed = outcomes[0].astra_calls + outcomes[0].jev_calls
    assert len(led.records) == claimed
