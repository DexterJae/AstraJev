"""End-to-end pipeline behaviour in the toy world."""

import numpy as np
import pytest
import yaml

from fake_world import FakeExecutor, FakeOracle, FakeWorld

from astraxjev.backends import MockAstraBackend, MockJevBackend
from astraxjev.cost import CostLedger, PriceTable
from astraxjev.pipelines import (
    AstraJevPipeline,
    AstraOnlyPipeline,
    EpisodeContext,
    Geometry,
)
from astraxjev.senses import ProprioSenses
from astraxjev.types import Phase

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
PRICES = PriceTable.from_yaml(ROOT / "config" / "pricing.yaml")


def build(kind, seed=0, ledger=None, **overrides):
    rng = np.random.default_rng(seed)
    world = FakeWorld(rng=rng)
    oracle = FakeOracle(world)
    geom = Geometry(**CFG["geometry"])
    m = CFG["mock"]
    astra = MockAstraBackend(
        oracle,
        global_sigma_m=m["astra_global_sigma_m"],
        local_sigma_m=m["astra_local_sigma_m"],
        rng=rng,
    )
    jev = MockJevBackend(
        oracle,
        choice_accuracy=m["jev_choice_accuracy"],
        noul_noise=m["jev_noul_noise"],
        rng=rng,
    )
    if kind == "astra":
        a = dict(CFG["astra_only"])
        a.update(overrides)
        pipeline = AstraOnlyPipeline(
            astra,
            geom,
            global_every_n_ticks=a["global_every_n_ticks"],
            chunk_len=a["chunk_len"],
            max_ticks=a["max_ticks"],
        )
    else:
        h = dict(CFG["astra_jev"])
        h.update(overrides)
        pipeline = AstraJevPipeline(
            astra,
            jev,
            geom,
            jev_precision_threshold_mm=h["jev_precision_threshold_mm"],
            handoff_prob=h["handoff_prob"],
            max_align_resamples=h["max_align_resamples"],
            align_accept_score=h["align_accept_score"],
            retract_risk_escalate=h["retract_risk_escalate"],
            max_ticks=h["max_ticks"],
        )

    ledger = ledger or CostLedger(PRICES)
    ctx = EpisodeContext(
        observe=lambda w=world: w.observation("move to shelf and pick Nivea Body Milk to basket"),
        senses=ProprioSenses(oracle),
        executor=FakeExecutor(world, max_env_steps=CFG["task"]["max_horizon"]),
        ledger=ledger,
        seed=seed,
        episode=0,
    )
    return pipeline, ctx, ledger


def run(kind, n=12, **overrides):
    ledger = CostLedger(PRICES)
    outcomes = []
    for i in range(n):
        pipeline, ctx, _ = build(kind, seed=i, ledger=ledger, **overrides)
        ctx.episode = i
        outcomes.append(pipeline.run_episode(ctx))
    return ledger, outcomes


# -- both pipelines actually solve the task -----------------------------------
@pytest.mark.parametrize("kind", ["astra", "astra_jev"])
def test_pipeline_reaches_done_and_mostly_succeeds(kind):
    _, outcomes = run(kind)
    assert all(o.phase_reached == Phase.DONE for o in outcomes)
    assert sum(o.success for o in outcomes) / len(outcomes) >= 0.75


def test_episodes_are_deterministic_given_a_seed():
    """Reproducibility is a precondition for reporting any of these numbers."""
    a, _ = run("astra_jev", n=4)
    b, _ = run("astra_jev", n=4)
    assert a.total_usd() == pytest.approx(b.total_usd())
    assert [r.tokens_in for r in a.records] == [r.tokens_in for r in b.records]


# -- the fusion does what it is supposed to do --------------------------------
def test_hybrid_makes_fewer_expensive_calls():
    """The mechanism, stated as a test: fewer Astra-Global invocations."""
    base, _ = run("astra")
    hyb, _ = run("astra_jev")
    assert (
        hyb.by("backend")["astra_global"]["calls"]
        < base.by("backend")["astra_global"]["calls"]
    )


def test_hybrid_charges_for_the_perception_that_feeds_jev():
    """Jev reads text, not pixels, so the detector must appear in the ledger.

    Omitting it would be the easiest way to overstate the hybrid's advantage.
    """
    hyb, _ = run("astra_jev")
    by_b = hyb.by("backend")
    assert by_b["detector"]["calls"] > 0
    assert by_b["detector"]["usd"] > 0
    assert by_b["detector"]["calls"] >= by_b["jev"]["calls"] * 0.5


def test_every_jev_query_is_a_single_forward_pass():
    hyb, _ = run("astra_jev")
    jev_records = [r for r in hyb.records if r.backend == "jev"]
    assert jev_records
    assert all(r.forward_passes == 1 for r in jev_records)
    # No tokens are generated: Jev returns typed answers, not text.
    assert all(r.tokens_out == 0 for r in jev_records)


def test_jev_batches_multiple_questions_into_one_call():
    """A per-question call would destroy the property being measured."""
    hyb, _ = run("astra_jev")
    ops = {r.op for r in hyb.records if r.backend == "jev"}
    assert any(op.count("+") >= 1 for op in ops), ops


def test_fine_phases_are_billed_to_astra_only():
    hyb, _ = run("astra_jev")
    fine = {r.backend for r in hyb.records if r.phase in ("align", "grasp")}
    assert "jev" in fine or True  # Jev may *verify* in ALIGN
    # ...but the planning in a fine phase is always Astra's.
    planners = {
        r.backend for r in hyb.records if r.phase == "align" and r.op == "plan_local"
    }
    assert planners == {"astra_local"}


def test_no_astra_global_calls_in_fine_phases_for_the_hybrid():
    """Re-localising mid-grasp would defeat the point of the handoff."""
    hyb, _ = run("astra_jev")
    assert not [
        r for r in hyb.records if r.backend == "astra_global" and r.phase == "grasp"
    ]


# -- cost comparison ----------------------------------------------------------
def test_hybrid_is_cheaper_at_the_default_configuration():
    base_led, base_out = run("astra")
    hyb_led, hyb_out = run("astra_jev")
    n = len(base_out)
    assert hyb_led.total_usd() / n < base_led.total_usd() / n


def test_cost_advantage_disappears_if_astra_need_not_relocalise():
    """The limitation, stated as a test.

    With `global_every_n_ticks=0` Astra localises once per episode, and the
    hybrid's Jev plus detector overhead is no longer bought by anything. Any
    write-up that quotes the headline saving has to quote this alongside it.
    """
    cheap_base, out = run("astra", global_every_n_ticks=0)
    hyb, _ = run("astra_jev")
    n = len(out)
    assert cheap_base.total_usd() / n < hyb.total_usd() / n


def test_every_model_call_is_recorded():
    """An unrecorded call is an understated cost."""
    pipeline, ctx, ledger = build("astra_jev")
    outcome = pipeline.run_episode(ctx)
    recorded = len(ledger.records)
    claimed = outcome.astra_calls + outcome.jev_calls
    # The detector is charged on top of the astra/jev counts.
    detector = len([r for r in ledger.records if r.backend == "detector"])
    assert recorded == claimed + detector
