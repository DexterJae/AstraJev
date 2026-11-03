import math

import numpy as np
import pytest

from astraxjev.cost import CostLedger, ModelPrice, PriceTable, summarise
from astraxjev.tokens import VisionTokenizerSpec, count_image_tokens, count_text_tokens
from astraxjev.types import EpisodeOutcome


def table():
    return PriceTable(
        {
            "astra": ModelPrice("astra", 1000.0, 10.0, 50.0, "test",
                                usd_per_mtok_cached_in=1.0),
            "jev": ModelPrice("jev", 2.0, 0.042, 0.0, "test"),
        }
    )


def outcome(success=True, seed=0):
    return EpisodeOutcome(seed, success, 10, "done", 7, 7, 0.0, 1, 1)


def test_published_rates_arithmetic():
    p = table()["astra"]
    assert p.usd(1_000_000, 0) == pytest.approx(10.0)
    assert p.usd(0, 1_000_000) == pytest.approx(50.0)
    # 2 * N * tokens, N in billions.
    assert p.gflops(100, 50) == pytest.approx(2 * 1000.0 * 150)


def test_output_dominates_the_frontier_model_bill():
    """Output is 5x input on gpt-6-astra, which is why prompts ask for JSON.

    If this ever stops holding the prompt design should change with it.
    """
    p = table()["astra"]
    assert p.usd(0, 1000) > 4 * p.usd(1000, 0)


def test_jev_output_is_free():
    p = table()["jev"]
    assert p.usd(0, 10_000_000) == pytest.approx(0.0)


def test_one_astra_call_costs_a_thousand_jev_calls():
    """The ratio the whole hybrid exploits. Guarded so a price edit shows up."""
    a, j = table()["astra"], table()["jev"]
    astra_call = a.usd(650, 280)
    jev_call = j.usd(183, 0)
    assert astra_call / jev_call > 1000


def test_cached_input_is_billed_at_the_cached_rate():
    """These pipelines re-send a long system prompt on every call."""
    p = table()["astra"]
    fresh = p.usd(1000, 0)
    cached = p.usd(1000, 0, cached_in=1000)
    assert cached == pytest.approx(fresh / 10)
    # Partial caching interpolates.
    assert fresh > p.usd(1000, 0, cached_in=500) > cached


def test_cached_tokens_never_exceed_input():
    p = table()["astra"]
    assert p.usd(100, 0, cached_in=500) == pytest.approx(p.usd(100, 0, cached_in=100))


def test_ledger_aggregates_by_backend_and_subgoal():
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("astra", "localize", "approach_item", 650, 280, images=3)
    led.record("jev", "query:a+b", "approach_item", 183, 0)
    led.record("jev", "query:a", "extract", 150, 0)

    by_b = led.by("backend")
    assert by_b["jev"]["calls"] == 2
    assert by_b["jev"]["tokens_in"] == 333
    by_s = led.by("subgoal")
    assert set(by_s) == {"approach_item", "extract"}


def test_provider_reported_cost_overrides_the_computed_one():
    """With the real APIs the dollar figure stops being an assumption."""
    led = CostLedger(table())
    led.begin_episode(0)
    rec = led.record("jev", "query:a", "x", 183, 0, reported_usd=1.4994e-05)
    assert rec.usd == pytest.approx(1.4994e-05)
    assert rec.usd_computed == pytest.approx(183 / 1e6 * 0.042)
    prov = led.cost_provenance()
    assert prov["fraction_measured"] == pytest.approx(1.0)


def test_cost_provenance_separates_measured_from_assumed():
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("astra", "localize", "x", 650, 280)  # assumed
    led.record("jev", "query:a", "x", 183, 0, reported_usd=1e-05)  # measured
    prov = led.cost_provenance()
    assert 0.0 < prov["fraction_measured"] < 1.0
    assert prov["calls_with_provider_cost"] == 1
    assert prov["usd_from_assumptions"] > 0


def test_recompute_changes_price_but_not_work():
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("astra", "localize", "x", 650, 280)
    cheaper = table().replace("astra", usd_per_mtok_in=1.0, usd_per_mtok_out=5.0)
    led2 = led.recompute(cheaper)
    assert [r.tokens_in for r in led2.records] == [r.tokens_in for r in led.records]
    assert led2.total_usd() == pytest.approx(led.total_usd() / 10)


def test_forward_passes_counted_separately_from_calls():
    """Jev's claim is one shared read of the state per batched query."""
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("jev", "query:a+b+c", "x", 180, 0, forward_passes=1)
    assert led.by("backend")["jev"]["forward_passes"] == 1
    assert led.by("backend")["jev"]["calls"] == 1


def test_usd_per_success_is_infinite_when_nothing_succeeds():
    """A pipeline must not look cheap by failing. Per-success cost shows it."""
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("astra", "localize", "x", 650, 280)
    s = summarise("p", led, [outcome(False, 0), outcome(False, 1)])
    assert s["success_rate"] == 0.0
    assert math.isinf(s["usd_per_success"])
    assert s["usd_per_episode"] > 0


def test_summary_carries_price_provenance():
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("astra", "localize", "x", 1, 1)
    s = summarise("p", led, [outcome()])
    assert s["prices"]["astra"]["provenance"] == "test"
    assert "cost_provenance" in s


def test_plan_progress_is_reported():
    """Separates "the plan was wrong" from "the controller could not run it"."""
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("astra", "plan_task", "plan", 700, 400)
    half = EpisodeOutcome(0, False, 10, "extract", 3, 7, 0.0, 1, 0)
    s = summarise("p", led, [half])
    assert s["mean_plan_progress"] == pytest.approx(3 / 7)


def test_text_tokens_are_deterministic_and_scale():
    a = count_text_tokens("subgoal=approach_item kind=standoff")
    assert a > 0 and count_text_tokens("") == 0
    assert count_text_tokens("x " * 100) > a


def test_image_tokens_follow_resolution():
    spec = VisionTokenizerSpec(patch_size=14, pool_factor=2)
    small = np.zeros((128, 128, 3), dtype=np.uint8)
    big = np.zeros((256, 256, 3), dtype=np.uint8)
    assert count_image_tokens([small], spec) < count_image_tokens([big], spec)
    assert count_image_tokens([big, big], spec) == 2 * count_image_tokens([big], spec)


def test_unknown_backend_is_an_error_not_a_free_call():
    """A silently-unpriced backend would understate a pipeline's cost."""
    led = CostLedger(table())
    led.begin_episode(0)
    with pytest.raises(KeyError):
        led.record("nope", "op", "x", 10, 0)
