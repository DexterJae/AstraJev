import math

import pytest

from astraxjev.cost import CostLedger, ModelPrice, PriceTable, summarise
from astraxjev.tokens import VisionTokenizerSpec, count_image_tokens, count_text_tokens
from astraxjev.types import EpisodeOutcome, Phase


def table():
    return PriceTable(
        {
            "big": ModelPrice("big", 7.0, 0.20, 0.60, "test"),
            "small": ModelPrice("small", 0.1, 0.01, 0.0, "test"),
        }
    )


def test_usd_and_gflops_arithmetic():
    p = table()["big"]
    assert p.usd(1_000_000, 0) == pytest.approx(0.20)
    assert p.usd(0, 1_000_000) == pytest.approx(0.60)
    # 2 * N * tokens, with N in billions.
    assert p.gflops(100, 50) == pytest.approx(2 * 7.0 * 150)


def test_per_call_overhead_is_charged():
    p = ModelPrice("x", 1.0, 0.0, 0.0, "test", usd_per_call=0.002)
    assert p.usd(0, 0) == pytest.approx(0.002)


def test_ledger_aggregates_by_backend_and_phase():
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("big", "localize", Phase.LOCATE, 400, 60, images=3)
    led.record("small", "detect", Phase.APPROACH, 100, 0, images=2)
    led.record("small", "detect", Phase.APPROACH, 100, 0, images=2)

    by_b = led.by("backend")
    assert by_b["small"]["calls"] == 2
    assert by_b["small"]["tokens_in"] == 200
    assert by_b["big"]["calls"] == 1

    by_p = led.by("phase")
    assert set(by_p) == {"locate", "approach"}
    assert led.total_usd() == pytest.approx(
        400 / 1e6 * 0.20 + 60 / 1e6 * 0.60 + 200 / 1e6 * 0.01
    )


def test_recompute_changes_price_but_not_work():
    """Re-pricing must preserve every token count.

    This is what lets a reader re-derive the dollar figure under their own
    assumptions without re-running the simulator.
    """
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("big", "localize", Phase.LOCATE, 400, 60)

    cheaper = table().replace("big", usd_per_mtok_in=0.02, usd_per_mtok_out=0.06)
    led2 = led.recompute(cheaper)

    assert [r.tokens_in for r in led2.records] == [r.tokens_in for r in led.records]
    assert [r.forward_passes for r in led2.records] == [1]
    assert led2.total_usd() == pytest.approx(led.total_usd() / 10)


def test_forward_passes_counted_separately_from_calls():
    """Jev's claim is one forward pass per batched query, so it is recorded."""
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("small", "query:a+b+c", Phase.APPROACH, 180, 0, forward_passes=1)
    assert led.by("backend")["small"]["forward_passes"] == 1
    assert led.by("backend")["small"]["calls"] == 1


def test_usd_per_success_is_infinite_when_nothing_succeeds():
    """A pipeline must not look cheap by failing. Per-success cost shows it."""
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("big", "localize", Phase.LOCATE, 400, 60)
    outcomes = [
        EpisodeOutcome(0, False, 10, Phase.ALIGN, 1e-4, 1, 0),
        EpisodeOutcome(1, False, 10, Phase.ALIGN, 1e-4, 1, 0),
    ]
    s = summarise("p", led, outcomes)
    assert s["success_rate"] == 0.0
    assert math.isinf(s["usd_per_success"])
    assert s["usd_per_episode"] > 0


def test_summary_carries_price_provenance():
    """Every result must be traceable to the assumptions that produced it."""
    led = CostLedger(table())
    led.begin_episode(0)
    led.record("big", "localize", Phase.LOCATE, 1, 1)
    s = summarise("p", led, [EpisodeOutcome(0, True, 1, Phase.DONE, 0.0, 1, 0)])
    assert s["prices"]["big"]["provenance"] == "test"


def test_text_tokens_are_deterministic_and_scale():
    a = count_text_tokens("phase=approach goal=standoff")
    assert a > 0
    assert count_text_tokens("") == 0
    assert count_text_tokens("x " * 100) > a


def test_image_tokens_follow_resolution():
    spec = VisionTokenizerSpec(patch_size=14, pool_factor=2)
    import numpy as np

    small = np.zeros((128, 128, 3), dtype=np.uint8)
    big = np.zeros((256, 256, 3), dtype=np.uint8)
    assert count_image_tokens([small], spec) < count_image_tokens([big], spec)
    assert count_image_tokens([big, big], spec) == 2 * count_image_tokens([big], spec)


def test_unknown_backend_is_an_error_not_a_free_call():
    """A silently-unpriced backend would understate a pipeline's cost."""
    led = CostLedger(table())
    led.begin_episode(0)
    with pytest.raises(KeyError):
        led.record("nope", "op", Phase.APPROACH, 10, 0)
