"""Cost ledger.

Every backend invocation is recorded with the token counts of the prompt that
was actually constructed, so the cost comparison between the two pipelines is
an audit of real work done rather than a hand-wave.

Three cost units are reported, because the paper claim is stronger if it does
not depend on any one provider's price sheet:

  usd     -- unit prices from config/pricing.yaml (an explicit assumption)
  gflops  -- 2 * params * tokens, hardware and vendor independent
  calls   -- number of model invocations, and for Jev the number of forward
             passes, which is the quantity its paper actually claims is 1

The headline metric is **cost per successful episode**: total cost divided by
the number of successes.  A pipeline that is cheap because it gives up early is
not cheaper in any sense a reader should accept, and cost-per-success is what
makes that visible.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional



@dataclass(frozen=True)
class ModelPrice:
    """Unit price for one backend.

    `provenance` is required and carried into the report: a cost claim whose
    inputs cannot be traced is not a result.
    """

    name: str
    params_b: float
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    provenance: str
    #: Cached input is an order of magnitude cheaper on the frontier model, and
    #: these pipelines re-send the same system prompt on every call, so
    #: ignoring it would materially overstate both arms' cost.
    usd_per_mtok_cached_in: Optional[float] = None
    # Fixed overhead charged per call regardless of tokens.
    usd_per_call: float = 0.0

    def usd(
        self, tokens_in: int, tokens_out: int, cached_in: int = 0
    ) -> float:
        cached_rate = (
            self.usd_per_mtok_in
            if self.usd_per_mtok_cached_in is None
            else self.usd_per_mtok_cached_in
        )
        # Clamp: a provider reporting more cached than total input must not be
        # able to inflate the bill past the tokens actually sent.
        cached = max(0, min(int(cached_in), int(tokens_in)))
        fresh = tokens_in - cached
        return (
            fresh / 1e6 * self.usd_per_mtok_in
            + cached / 1e6 * cached_rate
            + tokens_out / 1e6 * self.usd_per_mtok_out
            + self.usd_per_call
        )

    def gflops(self, tokens_in: int, tokens_out: int) -> float:
        """Forward-pass FLOPs, 2*N per parameter-token.

        Prefill is one pass over `tokens_in`; autoregressive decoding is one
        pass per output token over the growing context, which to first order is
        2*N*tokens_out as well.  This is the standard approximation and is
        enough to show the order-of-magnitude gap we care about.
        """
        return 2.0 * self.params_b * (tokens_in + tokens_out)


@dataclass
class CallRecord:
    backend: str
    op: str
    #: Name of the generated subgoal this call served. Free text, because the
    #: plan is generated rather than drawn from a fixed enum.
    subgoal: str
    tokens_in: int
    tokens_out: int
    images: int
    forward_passes: int
    wall_ms: float
    usd: float
    gflops: float
    episode: int
    env_step: int
    #: Input tokens served from the provider's prompt cache.
    cached_in: int = 0
    #: The provider's own USD figure, when the API returns one. When present,
    #: `usd` is set from it, so that part of the comparison is measured rather
    #: than assumed. `usd_computed` keeps our figure for comparison, which is
    #: how a wrong price assumption becomes visible instead of silent.
    reported_usd: Optional[float] = None
    usd_computed: Optional[float] = None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


class PriceTable:
    def __init__(self, prices: dict[str, ModelPrice]) -> None:
        self._prices = dict(prices)

    def __getitem__(self, backend: str) -> ModelPrice:
        if backend not in self._prices:
            raise KeyError(
                f"no price for backend {backend!r}; known: {sorted(self._prices)}"
            )
        return self._prices[backend]

    def __contains__(self, backend: str) -> bool:
        return backend in self._prices

    def names(self) -> list[str]:
        return sorted(self._prices)

    def replace(self, backend: str, **changes: Any) -> "PriceTable":
        """Return a copy with one price overridden -- used by the sweep."""
        prices = dict(self._prices)
        base = prices[backend]
        prices[backend] = ModelPrice(**{**asdict(base), **changes})
        return PriceTable(prices)

    @classmethod
    def from_dict(cls, raw: dict[str, dict]) -> "PriceTable":
        return cls({k: ModelPrice(name=k, **v) for k, v in raw.items()})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PriceTable":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw["models"])

    def to_dict(self) -> dict[str, dict]:
        return {k: asdict(v) for k, v in self._prices.items()}


class CostLedger:
    """Append-only record of every model call in a run."""

    def __init__(self, prices: PriceTable) -> None:
        self.prices = prices
        self.records: list[CallRecord] = []
        self._episode = -1
        self._env_step = 0

    # -- context -----------------------------------------------------------
    def begin_episode(self, episode: int) -> None:
        self._episode = episode
        self._env_step = 0

    def set_env_step(self, step: int) -> None:
        self._env_step = int(step)

    # -- recording ---------------------------------------------------------
    def record(
        self,
        backend: str,
        op: str,
        subgoal: str,
        tokens_in: int,
        tokens_out: int,
        *,
        images: int = 0,
        forward_passes: int = 1,
        wall_ms: float = 0.0,
        cached_in: int = 0,
        reported_usd: Optional[float] = None,
    ) -> CallRecord:
        price = self.prices[backend]
        computed = price.usd(tokens_in, tokens_out, cached_in)
        rec = CallRecord(
            backend=backend,
            op=op,
            subgoal=str(subgoal),
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            images=int(images),
            forward_passes=int(forward_passes),
            wall_ms=float(wall_ms),
            usd=float(reported_usd) if reported_usd is not None else computed,
            gflops=price.gflops(tokens_in, tokens_out),
            episode=self._episode,
            env_step=self._env_step,
            cached_in=int(cached_in),
            reported_usd=float(reported_usd) if reported_usd is not None else None,
            usd_computed=computed,
        )
        self.records.append(rec)
        return rec

    # -- aggregation -------------------------------------------------------
    def total_usd(self, records: Optional[Iterable[CallRecord]] = None) -> float:
        return sum(r.usd for r in (self.records if records is None else records))

    def total_gflops(self, records: Optional[Iterable[CallRecord]] = None) -> float:
        return sum(r.gflops for r in (self.records if records is None else records))

    def by(self, key: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self.records:
            k = getattr(r, key)
            agg = out.setdefault(
                k,
                {"calls": 0, "forward_passes": 0, "tokens_in": 0,
                 "tokens_out": 0, "cached_in": 0, "usd": 0.0, "gflops": 0.0,
                 "wall_ms": 0.0},
            )
            agg["calls"] += 1
            agg["forward_passes"] += r.forward_passes
            agg["tokens_in"] += r.tokens_in
            agg["tokens_out"] += r.tokens_out
            agg["cached_in"] += r.cached_in
            agg["usd"] += r.usd
            agg["gflops"] += r.gflops
            agg["wall_ms"] += r.wall_ms
        return out

    def per_episode_usd(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for r in self.records:
            out[r.episode] = out.get(r.episode, 0.0) + r.usd
        return out

    def latency_percentiles(self, backend: str) -> dict[str, float]:
        xs = sorted(r.wall_ms for r in self.records if r.backend == backend)
        if not xs:
            return {}
        return {
            "p50_ms": xs[len(xs) // 2],
            "p95_ms": xs[min(len(xs) - 1, int(0.95 * len(xs)))],
            "mean_ms": statistics.fmean(xs),
        }

    def cost_provenance(self) -> dict[str, Any]:
        reported = [r for r in self.records if r.reported_usd is not None]
        total = self.total_usd()
        rep_sum = sum(r.reported_usd for r in reported)
        drift = [
            abs(r.reported_usd - (r.usd_computed or 0.0))
            for r in reported
            if r.usd_computed
        ]
        return {
            "calls_total": len(self.records),
            "calls_with_provider_cost": len(reported),
            "usd_total": total,
            "usd_from_provider": rep_sum,
            "usd_from_assumptions": total - rep_sum,
            "fraction_measured": (rep_sum / total) if total else 0.0,
            "max_abs_price_drift_usd": max(drift) if drift else 0.0,
        }

    def recompute(self, prices: PriceTable) -> "CostLedger":
        """Re-price the same recorded work under a different price table.

        This is what makes the cost claim falsifiable: the *work* is fixed by
        the run, so a reader can re-derive the dollar figure under their own
        assumptions without re-running the simulator.
        """
        clone = CostLedger(prices)
        for r in self.records:
            p = prices[r.backend]
            clone.records.append(
                CallRecord(
                    **{
                        **r.as_row(),
                        "usd": p.usd(r.tokens_in, r.tokens_out, r.cached_in),
                        "gflops": p.gflops(r.tokens_in, r.tokens_out),
                    }
                )
            )
        return clone


def summarise(
    name: str, ledger: CostLedger, outcomes: list
) -> dict[str, Any]:
    """Build the per-pipeline result dict that the comparison script consumes."""
    n = len(outcomes)
    n_success = sum(1 for o in outcomes if o.success)
    total_usd = ledger.total_usd()
    total_gflops = ledger.total_gflops()
    per_ep = ledger.per_episode_usd()

    return {
        "pipeline": name,
        "episodes": n,
        "successes": n_success,
        "success_rate": (n_success / n) if n else 0.0,
        "total_usd": total_usd,
        "usd_per_episode": (total_usd / n) if n else 0.0,
        # The honest headline: infinite if nothing ever succeeds.
        "usd_per_success": (total_usd / n_success) if n_success else float("inf"),
        "total_gflops": total_gflops,
        "gflops_per_episode": (total_gflops / n) if n else 0.0,
        "gflops_per_success": (total_gflops / n_success)
        if n_success
        else float("inf"),
        "mean_env_steps": (sum(o.env_steps for o in outcomes) / n) if n else 0.0,
        # How far through the generated plan episodes got, which separates "the
        # plan was wrong" from "the controller could not execute it".
        "mean_plan_progress": (
            sum(
                o.subgoals_completed / max(1, o.subgoals_total) for o in outcomes
            )
            / n
        )
        if n
        else 0.0,
        "by_backend": ledger.by("backend"),
        "by_subgoal": ledger.by("subgoal"),
        "by_op": ledger.by("op"),
        "latency": {b: ledger.latency_percentiles(b) for b in ledger.prices.names()},
        "usd_per_episode_spread": {
            "min": min(per_ep.values()) if per_ep else 0.0,
            "max": max(per_ep.values()) if per_ep else 0.0,
        },
        "prices": ledger.prices.to_dict(),
        # How much of the dollar figure came from the provider rather than from
        # config/pricing.yaml. A reader can see at a glance how much of the
        # cost claim rests on an assumption.
        "cost_provenance": ledger.cost_provenance(),
        "outcomes": [
            {
                "seed": o.seed,
                "success": o.success,
                "env_steps": o.env_steps,
                "subgoal_reached": o.subgoal_reached,
                "subgoals_completed": o.subgoals_completed,
                "subgoals_total": o.subgoals_total,
                "cost_usd": o.cost_usd,
                "astra_calls": o.astra_calls,
                "jev_calls": o.jev_calls,
                "failure_reason": o.failure_reason,
            }
            for o in outcomes
        ],
    }


def write_result(path: str | Path, result: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
