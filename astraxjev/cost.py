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

from .types import Phase


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
    # Fixed overhead charged per call regardless of tokens (request setup,
    # per-image preprocessing, etc.).
    usd_per_call: float = 0.0

    def usd(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in / 1e6 * self.usd_per_mtok_in
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
    phase: str
    tokens_in: int
    tokens_out: int
    images: int
    forward_passes: int
    wall_ms: float
    usd: float
    gflops: float
    episode: int
    env_step: int

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
        phase: Phase | str,
        tokens_in: int,
        tokens_out: int,
        *,
        images: int = 0,
        forward_passes: int = 1,
        wall_ms: float = 0.0,
    ) -> CallRecord:
        price = self.prices[backend]
        rec = CallRecord(
            backend=backend,
            op=op,
            phase=phase.value if isinstance(phase, Phase) else str(phase),
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            images=int(images),
            forward_passes=int(forward_passes),
            wall_ms=float(wall_ms),
            usd=price.usd(tokens_in, tokens_out),
            gflops=price.gflops(tokens_in, tokens_out),
            episode=self._episode,
            env_step=self._env_step,
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
                {"calls": 0, "forward_passes": 0, "tokens_in": 0, "tokens_out": 0,
                 "usd": 0.0, "gflops": 0.0, "wall_ms": 0.0},
            )
            agg["calls"] += 1
            agg["forward_passes"] += r.forward_passes
            agg["tokens_in"] += r.tokens_in
            agg["tokens_out"] += r.tokens_out
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
                        "usd": p.usd(r.tokens_in, r.tokens_out),
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
        "by_backend": ledger.by("backend"),
        "by_phase": ledger.by("phase"),
        "by_op": ledger.by("op"),
        "latency": {b: ledger.latency_percentiles(b) for b in ledger.prices.names()},
        "usd_per_episode_spread": {
            "min": min(per_ep.values()) if per_ep else 0.0,
            "max": max(per_ep.values()) if per_ep else 0.0,
        },
        "prices": ledger.prices.to_dict(),
        "outcomes": [
            {
                "seed": o.seed,
                "success": o.success,
                "env_steps": o.env_steps,
                "phase_reached": o.phase_reached.value,
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
