"""Pipeline A -- Astra alone, zero shot.

Astra is used exactly as its own architecture prescribes (arXiv:2506.06205):
Astra-Global for localisation of the goal from pixels and language, Astra-Local
as the closed-loop local planner. The adaptation this project makes, and the
one to state plainly in a write-up, is that Astra-Local was published as a
*base* planner and is used here as a planner in end-effector space.

The cost structure of this pipeline is what the hybrid is attacking:

  * Astra-Local runs on every decision tick, because that is what closed-loop
    control means.
  * Astra-Global runs on a fixed cadence (and whenever Astra-Local reports low
    confidence), because Astra alone has no cheap module that can tell it
    whether its target belief has gone stale. Re-localising is the only way to
    find out, and finding out costs an MLLM forward pass.

`global_every_n_ticks` is therefore the single parameter that decides how
expensive this baseline is, which is precisely why scripts/sweep_pricing.py
sweeps it. Reporting one hand-picked cadence would be a straw man.
"""

from __future__ import annotations

from typing import Optional

from ..cost import CostLedger
from ..senses import Senses
from ..types import Observation, Phase, TargetEstimate, Waypoint
from .base import EpisodeContext, Geometry, PhaseMachine, Pipeline


class AstraOnlyPipeline(Pipeline):
    name = "astra"

    def __init__(
        self,
        astra,
        geom: Geometry,
        *,
        global_every_n_ticks: int = 4,
        chunk_len: int = 1,
        relocalize_confidence: float = 0.35,
        max_ticks: int = 120,
    ) -> None:
        super().__init__(geom, max_ticks=max_ticks)
        self.astra = astra
        self.global_every_n_ticks = int(global_every_n_ticks)
        self.chunk_len = int(chunk_len)
        self.relocalize_confidence = float(relocalize_confidence)
        self._target: Optional[TargetEstimate] = None
        self._last_confidence = 1.0
        self._n_global = 0
        self._n_local = 0

    def reset(self, ctx: EpisodeContext) -> None:
        self._target = None
        self._last_confidence = 1.0
        self._n_global = 0
        self._n_local = 0

    def call_counts(self) -> dict[str, int]:
        return {"astra": self._n_global + self._n_local, "jev": 0}

    # -- model calls -------------------------------------------------------
    def _localize(self, ctx: EpisodeContext, obs: Observation, phase: Phase) -> None:
        r = self.astra.localize(obs)
        ctx.ledger.record(
            self.astra.name_global,
            "localize",
            phase,
            r.tokens_in,
            r.tokens_out,
            images=r.images,
            wall_ms=r.wall_ms,
        )
        self._target = r.target
        self._n_global += 1

    def _plan(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        phase: Phase,
        goal: Waypoint,
    ) -> Waypoint:
        r = self.astra.plan_local(
            obs, self._target, phase, goal, self.chunk_len
        )
        ctx.ledger.record(
            self.astra.name_local,
            "plan_local",
            phase,
            r.tokens_in,
            r.tokens_out,
            images=r.images,
            wall_ms=r.wall_ms,
        )
        self._last_confidence = r.confidence
        self._n_local += 1
        return r.waypoints[-1]

    # -- decision ----------------------------------------------------------
    def decide(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        fsm: PhaseMachine,
    ) -> Optional[Waypoint]:
        if fsm.phase == Phase.LOCATE:
            self._localize(ctx, obs, fsm.phase)
            return None  # advance to APPROACH

        stale_by_cadence = (
            self.global_every_n_ticks > 0
            and fsm.ticks % self.global_every_n_ticks == 0
        )
        if stale_by_cadence or self._last_confidence < self.relocalize_confidence:
            self._localize(ctx, obs, fsm.phase)

        goal = fsm.goal(self._target, senses)
        # Only the coarse phases can be judged complete against the believed
        # goal; in the fine phases the driver decides from servo convergence.
        if not fsm.is_fine() and fsm.reached(senses, goal):
            return None
        return self._plan(ctx, obs, fsm.phase, goal)
