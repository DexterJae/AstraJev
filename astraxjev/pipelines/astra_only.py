"""Pipeline A -- Astra alone, zero shot.

The expensive model does everything: it plans, it grounds the references, and
it emits every end-effector correction. That is the baseline the hybrid
attacks, and its cost structure is simple to state -- one `gpt-6-astra`
response per decision tick, at $50 per million output tokens.

Two parameters set how expensive it is:

  `global_every_n_ticks`  how often it re-grounds. Astra alone has no cheap
      way to ask "have my reference frames drifted?", because asking costs the
      same frontier call as answering. So it re-localises on a schedule. 0
      means ground once per episode, the cheapest this baseline can be.

  the tick count itself, which follows from the plan and the geometry.

Both are swept by scripts/sweep_pricing.py. Reporting one hand-picked cadence
would be a straw man.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..plan import Frame, TaskCapabilities, TaskPlan, calibrate
from ..senses import Senses
from ..types import Observation, Waypoint
from .base import RETRY, EpisodeContext, Geometry, Pipeline, PlanCursor


class PlanCache:
    """The decomposition, reused across episodes of a task.

    The plan for "pick X to basket" does not change between episodes -- only
    the grounding does -- so paying a frontier planning call every episode
    would be waste, not rigour. At $0.027 a call that waste would also be the
    largest single line item in both pipelines, which would flatter neither.

    Keyed on the instruction with the item name stripped. Both pipelines use
    this same class, so neither gains from it, and `hits`/`misses` go into the
    results so the amortisation is visible.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._store: dict[str, TaskPlan] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(instruction: str) -> str:
        # Keep the task words, drop the item name: the plan shape depends on
        # the verb and the destination, not on which brand is being picked.
        words = [w for w in instruction.split() if not w[:1].isupper()]
        return " ".join(w.lower() for w in words if w.isalpha())

    def get(self, instruction: str) -> Optional[TaskPlan]:
        if not self.enabled:
            return None
        plan = self._store.get(self.key(instruction))
        if plan is not None:
            self.hits += 1
        return plan

    def put(self, instruction: str, plan: TaskPlan) -> None:
        self.misses += 1
        if self.enabled:
            self._store[self.key(instruction)] = plan

    def stats(self) -> dict[str, int]:
        return {"plan_cache_hits": self.hits, "plan_cache_misses": self.misses}


class AstraPipelineBase(Pipeline):
    """Shared Astra plumbing: planning, grounding, and the obstacle belief."""

    def __init__(
        self,
        astra,
        geom: Geometry,
        *,
        max_ticks: int = 160,
        plan_cache: Optional[PlanCache] = None,
        capabilities: Optional[TaskCapabilities] = None,
    ) -> None:
        super().__init__(geom, max_ticks=max_ticks)
        self.astra = astra
        self.plan_cache = plan_cache or PlanCache()
        self.capabilities = capabilities
        self._refs: dict[str, Frame] = {}
        self._obstacles: list[np.ndarray] = []
        self._needed: tuple[str, ...] = ()
        self._n_astra = 0

    def reset(self, ctx: EpisodeContext) -> None:
        self._refs = {}
        self._obstacles = []
        self._needed = ()
        self._n_astra = 0

    # -- planning ----------------------------------------------------------
    def make_plan(self, ctx: EpisodeContext, obs: Observation) -> TaskPlan:
        plan = self.plan_cache.get(obs.prompt)
        if plan is None:
            r = self.astra.plan_task(obs)
            self._record(ctx, "plan_task", "plan", r)
            plan = r.plan
            if self.capabilities is not None:
                # The planner's tolerances are guesses -- a language model has
                # no way to know the gripper stroke or the task's success
                # threshold. The environment's numbers win. See plan.calibrate.
                plan = calibrate(plan, self.capabilities)
            self.plan_cache.put(obs.prompt, plan)
        self._needed = plan.required_refs
        return plan

    # -- grounding ---------------------------------------------------------
    def _localize(self, ctx: EpisodeContext, obs: Observation, subgoal: str) -> None:
        r = self.astra.localize(obs, self._needed)
        self._record(ctx, "localize", subgoal, r)
        self._refs.update(r.frames)
        # Obstacles come from the same call, so the coarse controller needs no
        # perception of its own: it dead-reckons against this list until the
        # belief is judged stale.
        self._obstacles = [np.asarray(p, dtype=np.float64).reshape(3) for p in r.obstacles]

    def _act_fine(
        self, ctx: EpisodeContext, obs: Observation, cursor: PlanCursor, goal: Waypoint
    ) -> Waypoint:
        r = self.astra.act_fine(obs, cursor.current, goal)
        self._record(ctx, "act_fine", cursor.current.name, r)
        return r.waypoint

    def _record(self, ctx: EpisodeContext, op: str, subgoal: str, r) -> None:
        ctx.ledger.record(
            self.astra.name,
            op,
            subgoal,
            r.tokens_in,
            r.tokens_out,
            images=r.images,
            wall_ms=r.wall_ms,
            cached_in=getattr(r, "cached_in", 0),
            reported_usd=getattr(r, "reported_cost_usd", None),
        )
        self._n_astra += 1

    # -- free, derived from the stored belief ------------------------------
    def nearest_obstacle_m(self, senses: Senses) -> float:
        """Distance from the gripper to the closest known obstacle.

        Free: the obstacle list came from the last grounding call and the TCP
        comes from forward kinematics. No model is consulted for this.
        """
        if not self._obstacles:
            return 10.0
        tcp = np.asarray(senses.tcp_position, dtype=np.float64).reshape(3)
        return float(min(np.linalg.norm(p - tcp) for p in self._obstacles))


class AstraOnlyPipeline(AstraPipelineBase):
    name = "astra"

    def __init__(
        self,
        astra,
        geom: Geometry,
        *,
        global_every_n_ticks: int = 4,
        max_ticks: int = 160,
        plan_cache: Optional[PlanCache] = None,
        capabilities: Optional[TaskCapabilities] = None,
    ) -> None:
        super().__init__(
            astra, geom, max_ticks=max_ticks,
            plan_cache=plan_cache, capabilities=capabilities,
        )
        self.global_every_n_ticks = int(global_every_n_ticks)

    def call_counts(self) -> dict[str, int]:
        return {"astra": self._n_astra, "jev": 0}

    def decide(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        cursor: PlanCursor,
    ) -> Optional[Waypoint]:
        self._refs.update(senses.free_frames)

        missing = [r for r in self._needed if r not in self._refs]
        due = (
            self.global_every_n_ticks > 0
            and cursor.ticks % self.global_every_n_ticks == 0
        )
        if missing or due:
            self._localize(ctx, obs, cursor.current.name)
            if missing:
                return RETRY

        goal = cursor.goal(self._refs, senses)
        if goal is None:
            return None
        # Only the coarse subgoals can be judged complete against the believed
        # goal; in the fine ones the driver decides from servo convergence.
        if not cursor.is_fine() and cursor.reached(senses, goal):
            return None
        return self._act_fine(ctx, obs, cursor, goal)
