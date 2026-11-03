"""Shared plan driver.

Both pipelines execute the *same* generated plan through the *same* low-level
executor. The only thing that differs is **who decides each subgoal, and how
often an expensive model is invoked**. Holding everything else fixed is what
makes the cost delta attributable.

What changed from the first version of this file, and why it matters: the plan
used to be a hard-coded eight-entry enum with hand-written tolerances and an
if-else chain mapping each entry to a waypoint. That made the routing
mechanism general but the thing it routed over task-specific, so a new task
meant new code -- and the scaffold, not the models, was doing the work. Now
the plan is data (`astraxjev/plan.py`), generated per episode by Astra-Global
and calibrated against quantities the environment declares. The routing rule
below reads only `tolerance_mm` and `needs_pixels`, so it runs unchanged on a
plan it has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..cost import CostLedger
from ..executor import WaypointExecutor
from ..plan import Frame, GoalKind, Subgoal, TaskPlan, resolve
from ..senses import ProprioSenses, Senses
from ..types import (
    FINE_PRECISION_MM,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    EpisodeOutcome,
    Observation,
    SymbolicState,
    Waypoint,
)


class _Retry:
    """Sentinel: no waypoint this tick, but the subgoal is NOT finished.

    `decide` returning None means "this subgoal is complete, advance". A
    pipeline that spends its tick on something other than motion -- most
    importantly re-localising -- must not be read as having finished, which is
    what this sentinel is for.
    """

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "RETRY"


RETRY = _Retry()


@dataclass
class EpisodeContext:
    observe: Callable[[], Observation]
    senses: ProprioSenses
    executor: WaypointExecutor
    ledger: CostLedger
    seed: int
    episode: int


@dataclass
class Geometry:
    """Parameters of the coarse controller, not of any particular task."""

    #: A coarse controller steps in increments proportional to the error the
    #: subgoal allows: a 200 mm-tolerance drop into a basket can be taken in
    #: far bigger bites than a 35 mm-tolerance extraction from between two
    #: bottles. The step is always well above the fine tolerance, so coarse
    #: control structurally cannot pass itself off as fine control.
    #:
    #: This is the hybrid pipeline's main cost knob -- a smaller step means
    #: more Jev ticks per episode -- exactly as the re-localisation cadence is
    #: the baseline's. scripts/sweep_pricing.py sweeps both, because reporting
    #: one hand-picked value of either would be choosing the answer.
    coarse_step_scale: float = 4.0
    coarse_step_min_m: float = 0.05
    coarse_step_max_m: float = 0.30
    finger_length_m: float = 0.03
    #: Radius within which the cheap controller offers to hand over.
    handoff_radius_m: float = 0.06

    def coarse_step(self, tolerance_m: float) -> float:
        return float(
            min(
                self.coarse_step_max_m,
                max(self.coarse_step_min_m, self.coarse_step_scale * tolerance_m),
            )
        )


class PlanCursor:
    """Position in a generated plan, plus the completion tests."""

    #: Consecutive non-improving ticks after which a servo loop is declared
    #: plateaued. A visuomotor controller cannot servo below its own noise
    #: floor, so insisting on a sub-tolerance correction can loop forever.
    #: Applies identically to both pipelines -- the fine subgoals are Astra's
    #: in either case, so this must not be a knob that favours one of them.
    STALL_LIMIT = 1

    def __init__(self, plan: TaskPlan, geom: Geometry) -> None:
        self.plan = plan
        self.geom = geom
        self.index = 0
        self.steps_in_subgoal = 0
        self.ticks = 0
        self.best_correction = float("inf")
        self.stalled_ticks = 0
        #: The last waypoint the executor actually reached.
        self.last_reached: Optional[Waypoint] = None
        #: The pose reached when the current subgoal began. The relative
        #: primitives (WITHDRAW, ARC) measure from this, not from
        #: `last_reached`, which changes every tick and would make their goal
        #: recede by the same increment forever.
        self.anchor: Optional[Waypoint] = None
        self.anchor_tcp: Optional[np.ndarray] = None

    # -- position ----------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.index >= len(self.plan)

    @property
    def current(self) -> Subgoal:
        return self.plan[self.index]

    @property
    def name(self) -> str:
        return "done" if self.done else self.current.name

    def advance(self) -> None:
        self.index += 1
        self.steps_in_subgoal = 0
        self.best_correction = float("inf")
        self.stalled_ticks = 0
        self.anchor = self.last_reached
        self.anchor_tcp = None  # captured on the subgoal's first tick

    # -- the routing inputs ------------------------------------------------
    def tolerance_m(self) -> float:
        return self.current.tolerance_m

    def is_fine(self) -> bool:
        return self.current.is_fine

    # -- goal --------------------------------------------------------------
    def goal(self, refs: dict[str, Frame], senses: Senses) -> Optional[Waypoint]:
        if self.anchor_tcp is None:
            self.anchor_tcp = np.asarray(
                senses.tcp_position, dtype=np.float64
            ).reshape(3).copy()
        return resolve(
            self.current,
            refs,
            tcp_position=senses.tcp_position,
            anchor=self.anchor,
            anchor_tcp=self.anchor_tcp,
            finger_length_m=self.geom.finger_length_m,
        )

    # -- completion --------------------------------------------------------
    def reached(self, senses: Senses, goal: Waypoint) -> bool:
        """Belief-based test, valid only for the coarse subgoals.

        It compares the gripper against a goal derived from the *believed*
        reference frame. Fine when the tolerance is wider than the belief
        error, and wrong when it is not -- see `converged`.
        """
        return (
            float(
                np.linalg.norm(
                    np.asarray(senses.tcp_position).reshape(3)
                    - goal.position.reshape(3)
                )
            )
            <= self.tolerance_m()
        )

    def converged(self, senses_before: Senses, executed: Waypoint) -> bool:
        """Completion test for the fine subgoals.

        A visuomotor controller servos on what it sees, not on the stale
        global estimate, so its goal and the believed goal differ by more than
        a millimetre tolerance and `reached` would never fire. Two things do
        indicate it is finished: a correction smaller than the tolerance, or a
        correction that has stopped shrinking -- the loop has hit its own noise
        floor and further ticks only cost money.
        """
        correction = float(
            np.linalg.norm(
                np.asarray(senses_before.tcp_position).reshape(3)
                - executed.position.reshape(3)
            )
        )
        if correction <= self.tolerance_m():
            return True
        if correction < self.best_correction:
            self.best_correction = correction
            self.stalled_ticks = 0
            return False
        self.stalled_ticks += 1
        return self.stalled_ticks >= self.STALL_LIMIT

    def complete(self, senses_before: Senses, executed: Waypoint) -> bool:
        return (
            self.converged(senses_before, executed)
            if self.is_fine()
            else self.reached(senses_before, executed)
        )


class Pipeline:
    """Common episode driver.

    Subclasses implement `make_plan`, which must obtain a TaskPlan (and charge
    for it), and `decide`, which may call models (and must charge for them).
    """

    name = "pipeline"

    #: Ticks a single subgoal may spend without producing motion before the
    #: episode is declared stuck. Identical for both pipelines.
    RETRY_LIMIT = 8

    def __init__(self, geom: Geometry, *, max_ticks: int = 160) -> None:
        self.geom = geom
        self.max_ticks = int(max_ticks)

    # -- subclass hooks ----------------------------------------------------
    def reset(self, ctx: EpisodeContext) -> None:
        raise NotImplementedError

    def make_plan(self, ctx: EpisodeContext, obs: Observation) -> TaskPlan:
        """Obtain the task decomposition. One Astra-Global call."""
        raise NotImplementedError

    def decide(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        cursor: PlanCursor,
    ) -> Optional[Waypoint]:
        """Return the next waypoint, None to finish the subgoal, or RETRY.

        Every model call made here must be recorded in `ctx.ledger`.
        """
        raise NotImplementedError

    def call_counts(self) -> dict[str, int]:
        return {}

    # -- driver ------------------------------------------------------------
    def run_episode(self, ctx: EpisodeContext) -> EpisodeOutcome:
        ctx.ledger.begin_episode(ctx.episode)
        self.reset(ctx)
        failure = ""

        obs = ctx.observe()
        try:
            plan = self.make_plan(ctx, obs)
        except Exception as exc:  # a malformed plan is an episode failure
            return self._outcome(ctx, None, 0, f"planning failed: {exc}")

        cursor = PlanCursor(plan, self.geom)
        entered = -1

        while (
            cursor.ticks < self.max_ticks
            and not ctx.executor.episode_done()
            and not cursor.done
        ):
            ctx.ledger.set_env_step(ctx.executor.env_steps)
            cursor.ticks += 1
            cursor.steps_in_subgoal += 1
            sg = cursor.current

            # One-shot work on entering a subgoal. A torso lift is motion but
            # not a decision, so it costs no model call for either pipeline.
            if entered != cursor.index:
                entered = cursor.index
                lift = float(sg.params.get("torso_lift", 0.0))
                if abs(lift) > 1e-4:
                    ctx.executor.lift(lift)

            # Free primitives: pure actuation, no model call, identical for
            # both pipelines, so neither is charged or credited for them.
            if sg.kind is GoalKind.SET_GRIPPER:
                res = ctx.executor.set_gripper(sg.gripper)
                if not res.ok:
                    failure = f"{sg.name}: {res.reason}"
                    break
                cursor.advance()
                continue
            if sg.kind is GoalKind.HOLD:
                ctx.executor.idle(int(sg.params.get("steps", 20)))
                cursor.advance()
                continue

            obs = ctx.observe()
            senses = ctx.senses.read(obs)

            wp = self.decide(ctx, obs, senses, cursor)
            if wp is RETRY:
                # A tick spent on something other than motion still counts
                # against the subgoal. Without this, a pipeline that keeps
                # deciding to re-ground instead of moving spins until the
                # episode budget runs out, buying an expensive call each time.
                if cursor.steps_in_subgoal > self.RETRY_LIMIT:
                    failure = f"{sg.name}: {self.RETRY_LIMIT} ticks without motion"
                    break
                continue
            if wp is None:
                cursor.advance()
                continue

            res = ctx.executor.execute(wp)
            if not res.ok:
                # A failed motion plan is not fatal: re-decide from the new
                # state. Both pipelines get the same retry budget.
                if cursor.steps_in_subgoal > 12:
                    failure = f"{sg.name}: {res.reason}"
                    break
                continue

            cursor.last_reached = wp
            if cursor.complete(senses, wp):
                cursor.advance()

        if not failure and not cursor.done:
            failure = f"tick budget exhausted on {cursor.name}"

        return self._outcome(ctx, cursor, cursor.index, failure)

    def _outcome(
        self,
        ctx: EpisodeContext,
        cursor: Optional[PlanCursor],
        completed: int,
        failure: str,
    ) -> EpisodeOutcome:
        counts = self.call_counts()
        ep_usd = sum(r.usd for r in ctx.ledger.records if r.episode == ctx.episode)
        return EpisodeOutcome(
            seed=ctx.seed,
            success=ctx.executor.success(),
            env_steps=ctx.executor.env_steps,
            subgoal_reached=cursor.name if cursor else "plan",
            subgoals_completed=completed,
            subgoals_total=len(cursor.plan) if cursor else 0,
            cost_usd=ep_usd,
            astra_calls=counts.get("astra", 0),
            jev_calls=counts.get("jev", 0),
            failure_reason=failure,
        )


def symbolic_state(
    cursor: PlanCursor,
    senses: Senses,
    goal: Waypoint,
    frame_normal: np.ndarray,
    nearest_neighbour_m: float,
) -> SymbolicState:
    """Pack the coarse, text-serialisable view of the world Jev will read."""
    return SymbolicState(
        subgoal=cursor.current.name,
        goal_kind=cursor.current.kind.value,
        tcp_position=senses.tcp_position,
        goal_position=goal.position,
        frame_normal=frame_normal,
        gripper_closed=senses.gripper_closed,
        holding_item=senses.holding_item,
        nearest_neighbour_cm=nearest_neighbour_m * 100.0,
        steps_in_subgoal=cursor.steps_in_subgoal,
        steps_total=cursor.ticks,
    )
