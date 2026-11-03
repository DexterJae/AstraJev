"""Shared phase machine.

Both pipelines solve PickToBasket through the same seven-phase decomposition
and the same geometry, and execute through the same low-level executor. The
only thing that differs is **who decides, and how often an expensive model is
asked**. Holding everything else fixed is what makes the cost delta
attributable.

Phase tolerances (astraxjev/types.PHASE_SPEC) are what assign work:

  LOCATE    50 mm, needs pixels   -> Astra-Global. No substitute: only an MLLM
                                     turns "pick the Nivea Body Milk" plus a
                                     cluttered shelf image into a 3D point.
  APPROACH  40 mm                 -> coarse. Jev's job in the hybrid.
  ALIGN      6 mm, needs pixels   -> Astra-Local. The hybrid never delegates
  GRASP      6 mm, needs pixels      this; a centimetre of error here is a
                                     knocked-over neighbour and a void episode.
  RETRACT   25 mm                 -> coarse, but risk-gated: success is void if
                                     a non-target item is displaced.
  PLACE     60 mm                 -> coarse. The basket is a loose target.
  SETTLE    --                    -> free. Success requires `is_robot_static`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .. import geometry
from ..cost import CostLedger
from ..executor import WaypointExecutor
from ..senses import ProprioSenses, Senses
from ..types import (
    FINE_PRECISION_MM,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    EpisodeOutcome,
    Observation,
    Phase,
    PHASE_SPEC,
    SymbolicState,
    TargetEstimate,
    Waypoint,
)

class _Retry:
    """Sentinel: no waypoint this tick, but the phase is NOT finished.

    `decide` returning None means "this phase is complete, advance". A
    pipeline that spends its tick on something other than motion -- most
    importantly re-localising -- must not be read as having finished the
    phase, which is what this sentinel is for.
    """

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "RETRY"


RETRY = _Retry()

PHASE_ORDER = [
    Phase.LOCATE,
    Phase.APPROACH,
    Phase.ALIGN,
    Phase.GRASP,
    Phase.RETRACT,
    Phase.PLACE,
    Phase.SETTLE,
    Phase.DONE,
]


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
    standoff_m: float = 0.18
    handoff_radius_m: float = 0.06
    retract_m: float = 0.30
    finger_length_m: float = 0.03
    #: A coarse controller steps in increments proportional to the error the
    #: phase allows: a 60 mm-tolerance move to the basket can be taken in far
    #: bigger bites than a 25 mm-tolerance extraction from between two
    #: bottles. The step is always at least an order of magnitude above the
    #: 6 mm fine tolerance, so coarse control structurally cannot pass itself
    #: off as fine control.
    #:
    #: This is the hybrid pipeline's main cost knob -- a smaller step means
    #: more Jev ticks per episode -- exactly as `global_every_n_ticks` is the
    #: baseline's. scripts/sweep_pricing.py sweeps both, because reporting one
    #: hand-picked value of either would be choosing the answer.
    coarse_step_scale: float = 4.0
    coarse_step_min_m: float = 0.05
    coarse_step_max_m: float = 0.30
    #: Offset from the basket success centre to the release pose, and how far
    #: to raise the torso before reaching over the basket. Both copied from
    #: `drop_to_basket` in the benchmark's reference solution; the arm cannot
    #: reach the release pose from a low torso.
    basket_release_offset_m: tuple = (0.05, 0.0, 0.40)
    pre_place_lift_m: float = 0.30

    def coarse_step(self, tolerance_m: float) -> float:
        return float(
            min(
                self.coarse_step_max_m,
                max(self.coarse_step_min_m, self.coarse_step_scale * tolerance_m),
            )
        )


class PhaseMachine:
    """Tracks the phase and computes each phase's goal waypoint."""

    def __init__(self, geom: Geometry) -> None:
        self.geom = geom
        self.phase = Phase.LOCATE
        self.steps_in_phase = 0
        self.ticks = 0
        self.grasp_pose: Optional[Waypoint] = None
        #: Smallest correction the fine-phase controller has asked for so far,
        #: and how many ticks in a row have failed to improve on it.
        self.best_correction = float("inf")
        self.stalled_ticks = 0
        #: The last waypoint the executor actually reached. Downstream
        #: phases anchor on this rather than on the believed target, so a
        #: centimetre of localisation error does not propagate into RETRACT.
        self.last_executed: Optional[Waypoint] = None

    def advance(self) -> Phase:
        i = PHASE_ORDER.index(self.phase)
        self.phase = PHASE_ORDER[min(i + 1, len(PHASE_ORDER) - 1)]
        self.steps_in_phase = 0
        self.best_correction = float("inf")
        self.stalled_ticks = 0
        return self.phase

    def tolerance_m(self) -> float:
        return PHASE_SPEC[self.phase]["precision_mm"] / 1000.0

    def goal(self, target: TargetEstimate, senses: Senses) -> Waypoint:
        g = self.geom
        if self.phase in (Phase.LOCATE, Phase.APPROACH):
            return geometry.standoff_waypoint(target, g.standoff_m)
        if self.phase in (Phase.ALIGN, Phase.GRASP):
            wp = geometry.grasp_waypoint(target, g.finger_length_m)
            self.grasp_pose = wp
            return wp
        if self.phase == Phase.RETRACT:
            base = self.grasp_pose or geometry.grasp_waypoint(
                target, g.finger_length_m
            )
            return geometry.retract_waypoint(base, target, g.retract_m)
        if self.phase == Phase.PLACE:
            return geometry.place_waypoint(
                senses.basket_position,
                release_offset=self.geom.basket_release_offset_m,
            )
        # SETTLE / DONE: stay where we are, gripper open.
        return Waypoint(
            position=senses.tcp_position,
            approach=np.array([0.0, 0.0, -1.0]),
            closing=np.array([0.0, 1.0, 0.0]),
            gripper=GRIPPER_OPEN,
        )

    def reached(self, senses: Senses, goal: Waypoint) -> bool:
        """Belief-based completion test, valid only for the coarse phases.

        It compares the gripper against the goal derived from the *believed*
        target. That is fine when the tolerance (25-60 mm) is wider than the
        belief error, and wrong when it is not -- see `converged`.
        """
        return geometry.distance(senses.tcp_position, goal.position) <= self.tolerance_m()

    #: Consecutive non-improving ticks after which a servo loop is declared
    #: plateaued. A visuomotor controller cannot servo below its own noise
    #: floor, so insisting on a sub-tolerance correction can loop forever.
    #: Applies identically to both pipelines -- the fine phases are Astra's in
    #: either case, so this must not be a knob that favours one of them.
    STALL_LIMIT = 1

    def converged(self, senses_before: Senses, executed: Waypoint) -> bool:
        """Completion test for the fine phases.

        A visuomotor controller in ALIGN servos on what it sees, not on the
        stale global estimate, so its goal and the believed goal differ by more
        than the 6 mm tolerance and `reached` would never fire. Two things do
        indicate it is finished: a correction smaller than the tolerance, or a
        correction that has stopped shrinking -- the loop has reached its own
        noise floor and further ticks only cost money.
        """
        correction = geometry.distance(senses_before.tcp_position, executed.position)
        if correction <= self.tolerance_m():
            return True
        if correction < self.best_correction:
            self.best_correction = correction
            self.stalled_ticks = 0
            return False
        self.stalled_ticks += 1
        return self.stalled_ticks >= self.STALL_LIMIT

    def is_fine(self) -> bool:
        return PHASE_SPEC[self.phase]["precision_mm"] <= FINE_PRECISION_MM

    def complete(self, senses_before: Senses, executed: Waypoint) -> bool:
        return (
            self.converged(senses_before, executed)
            if self.is_fine()
            else self.reached(senses_before, executed)
        )


class Pipeline:
    """Common episode driver.

    Subclasses implement `decide`, which may call models (and must record every
    call in the ledger) and returns the next waypoint to execute, or None to
    advance the phase.
    """

    name = "pipeline"

    def __init__(self, geom: Geometry, *, max_ticks: int = 120) -> None:
        self.geom = geom
        self.max_ticks = int(max_ticks)

    # -- subclass hooks ----------------------------------------------------
    def reset(self, ctx: EpisodeContext) -> None:
        raise NotImplementedError

    def decide(
        self, ctx: EpisodeContext, obs: Observation, senses: Senses, fsm: PhaseMachine
    ) -> Optional[Waypoint]:
        """Return the next waypoint, None to finish the phase, or RETRY.

        Every model call made here must be recorded in `ctx.ledger`.
        """
        raise NotImplementedError

    def call_counts(self) -> dict[str, int]:
        return {}

    # -- driver ------------------------------------------------------------
    def run_episode(self, ctx: EpisodeContext) -> EpisodeOutcome:
        ctx.ledger.begin_episode(ctx.episode)
        fsm = PhaseMachine(self.geom)
        self.reset(ctx)
        failure = ""

        while fsm.ticks < self.max_ticks and not ctx.executor.episode_done():
            ctx.ledger.set_env_step(ctx.executor.env_steps)
            fsm.ticks += 1
            fsm.steps_in_phase += 1

            obs = ctx.observe()
            senses = ctx.senses.read(obs)

            # Phases that are pure actuation need no model call at all, so
            # neither pipeline is charged or credited for them.
            if fsm.phase == Phase.GRASP:
                res = ctx.executor.set_gripper(GRIPPER_CLOSED)
                if not res.ok:
                    failure = f"grasp: {res.reason}"
                    break
                self._finish_phase(ctx, fsm)
                continue
            if fsm.phase == Phase.SETTLE:
                # Success requires `is_robot_static`, so hold still before the
                # episode is scored.
                ctx.executor.idle(20)
                self._finish_phase(ctx, fsm)
                continue
            if fsm.phase == Phase.DONE:
                break

            wp = self.decide(ctx, obs, senses, fsm)
            if wp is RETRY:
                continue
            if wp is None:
                self._finish_phase(ctx, fsm)
                continue

            res = ctx.executor.execute(wp)
            if not res.ok:
                # A failed motion plan is not fatal: re-decide from the new
                # state. Both pipelines get the same retry budget.
                if fsm.steps_in_phase > 12:
                    failure = f"{fsm.phase.value}: {res.reason}"
                    break
                continue

            fsm.last_executed = wp
            if fsm.complete(senses, wp):
                self._finish_phase(ctx, fsm)

        if not failure and fsm.phase != Phase.DONE:
            failure = f"tick budget exhausted in {fsm.phase.value}"

        counts = self.call_counts()
        ep_usd = sum(r.usd for r in ctx.ledger.records if r.episode == ctx.episode)
        return EpisodeOutcome(
            seed=ctx.seed,
            success=ctx.executor.success(),
            env_steps=ctx.executor.env_steps,
            phase_reached=fsm.phase,
            cost_usd=ep_usd,
            astra_calls=counts.get("astra", 0),
            jev_calls=counts.get("jev", 0),
            failure_reason=failure,
        )

    def _finish_phase(self, ctx: EpisodeContext, fsm: PhaseMachine) -> None:
        """Leave the current phase, doing whatever leaving it entails."""
        if fsm.phase == Phase.ALIGN and fsm.last_executed is not None:
            # Anchor RETRACT on where the gripper actually ended up, not on the
            # believed target, so localisation error does not propagate.
            fsm.grasp_pose = fsm.last_executed
        if fsm.phase == Phase.RETRACT and self.geom.pre_place_lift_m:
            # The reference solution raises the torso before reaching over the
            # basket; without it the release pose is outside the arm's range.
            ctx.executor.lift(self.geom.pre_place_lift_m)
        if fsm.phase == Phase.PLACE:
            # Releasing over the basket is what actually completes the task.
            ctx.executor.set_gripper(GRIPPER_OPEN)
        fsm.advance()


def symbolic_state(
    fsm: PhaseMachine,
    senses: Senses,
    target: TargetEstimate,
    geom: Geometry,
    nearest_neighbour_m: float,
) -> SymbolicState:
    """Pack the coarse, text-serialisable view of the world Jev will read."""
    standoff = geometry.standoff_waypoint(target, geom.standoff_m).position
    return SymbolicState(
        phase=fsm.phase,
        tcp_position=senses.tcp_position,
        target_position=target.position,
        standoff_position=standoff,
        basket_position=senses.basket_position,
        shelf_normal=target.shelf_normal,
        gripper_closed=senses.gripper_closed,
        holding_item=senses.holding_item,
        nearest_neighbour_cm=nearest_neighbour_m * 100.0,
        steps_in_phase=fsm.steps_in_phase,
        steps_total=fsm.ticks,
    )
