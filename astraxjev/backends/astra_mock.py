"""Mock Astra backend.

Stands in for `gpt-6-astra` so the pipelines, the plan driver and the cost
ledger can be exercised without an API key or a bill. Token counts come from
astraxjev/prompts.py -- the prompts really constructed -- so the ledger a mock
run produces is the ledger a real run would produce.

The capability model is the substantive part, because it decides whether the
division of labour makes sense at all. Three claims, each about a frontier
vision LLM:

  It plans well. Decomposing "pick the Nivea Body Milk into the basket" into
      reach / grasp / extract / place is exactly the kind of commonsense
      structure such a model supplies. The planner here is a *stand-in* that
      keys off the instruction text to produce the plan a real call would be
      asked for -- the single biggest simulation left in this project.

  It localises semantically but coarsely. It reads the brand off the packaging
      and returns a 3D centre with a few centimetres of error. The error
      shrinks as the gripper gets closer, because the item then fills more of
      the wrist image -- which is why re-localising is worth anything, and
      therefore why "should I re-localise?" is a question worth paying for.

  It can correct a pose to millimetres when it is looking at the thing. So
      `act_fine` servos on the true reference rather than on the stale stored
      estimate. This is the capability the hybrid pays for and never
      delegates. It is also the most optimistic of the three assumptions, and
      the one a real run is most likely to contradict.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import prompts
from ..oracle import Oracle
from ..plan import Frame, GoalKind, Subgoal, TaskPlan, resolve
from ..types import GRIPPER_CLOSED, GRIPPER_OPEN, Observation, Waypoint
from .base import AstraActResponse, AstraLocalizeResponse, AstraPlanResponse


class MockAstraBackend:
    name = "astra"

    def __init__(
        self,
        oracle: Oracle,
        *,
        localize_sigma_m: float = 0.025,
        act_sigma_m: float = 0.004,
        rng: np.random.Generator | None = None,
        plan_wall_ms: float = 4200.0,
        localize_wall_ms: float = 2100.0,
        act_wall_ms: float = 1800.0,
        finger_length_m: float = 0.03,
        #: Distance at which localisation performs at its nominal sigma.
        nominal_view_distance_m: float = 0.5,
    ) -> None:
        self.oracle = oracle
        self.localize_sigma_m = float(localize_sigma_m)
        self.act_sigma_m = float(act_sigma_m)
        self.rng = rng or np.random.default_rng(0)
        self.plan_wall_ms = plan_wall_ms
        self.localize_wall_ms = localize_wall_ms
        self.act_wall_ms = act_wall_ms
        self.finger_length_m = float(finger_length_m)
        self.nominal_view_distance_m = float(nominal_view_distance_m)

    # -- planning ----------------------------------------------------------
    def plan_task(self, obs: Observation) -> AstraPlanResponse:
        tin, tout, nimg, cache = prompts.plan_cost(obs)
        return AstraPlanResponse(
            plan=synthesise_plan(obs.prompt),
            tokens_in=tin,
            tokens_out=tout,
            images=nimg,
            cached_in=cache,
            wall_ms=self.plan_wall_ms,
        )

    # -- localisation ------------------------------------------------------
    def localize(
        self, obs: Observation, refs: Sequence[str]
    ) -> AstraLocalizeResponse:
        refs = list(refs) or ["target"]
        truth = self.oracle.frames()
        sigma = self._view_dependent_sigma(truth)
        frames: dict[str, Frame] = {}
        for name in refs:
            f = truth.get(name)
            if f is None:
                continue
            frames[name] = Frame(
                origin=f.origin + self.rng.normal(0.0, sigma, size=3),
                normal=f.normal,
                extent=f.extent,
                up=f.up,
                axis=f.axis,
            )
        obstacles = [
            np.asarray(p, dtype=np.float64) + self.rng.normal(0.0, sigma, size=3)
            for p in self.oracle.obstacles()
        ]
        tin, tout, nimg, cache = prompts.localize_cost(obs, refs)
        return AstraLocalizeResponse(
            frames=frames,
            obstacles=obstacles,
            # Confidence reflects viewing conditions, not the realised error --
            # a model does not know its own error. That is what makes
            # confidence a weak re-localisation trigger for the Astra-only
            # pipeline, and why the hybrid uses a Jev noul instead.
            confidence=float(
                np.clip(self.localize_sigma_m / max(sigma, 1e-6), 0.0, 1.0)
            ),
            tokens_in=tin,
            tokens_out=tout,
            images=nimg,
            cached_in=cache,
            wall_ms=self.localize_wall_ms,
        )

    def _view_dependent_sigma(self, truth: dict[str, Frame]) -> float:
        anchor = truth.get("target") or next(iter(truth.values()), None)
        if anchor is None:
            return self.localize_sigma_m
        d = float(np.linalg.norm(self.oracle.tcp_position() - anchor.origin))
        scale = float(np.clip(d / self.nominal_view_distance_m, 0.3, 2.0))
        return self.localize_sigma_m * scale

    # -- fine action -------------------------------------------------------
    def act_fine(
        self, obs: Observation, subgoal: Subgoal, goal: Waypoint
    ) -> AstraActResponse:
        tcp = self.oracle.tcp_position()
        anchor = self._truth_pose(subgoal, tcp, goal)
        jitter = self.rng.normal(0.0, self.act_sigma_m, size=3)
        tin, tout, nimg, cache = prompts.act_cost(obs, subgoal, goal)
        return AstraActResponse(
            waypoint=Waypoint(
                position=anchor + jitter,
                approach=goal.approach,
                closing=goal.closing,
                gripper=goal.gripper,
                torso_delta=goal.torso_delta,
            ),
            tokens_in=tin,
            tokens_out=tout,
            images=nimg,
            cached_in=cache,
            wall_ms=self.act_wall_ms,
        )

    def _truth_pose(
        self, subgoal: Subgoal, tcp: np.ndarray, fallback: Waypoint
    ) -> np.ndarray:
        """Where the subgoal's pose really is, per the simulator."""
        truth = self.oracle.frames()
        if subgoal.ref not in truth:
            return fallback.position
        wp = resolve(
            subgoal,
            truth,
            tcp_position=tcp,
            anchor=fallback,
            anchor_tcp=tcp,
            finger_length_m=self.finger_length_m,
        )
        return wp.position if wp is not None else fallback.position


# ---------------------------------------------------------------------------
# The planner stand-in.
#
# A real run asks `gpt-6-astra` for the decomposition (see AstraApiBackend).
# This function produces the plan such a call would return, keyed off the
# instruction strings RoboBenchMart actually emits, so the rest of the system
# can be exercised without an API key. It is deliberately kept in the *mock*
# backend and not in the pipeline: the pipelines must never contain a
# hard-coded decomposition, or the generality is lost.
#
# Tolerances below are the planner's guesses. `plan.calibrate` overwrites them
# with values the environment declares, and records that it did.
# ---------------------------------------------------------------------------
def synthesise_plan(instruction: str) -> TaskPlan:
    text = instruction.lower()
    articulated = any(w in text for w in ("fridge", "showcase", "door"))
    if articulated and "open" in text:
        return _door_plan(instruction, opening=True)
    if articulated and "close" in text:
        return _door_plan(instruction, opening=False)
    if "basket" in text:
        return _pick_and_place_plan(instruction, "basket", height=0.40, lift=0.30)
    if "floor" in text and "shelf" in text:
        return _pick_and_place_plan(instruction, "goal_site", height=0.18, lift=0.30)
    if "board" in text:
        return _pick_and_place_plan(instruction, "goal_site", height=0.18, lift=0.15)
    # Unrecognised: pick-and-place to the basket is the most common shape in
    # this benchmark. Failing silently here would hide that the planner did not
    # understand the task, so the caller sees a plan whose refs may not resolve.
    return _pick_and_place_plan(instruction, "basket", height=0.40, lift=0.30)


def _pick_and_place_plan(
    instruction: str, destination: str, *, height: float, lift: float
) -> TaskPlan:
    return TaskPlan(
        instruction=instruction,
        subgoals=[
            Subgoal("approach_item", GoalKind.STANDOFF, "target",
                    tolerance_mm=40.0, needs_pixels=False, gripper=GRIPPER_OPEN,
                    params={"distance": 0.18}),
            Subgoal("align_to_item", GoalKind.CONTACT, "target",
                    tolerance_mm=6.0, needs_pixels=True, gripper=GRIPPER_OPEN),
            Subgoal("close_gripper", GoalKind.SET_GRIPPER, gripper=GRIPPER_CLOSED),
            Subgoal("extract", GoalKind.WITHDRAW,
                    tolerance_mm=25.0, needs_pixels=False, gripper=GRIPPER_CLOSED,
                    params={"distance": 0.30}),
            Subgoal(f"over_{destination}", GoalKind.OVER, destination,
                    tolerance_mm=60.0, needs_pixels=False, gripper=GRIPPER_CLOSED,
                    params={"height": height, "torso_lift": lift}),
            Subgoal("release", GoalKind.SET_GRIPPER, gripper=GRIPPER_OPEN),
            Subgoal("settle", GoalKind.HOLD, params={"steps": 20}),
        ],
    )


def _door_plan(instruction: str, *, opening: bool) -> TaskPlan:
    """Articulated object: reach the handle, then swing about the hinge.

    Structurally unlike pick-and-place -- no free object is carried and the
    motion is rotational -- which is what makes it the test of whether the
    routing generalises or was fitted to one task.

    The swing is split into several short arcs. A gripper holding a handle has
    to stay on the hinge arc, and any coarse controller will chase a single
    large arc in a straight line and tear the grip off. That constraint belongs
    in the plan, not in the controller, which is the point of plans being data.
    """
    total = 75.0 if opening else -75.0
    n_swings = 3
    swings = [
        Subgoal(f"swing_{i + 1}", GoalKind.ARC, "hinge",
                tolerance_mm=30.0, needs_pixels=False, gripper=GRIPPER_CLOSED,
                params={"angle_deg": total / n_swings})
        for i in range(n_swings)
    ]
    return TaskPlan(
        instruction=instruction,
        subgoals=[
            Subgoal("approach_handle", GoalKind.STANDOFF, "handle",
                    tolerance_mm=40.0, needs_pixels=False, gripper=GRIPPER_OPEN,
                    params={"distance": 0.15}),
            Subgoal("align_to_handle", GoalKind.CONTACT, "handle",
                    tolerance_mm=6.0, needs_pixels=True, gripper=GRIPPER_OPEN),
            Subgoal("grip_handle", GoalKind.SET_GRIPPER, gripper=GRIPPER_CLOSED),
            *swings,
            Subgoal("release_handle", GoalKind.SET_GRIPPER, gripper=GRIPPER_OPEN),
            Subgoal("back_off", GoalKind.WITHDRAW,
                    tolerance_mm=40.0, needs_pixels=False, gripper=GRIPPER_OPEN,
                    params={"distance": 0.20}),
            Subgoal("settle", GoalKind.HOLD, params={"steps": 20}),
        ],
    )
