"""Pipeline B -- Astra + Jev.

The division of labour, and why each piece lands where it does:

  Jev drives the coarse phases (APPROACH, RETRACT, PLACE).
      These have 25-60 mm tolerances. Jev reads a centimetre-quantised text
      description of the scene and returns a choice over seven coarse
      directions. It cannot do better than 4 cm steps off that interface, and
      it does not need to.

  Jev gates the expensive model (`belief_stale`, `ready_for_fine`).
      This is where most of the saving comes from, and it is the mechanism
      worth defending in a paper. Astra alone must re-localise on a cadence
      because asking "has my target belief drifted?" costs the same MLLM
      forward pass as answering it. Jev answers that question as a noul for
      roughly a tenth of the price, so Astra-Global is invoked *on demand*
      instead of on a schedule.

  Astra-Local owns the fine phases (ALIGN, GRASP).
      6 mm tolerance on a cluttered shelf board. Never delegated: the success
      criterion voids the episode if a neighbouring item is displaced, so a
      centimetre of error here is not a slower grasp, it is a failed one.

  Jev scores Astra's proposal (`align_ok`).
      Jev as an ordinal Q-function over actions. A proposal that scores below
      threshold is rejected and Astra is asked again, bounded by
      `max_align_resamples`. One cheap verification call can save an expensive
      wasted execution -- but the gate is budgeted, because an unbounded
      accept/reject loop would spend more on Astra than it saves.

Every tick in a coarse phase costs exactly one Jev call (all questions batched
into a single forward pass, per arXiv:2610.11692) plus one cheap detector pass,
because Jev reads text and not pixels and somebody has to produce the text.
The detector is charged explicitly; leaving it out would be the easiest way to
overstate this pipeline's advantage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import geometry, prompts, verbalize
from ..backends.base import (
    QID_ALIGN_OK,
    QID_COARSE,
    QID_READY,
    QID_RETRACT_RISK,
    QID_STALE,
    Question,
)
from ..senses import Senses
from ..types import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    Observation,
    Phase,
    PHASE_SPEC,
    TargetEstimate,
    Waypoint,
)
from ..verbalize import COARSE_ACTIONS
from .base import (
    RETRY,
    EpisodeContext,
    Geometry,
    PhaseMachine,
    Pipeline,
    symbolic_state,
)

COARSE_PHASES = (Phase.APPROACH, Phase.RETRACT, Phase.PLACE)


class AstraJevPipeline(Pipeline):
    name = "astra_jev"

    def __init__(
        self,
        astra,
        jev,
        geom: Geometry,
        *,
        detector_name: str = "detector",
        jev_precision_threshold_mm: float = 20.0,
        handoff_prob: float = 0.75,
        relocalize_prob: float = 0.6,
        max_align_resamples: int = 2,
        align_accept_score: float = 0.5,
        retract_risk_escalate: float = 0.5,
        chunk_len: int = 1,
        max_ticks: int = 160,
    ) -> None:
        super().__init__(geom, max_ticks=max_ticks)
        self.astra = astra
        self.jev = jev
        self.detector_name = detector_name
        self.jev_precision_threshold_mm = float(jev_precision_threshold_mm)
        self.handoff_prob = float(handoff_prob)
        self.relocalize_prob = float(relocalize_prob)
        self.max_align_resamples = int(max_align_resamples)
        self.align_accept_score = float(align_accept_score)
        self.retract_risk_escalate = float(retract_risk_escalate)
        self.chunk_len = int(chunk_len)

        self._target: Optional[TargetEstimate] = None
        self._nearest_neighbour_m = 0.5
        self._n_astra = 0
        self._n_jev = 0

    def reset(self, ctx: EpisodeContext) -> None:
        self._target = None
        self._nearest_neighbour_m = 0.5
        self._n_astra = 0
        self._n_jev = 0

    def call_counts(self) -> dict[str, int]:
        return {"astra": self._n_astra, "jev": self._n_jev}

    def jev_owns(self, phase: Phase) -> bool:
        """Phase assignment is derived from tolerance, not hard-coded.

        A phase goes to Jev only if its positional tolerance is coarser than
        the threshold *and* it does not require reasoning over raw pixels.
        """
        spec = PHASE_SPEC[phase]
        return (
            spec["precision_mm"] >= self.jev_precision_threshold_mm
            and not spec["needs_pixels"]
        )

    # -- charged model calls ----------------------------------------------
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
        self._n_astra += 1

    def _plan(
        self, ctx: EpisodeContext, obs: Observation, phase: Phase, goal: Waypoint
    ) -> Waypoint:
        r = self.astra.plan_local(obs, self._target, phase, goal, self.chunk_len)
        ctx.ledger.record(
            self.astra.name_local,
            "plan_local",
            phase,
            r.tokens_in,
            r.tokens_out,
            images=r.images,
            wall_ms=r.wall_ms,
        )
        self._n_astra += 1
        return r.waypoints[-1]

    def _detect(self, ctx: EpisodeContext, obs: Observation, phase: Phase) -> None:
        """Cheap perception that produces the text Jev reads."""
        tin, tout, nimg = prompts.detector_cost(obs)
        ctx.ledger.record(
            self.detector_name, "detect", phase, tin, tout, images=nimg, wall_ms=8.0
        )
        # In the mock path the detector's numbers reach the pipeline through
        # the oracle-backed senses; with a real detector, wire its outputs in
        # here. The *cost* of the call is what this experiment measures.
        if getattr(ctx.senses, "oracle", None) is not None:
            self._nearest_neighbour_m = ctx.senses.oracle.nearest_neighbour_distance()

    def _ask_jev(
        self, ctx: EpisodeContext, questions: list[Question], situation: str, hint: dict
    ):
        r = self.jev.query(situation, questions, hint=hint)
        ctx.ledger.record(
            self.jev.name,
            "query:" + "+".join(q.id for q in questions),
            hint["state"].phase,
            r.tokens_in,
            0,
            forward_passes=r.forward_passes,
            wall_ms=r.wall_ms,
        )
        self._n_jev += 1
        return r

    # -- decision ----------------------------------------------------------
    def decide(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        fsm: PhaseMachine,
    ) -> Optional[Waypoint]:
        if fsm.phase == Phase.LOCATE:
            # Unavoidable: nothing but an MLLM turns the instruction plus a
            # cluttered shelf into a 3D target.
            self._localize(ctx, obs, fsm.phase)
            return None

        if self.jev_owns(fsm.phase):
            return self._coarse_tick(ctx, obs, senses, fsm)
        return self._fine_tick(ctx, obs, senses, fsm)

    # -- coarse phases: Jev drives, and gates Astra ------------------------
    def _coarse_tick(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        fsm: PhaseMachine,
    ) -> Optional[Waypoint]:
        self._detect(ctx, obs, fsm.phase)

        state = symbolic_state(
            fsm, senses, self._target, self.geom, self._nearest_neighbour_m
        )
        goal = fsm.goal(self._target, senses)
        if fsm.reached(senses, goal):
            return None

        questions = [
            Question(
                id=QID_COARSE,
                kind="choice",
                instruction=(
                    "Best single coarse gripper motion toward goal, "
                    "without touching a non-target item?"
                ),
                options=COARSE_ACTIONS,
            )
        ]
        # Only worth asking before the grasp. Once the item is in the gripper
        # its position is known from proprioception, so re-localising it would
        # be paying an MLLM for something free.
        if not senses.holding_item:
            questions.append(
                Question(
                    id=QID_STALE,
                    kind="noul",
                    instruction="",
                    proposition=(
                        "Stored target position is too stale to grasp from; "
                        "re-localise."
                    ),
                )
            )
        if fsm.phase == Phase.APPROACH:
            questions.append(
                Question(
                    id=QID_READY,
                    kind="noul",
                    instruction="",
                    proposition=(
                        "Gripper is close enough to standoff that the precise "
                        "grasp controller should take over."
                    ),
                )
            )
        if fsm.phase == Phase.RETRACT:
            questions.append(
                Question(
                    id=QID_RETRACT_RISK,
                    kind="noul",
                    instruction="",
                    proposition=(
                        "Pulling straight out along the shelf normal would "
                        "displace a non-target item."
                    ),
                )
            )

        step_m = self.geom.coarse_step(fsm.tolerance_m())
        hint = {
            "state": state,
            "step_m": step_m,
            "handoff_radius_m": self.geom.handoff_radius_m,
            "finger_length_m": self.geom.finger_length_m,
        }
        # One forward pass for every question above -- that is the property
        # the whole cost argument rests on.
        r = self._ask_jev(
            ctx, questions, verbalize.describe(state, obs.prompt), hint
        )

        # Need-based re-localisation, instead of Astra's cadence-based one.
        if (
            QID_STALE in r.answers
            and r[QID_STALE].probability > self.relocalize_prob
        ):
            self._localize(ctx, obs, fsm.phase)
            return RETRY  # re-decide next tick against the refreshed target

        if (
            fsm.phase == Phase.APPROACH
            and QID_READY in r.answers
            and r[QID_READY].probability > self.handoff_prob
        ):
            return None  # hand off to ALIGN

        if (
            fsm.phase == Phase.RETRACT
            and QID_RETRACT_RISK in r.answers
            and r[QID_RETRACT_RISK].probability > self.retract_risk_escalate
        ):
            # Too tight to pull straight out. Buy one Astra-Local plan to get
            # out of the clutter; this is the escalation path, and it is
            # counted like any other Astra call.
            return self._plan(ctx, obs, fsm.phase, goal)

        choice = r[QID_COARSE].argmax()
        delta = verbalize.resolve_coarse_action(choice, state, step_m)
        if float(np.linalg.norm(delta)) < 1e-6:
            # Jev chose "hold" while the goal is not yet reached. Spend the
            # tick and ask again rather than declaring the phase done.
            return RETRY

        approach, closing = geometry.approach_and_closing(self._target.shelf_normal)
        gripper = GRIPPER_CLOSED if senses.holding_item else GRIPPER_OPEN
        if fsm.phase == Phase.PLACE:
            approach = np.array([0.0, 0.0, -1.0])
            closing = np.array([0.0, 1.0, 0.0])
        return Waypoint(
            position=senses.tcp_position + delta,
            approach=approach,
            closing=closing,
            gripper=gripper,
        )

    # -- fine phases: Astra plans, Jev verifies ----------------------------
    def _fine_tick(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        fsm: PhaseMachine,
    ) -> Optional[Waypoint]:
        # No belief-based early-out here: ALIGN is a 6 mm phase and the
        # believed goal is centimetres off. The driver advances the phase when
        # Astra-Local's own correction falls below tolerance.
        goal = fsm.goal(self._target, senses)
        proposal = self._plan(ctx, obs, fsm.phase, goal)

        for _ in range(self.max_align_resamples):
            state = symbolic_state(
                fsm, senses, self._target, self.geom, self._nearest_neighbour_m
            )
            q = Question(
                id=QID_ALIGN_OK,
                kind="score",
                instruction="Quality of the proposed gripper pose as a clean grasp?",
            )
            hint = {
                "state": state,
                "proposal": proposal,
                "step_m": self.geom.coarse_step(fsm.tolerance_m()),
                "finger_length_m": self.geom.finger_length_m,
            }
            r = self._ask_jev(
                ctx, [q], verbalize.describe(state, obs.prompt), hint
            )
            if r[QID_ALIGN_OK].score_fraction() >= self.align_accept_score:
                return proposal
            # Rejected: the cheap verifier thinks this pose would miss. Ask
            # Astra once more rather than execute a pose we expect to fail.
            proposal = self._plan(ctx, obs, fsm.phase, goal)

        return proposal
