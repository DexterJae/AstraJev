"""Pipeline B -- Astra + Jev.

One `gpt-6-astra` response costs about as much as two thousand `jev-1.14`
queries. Everything below is an attempt to spend the frontier model only where
nothing cheaper will do, and the three places it is unavoidable are:

  planning       decomposing an instruction into subgoals
  grounding      turning "the Nivea Body Milk" plus a cluttered shelf image
                 into a 3D frame
  fine motion    correcting a pose to millimetres while looking at the thing

Jev takes the rest, in three roles matching the three its model card
describes:

  1. Coarse policy. A `choice` over seven generic directions. Jev reads a
     centimetre-quantised text state, so it cannot do better than its coarse
     step off that interface, and the coarse subgoals do not need better.
     Which subgoals those are comes from the generated plan, not from any
     hard-coded phase list, so this works on a plan it has never seen.

  2. Gate on the expensive model. A `noul` asking whether the stored
     references have gone stale. This is where most of the saving comes from:
     Astra alone must re-ground on a schedule, because *asking* the question
     costs the same frontier call as *answering* it. Jev answers it for
     1/2000th of the price, so grounding becomes need-based. A second noul
     decides when to hand over to fine motion.

  3. Ordinal Q-function over actions. A `score` rating Astra's proposed pose.
     Below threshold the pose is rejected and Astra is asked again -- bounded
     by `max_align_resamples`, because at $0.009 a correction an unbounded
     accept/reject loop would spend more on Astra than it saves.

Note what this pipeline does *not* have: any per-tick perception of its own.
Jev reads text, and the text is built from forward kinematics (free) plus the
references and obstacle list from the last grounding call. The coarse
controller dead-reckons, which is precisely why "is my belief stale?" is the
question worth paying for.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import verbalize
from ..backends.base import (
    QID_ALIGN_OK,
    QID_COARSE,
    QID_READY,
    QID_RETRACT_RISK,
    QID_STALE,
    Question,
)
from ..plan import TaskCapabilities
from ..senses import Senses
from ..types import GRIPPER_CLOSED, Observation, Waypoint
from ..verbalize import COARSE_ACTIONS
from .astra_only import AstraPipelineBase, PlanCache
from .base import RETRY, EpisodeContext, Geometry, PlanCursor, symbolic_state


class AstraJevPipeline(AstraPipelineBase):
    name = "astra_jev"

    def __init__(
        self,
        astra,
        jev,
        geom: Geometry,
        *,
        jev_precision_threshold_mm: float = 20.0,
        handoff_prob: float = 0.75,
        relocalize_prob: float = 0.6,
        max_align_resamples: int = 2,
        align_accept_score: float = 0.5,
        obstacle_risk_escalate: float = 0.5,
        obstacle_attention_m: float = 0.15,
        max_relocalizes_per_subgoal: int = 2,
        max_ticks: int = 200,
        plan_cache: Optional[PlanCache] = None,
        capabilities: Optional[TaskCapabilities] = None,
    ) -> None:
        super().__init__(
            astra, geom, max_ticks=max_ticks,
            plan_cache=plan_cache, capabilities=capabilities,
        )
        self.jev = jev
        self.jev_precision_threshold_mm = float(jev_precision_threshold_mm)
        self.handoff_prob = float(handoff_prob)
        self.relocalize_prob = float(relocalize_prob)
        self.max_align_resamples = int(max_align_resamples)
        self.align_accept_score = float(align_accept_score)
        self.obstacle_risk_escalate = float(obstacle_risk_escalate)
        self.obstacle_attention_m = float(obstacle_attention_m)
        self.max_relocalizes_per_subgoal = int(max_relocalizes_per_subgoal)
        self._n_jev = 0
        self._relocalizes: dict[int, int] = {}

    def reset(self, ctx: EpisodeContext) -> None:
        super().reset(ctx)
        self._n_jev = 0
        self._relocalizes = {}

    def call_counts(self) -> dict[str, int]:
        return {"astra": self._n_astra, "jev": self._n_jev}

    # -- the routing rule --------------------------------------------------
    def jev_owns(self, cursor: PlanCursor) -> bool:
        """Derived from the subgoal, never from its name.

        Three conditions, all read off the generated plan, none of which know
        which task is being solved:

          tolerance       coarser than the threshold;
          needs_pixels    false, i.e. the pose follows from stored coordinates
                          rather than from what is visible right now;
          path_constrained  false, i.e. the motion is free in space.

        The third was added after the door task. A coarse controller choosing
        among seven fixed directions cuts the chord of an arc, and a gripper
        holding a handle has to stay on the hinge radius -- so routing on
        precision alone sent the swings to Jev, the grip slipped on the first
        step, and the hybrid came out both less reliable and dearer than the
        baseline. Precision is not the only thing that decides who can drive.
        """
        sg = cursor.current
        return (
            sg.tolerance_mm >= self.jev_precision_threshold_mm
            and not sg.needs_pixels
            and not sg.path_constrained
        )

    def _ask_jev(self, ctx, questions, situation, hint, subgoal):
        r = self.jev.query(situation, questions, hint=hint)
        ctx.ledger.record(
            self.jev.name,
            "query:" + "+".join(q.id for q in questions),
            subgoal,
            r.tokens_in,
            0,
            forward_passes=r.forward_passes,
            wall_ms=r.wall_ms,
            reported_usd=getattr(r, "reported_cost_usd", None),
        )
        self._n_jev += 1
        return r

    # -- decision ----------------------------------------------------------
    def decide(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        cursor: PlanCursor,
    ) -> Optional[Waypoint]:
        self._refs.update(senses.free_frames)

        missing = [r for r in self._needed if r not in self._refs]
        if missing:
            # Unavoidable: nothing cheaper turns an instruction plus a
            # cluttered shelf into a 3D frame.
            self._localize(ctx, obs, cursor.current.name)
            return RETRY

        if self.jev_owns(cursor):
            return self._coarse_tick(ctx, obs, senses, cursor)
        return self._fine_tick(ctx, obs, senses, cursor)

    # -- coarse subgoals: Jev drives, and gates Astra ----------------------
    def _coarse_tick(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        cursor: PlanCursor,
    ) -> Optional[Waypoint]:
        goal = cursor.goal(self._refs, senses)
        if goal is None:
            return None
        if cursor.reached(senses, goal):
            return None

        sg = cursor.current
        frame = self._refs.get(sg.ref)
        normal = frame.normal if frame else -goal.approach
        obstacle_m = self.nearest_obstacle_m(senses)
        state = symbolic_state(cursor, senses, goal, normal, obstacle_m)
        step_m = self.geom.coarse_step(cursor.tolerance_m())

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
        # Worth asking only while the gripper is empty -- once something is
        # held its position follows from proprioception -- and only while there
        # is budget left to act on the answer. Asking a question whose answer
        # cannot be acted on is pure cost, and an unbounded re-grounding loop
        # is worse: a stuck gripper that keeps reporting "stale" would buy
        # frontier calls forever. This cap is the gate on the gate.
        budget_left = (
            self._relocalizes.get(cursor.index, 0) < self.max_relocalizes_per_subgoal
        )
        if not senses.holding_item and budget_left:
            questions.append(
                Question(
                    id=QID_STALE, kind="noul", instruction="",
                    proposition=(
                        "Stored reference position is too stale to act on; "
                        "re-localise."
                    ),
                )
            )
        # Asked on any coarse subgoal that immediately precedes a fine one --
        # the plan says where that is.
        if self._next_is_fine(cursor):
            questions.append(
                Question(
                    id=QID_READY, kind="noul", instruction="",
                    proposition=(
                        "Gripper is close enough to the goal that the precise "
                        "controller should take over."
                    ),
                )
            )
        # Asked whenever something could be knocked over. The generic
        # condition is "moving while something else is nearby".
        if obstacle_m < self.obstacle_attention_m:
            questions.append(
                Question(
                    id=QID_RETRACT_RISK, kind="noul", instruction="",
                    proposition=(
                        "Moving along the planned path would displace a "
                        "non-target item."
                    ),
                )
            )

        hint = {
            "state": state,
            "subgoal": sg,
            "step_m": step_m,
            "handoff_radius_m": self.geom.handoff_radius_m,
            "finger_length_m": self.geom.finger_length_m,
            # The pipeline knows where it *believes* the goal is. Whether that
            # belief is stale is what it is paying Jev to judge; the mock
            # backend resolves the truth from its own oracle.
            "believed_goal": goal.position,
            "obstacle_m": obstacle_m,
        }
        # One shared read of the state for every question above -- the property
        # the whole cost argument rests on.
        r = self._ask_jev(
            ctx, questions, verbalize.describe(state, obs.prompt), hint, sg.name
        )

        if QID_STALE in r.answers and r[QID_STALE].probability > self.relocalize_prob:
            self._relocalizes[cursor.index] = (
                self._relocalizes.get(cursor.index, 0) + 1
            )
            self._localize(ctx, obs, sg.name)
            return RETRY

        if QID_READY in r.answers and r[QID_READY].probability > self.handoff_prob:
            return None  # hand over to the next, fine subgoal

        if (
            QID_RETRACT_RISK in r.answers
            and r[QID_RETRACT_RISK].probability > self.obstacle_risk_escalate
        ):
            # Too tight to move coarsely. Buy one frontier correction to get
            # clear; counted like any other Astra call.
            return self._act_fine(ctx, obs, cursor, goal)

        choice = r[QID_COARSE].argmax()
        delta = verbalize.resolve_coarse_action(choice, state, step_m)
        if float(np.linalg.norm(delta)) < 1e-6:
            # Jev chose "hold" while the goal is not yet reached. Spend the
            # tick and ask again rather than declaring the subgoal done.
            return RETRY

        return Waypoint(
            position=senses.tcp_position + delta,
            approach=goal.approach,
            closing=goal.closing,
            gripper=GRIPPER_CLOSED if senses.holding_item else sg.gripper,
        )

    def _next_is_fine(self, cursor: PlanCursor) -> bool:
        """Is a fine subgoal coming up, skipping the free primitives?"""
        for sg in cursor.plan.subgoals[cursor.index + 1 :]:
            if sg.is_free:
                continue
            return sg.is_fine
        return False

    # -- fine subgoals: Astra proposes, Jev verifies -----------------------
    def _fine_tick(
        self,
        ctx: EpisodeContext,
        obs: Observation,
        senses: Senses,
        cursor: PlanCursor,
    ) -> Optional[Waypoint]:
        goal = cursor.goal(self._refs, senses)
        if goal is None:
            return None

        # Two separate axes, easy to conflate:
        #   `jev_owns`  decides *who drives* -- precision, pixels, constraint;
        #   `is_fine`   decides *how completion is judged* -- servo
        #               convergence for a millimetre subgoal, plain arrival
        #               for a coarse one.
        # A path-constrained subgoal with a coarse tolerance lands here
        # because Jev cannot drive it, but it is still judged by arrival, so
        # it needs the early-out that a truly fine subgoal must not have.
        if not cursor.is_fine() and cursor.reached(senses, goal):
            return None

        proposal = self._act_fine(ctx, obs, cursor, goal)
        sg = cursor.current

        # Verification is worth paying for where a wrong pose is expensive to
        # discover -- a missed grasp -- not where arrival is self-evident.
        if not cursor.is_fine():
            return proposal

        for _ in range(self.max_align_resamples):
            frame = self._refs.get(sg.ref)
            normal = frame.normal if frame else -goal.approach
            state = symbolic_state(
                cursor, senses, goal, normal, self.nearest_obstacle_m(senses)
            )
            q = Question(
                id=QID_ALIGN_OK,
                kind="score",
                instruction="Quality of the proposed gripper pose for this subgoal?",
            )
            hint = {
                "state": state,
                "subgoal": sg,
                "proposal": proposal,
                "step_m": self.geom.coarse_step(cursor.tolerance_m()),
                "finger_length_m": self.geom.finger_length_m,
            }
            r = self._ask_jev(
                ctx, [q], verbalize.describe(state, obs.prompt), hint, sg.name
            )
            if r[QID_ALIGN_OK].score_fraction() >= self.align_accept_score:
                return proposal
            # Rejected: the cheap verifier expects this pose to miss. One more
            # frontier correction beats executing a pose we expect to fail.
            proposal = self._act_fine(ctx, obs, cursor, goal)

        return proposal
