"""Mock Jev backend.

Reproduces the three answer forms of arXiv:2610.11692 and, critically, the cost
structure: one forward pass answers the whole question batch and no tokens are
generated.

The accuracy model is deliberately unkind to Jev. A choice answer is the
distance-reducing action with probability `choice_accuracy` and a uniformly
random legal action otherwise; noul probabilities are the true value plus
Gaussian noise. If the hybrid still wins at 88% coarse-action accuracy, the
result does not rest on Jev being near-perfect.

Answers are synthesised from the oracle, never from the billed prompt, so the
token counts this backend reports are exactly what a real deployment would pay.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geometry import distance, grasp_waypoint, standoff_waypoint
from ..oracle import Oracle
from ..tokens import count_text_tokens
from ..types import SymbolicState, TargetEstimate
from ..verbalize import COARSE_ACTIONS, PHASE_GOAL, resolve_coarse_action
from .base import (
    DEFAULT_RUBRIC,
    QID_ALIGN_OK,
    QID_READY,
    QID_RETRACT_RISK,
    QID_STALE,
    Answer,
    JevResponse,
    Question,
    render_request,
)


class MockJevBackend:
    name = "jev"

    def __init__(
        self,
        oracle: Oracle,
        *,
        choice_accuracy: float = 0.88,
        noul_noise: float = 0.08,
        rng: np.random.Generator | None = None,
        wall_ms: float = 12.0,
    ) -> None:
        self.oracle = oracle
        self.choice_accuracy = float(choice_accuracy)
        self.noul_noise = float(noul_noise)
        self.rng = rng or np.random.default_rng(0)
        self.wall_ms = wall_ms

    # -- Jev interface -----------------------------------------------------
    def query(
        self,
        situation: str,
        questions: Sequence[Question],
        *,
        hint: Optional[dict] = None,
    ) -> JevResponse:
        hint = hint or {}
        answers: dict[str, Answer] = {}
        for q in questions:
            if q.kind == "choice":
                answers[q.id] = self._choice(q, hint)
            elif q.kind == "noul":
                answers[q.id] = self._noul(q, hint)
            else:
                answers[q.id] = self._score(q, hint)

        return JevResponse(
            answers=answers,
            tokens_in=count_text_tokens(render_request(situation, questions)),
            # The paper's central cost claim: one pass for the whole batch.
            forward_passes=1,
            wall_ms=self.wall_ms,
        )

    # -- answer forms ------------------------------------------------------
    def _choice(self, q: Question, hint: dict) -> Answer:
        options = list(q.options or COARSE_ACTIONS)
        best = self._best_coarse_action(options, hint)
        picked = (
            best
            if self.rng.random() < self.choice_accuracy
            else options[int(self.rng.integers(len(options)))]
        )
        p_main = 0.55 + 0.35 * self.rng.random()
        rest = (1.0 - p_main) / max(1, len(options) - 1)
        dist = {o: rest for o in options}
        dist[picked] = p_main
        return Answer(id=q.id, kind="choice", distribution=dist)

    def _best_coarse_action(self, options: Sequence[str], hint: dict) -> str:
        state: SymbolicState | None = hint.get("state")
        if state is None:
            return options[0]
        step_m = float(hint.get("step_m", 0.04))
        goal_name = PHASE_GOAL.get(state.phase, "standoff")
        to_goal = state.vector_to(goal_name)
        best, best_d = options[0], float("inf")
        for o in options:
            try:
                delta = resolve_coarse_action(o, state, step_m)
            except ValueError:
                continue
            d = float(np.linalg.norm(to_goal - delta))
            if d < best_d:
                best, best_d = o, d
        return best

    def _noul(self, q: Question, hint: dict) -> Answer:
        truth = self._noul_truth(q.id, hint)
        p = float(np.clip(truth + self.rng.normal(0.0, self.noul_noise), 0.0, 1.0))
        return Answer(id=q.id, kind="noul", probability=p)

    def _noul_truth(self, qid: str, hint: dict) -> float:
        state: SymbolicState | None = hint.get("state")
        if state is None:
            return 0.5

        if qid == QID_READY:
            # True when the gripper is inside the handoff ball around the
            # pre-grasp standoff point.
            radius = float(hint.get("handoff_radius_m", 0.06))
            d = float(np.linalg.norm(state.vector_to("standoff")))
            return _soft_below(d, radius)

        if qid == QID_STALE:
            # True when the believed target has drifted from the real one by
            # more than the coarse step. This is the question that lets the
            # hybrid call Astra-Global on demand instead of on a cadence.
            believed = np.asarray(state.target_position, dtype=np.float64)
            real = self.oracle.target_position()
            drift = float(np.linalg.norm(believed - real))
            return _soft_above(drift, float(hint.get("step_m", 0.04)))

        if qid == QID_RETRACT_RISK:
            # True when a non-target item is close enough to the gripper that
            # pulling straight out is likely to drag it along.
            d = self.oracle.nearest_neighbour_distance()
            return _soft_below(d, 0.08)

        return 0.5

    def _score(self, q: Question, hint: dict) -> Answer:
        rubric = [r for r in (q.rubric or DEFAULT_RUBRIC) if r != "unknown"]
        truth = self._score_truth(q.id, hint)
        if truth is None:
            return Answer(id=q.id, kind="score", score="unknown")
        noisy = float(np.clip(truth + self.rng.normal(0.0, self.noul_noise), 0.0, 1.0))
        idx = int(round(noisy * (len(rubric) - 1)))
        return Answer(id=q.id, kind="score", score=rubric[idx])

    def _score_truth(self, qid: str, hint: dict) -> Optional[float]:
        if qid != QID_ALIGN_OK:
            return None
        proposal = hint.get("proposal")
        if proposal is None:
            return None
        # How good is Astra's proposed grasp pose, judged against the real
        # item? This is Jev used as an ordinal Q-function over actions -- the
        # second role its paper describes -- and it is what lets the hybrid
        # reject a bad chunk for the price of a Jev call instead of discovering
        # the failure after executing it.
        truth_target = TargetEstimate(
            position=self.oracle.target_position(),
            extent=self.oracle.target_extent(),
            shelf_normal=self.oracle.shelf_normal(),
        )
        ideal = grasp_waypoint(truth_target, float(hint.get("finger_length_m", 0.03)))
        err = distance(proposal, ideal)
        # 0 mm -> 1.0, 20 mm -> ~0.0
        return float(np.clip(1.0 - err / 0.02, 0.0, 1.0))


def _soft_below(x: float, threshold: float, width: float = 0.25) -> float:
    """Smooth indicator for x < threshold, so answers look calibrated."""
    return float(1.0 / (1.0 + np.exp((x - threshold) / (width * threshold + 1e-9))))


def _soft_above(x: float, threshold: float, width: float = 0.25) -> float:
    return 1.0 - _soft_below(x, threshold, width)
