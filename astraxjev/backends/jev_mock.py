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
Nothing here knows about any particular task: the noul truths are generic
spatial predicates over the current subgoal.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..oracle import Oracle
from ..plan import resolve
from ..tokens import count_text_tokens
from ..types import SymbolicState
from ..verbalize import COARSE_ACTIONS, resolve_coarse_action
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
        step_m = float(hint.get("step_m", 0.12))
        to_goal = state.to_goal()
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
            # current coarse goal.
            radius = float(hint.get("handoff_radius_m", 0.06))
            return _soft_below(float(np.linalg.norm(state.to_goal())), radius)

        if qid == QID_STALE:
            # True when the believed goal has drifted from where it really is
            # by more than a coarse step. The caller supplies only its own
            # belief -- judging that belief is what it is paying for -- so the
            # truth is resolved here, from the oracle.
            believed = hint.get("believed_goal")
            subgoal = hint.get("subgoal")
            if believed is None or subgoal is None:
                return 0.0
            truth = self.oracle.frames()
            if subgoal.ref not in truth:
                return 0.0
            ideal = resolve(
                subgoal,
                truth,
                tcp_position=np.asarray(state.tcp_position),
                anchor=None,
                finger_length_m=float(hint.get("finger_length_m", 0.03)),
            )
            if ideal is None:
                return 0.0
            drift = float(np.linalg.norm(np.asarray(believed) - ideal.position))
            return _soft_above(drift, float(hint.get("step_m", 0.12)))

        if qid == QID_RETRACT_RISK:
            # True when something that must not move is close enough to the
            # gripper that the planned path is likely to drag it along. The
            # distance comes from the caller's dead-reckoned belief, which is
            # what a real deployment would have; the oracle supplies the truth
            # the answer is noised around.
            believed = float(hint.get("obstacle_m", 10.0))
            true_d = self.oracle.nearest_neighbour_distance()
            return _soft_below(min(believed, true_d), 0.08)

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
        subgoal = hint.get("subgoal")
        if proposal is None or subgoal is None:
            return None
        # How good is Astra's proposed pose, judged against the real reference?
        # This is Jev used as an ordinal Q-function over actions -- the second
        # role its paper describes -- and it is what lets the hybrid reject a
        # bad chunk for the price of a Jev call instead of discovering the
        # failure after executing it.
        truth = self.oracle.frames()
        if subgoal.ref not in truth:
            return None
        ideal = resolve(
            subgoal,
            truth,
            tcp_position=np.asarray(hint["state"].tcp_position),
            anchor=proposal,
            finger_length_m=float(hint.get("finger_length_m", 0.03)),
        )
        if ideal is None:
            return None
        err = float(np.linalg.norm(proposal.position - ideal.position))
        tol = max(subgoal.tolerance_m, 1e-4)
        # At the tolerance the score is ~0.5; at zero error it is 1.
        return float(np.clip(1.0 - err / (2.0 * tol), 0.0, 1.0))


def _soft_below(x: float, threshold: float, width: float = 0.25) -> float:
    """Smooth indicator for x < threshold, so answers look calibrated."""
    return float(1.0 / (1.0 + np.exp((x - threshold) / (width * threshold + 1e-9))))


def _soft_above(x: float, threshold: float, width: float = 0.25) -> float:
    return 1.0 - _soft_below(x, threshold, width)
