"""Backend protocols.

Two model families sit behind these interfaces:

Astra (arXiv:2506.06205) -- a hierarchical pair.
  * Astra-Global: a multimodal LLM. Given camera images and the language
    instruction it does goal localisation. This is the only component that can
    turn pixels + "pick the Nivea Body Milk" into a 3D target, so neither
    pipeline can avoid calling it.
  * Astra-Local: a small multitask network over a 4D spatial-temporal encoder.
    In the original paper it does local path planning for a *base*; here it is
    used as the local planner in end-effector space, which is the adaptation
    this project makes and the one to state plainly in any write-up.

Jev (arXiv:2610.11692) -- a closed decision model. It consumes a *textual*
situation plus a list of typed questions and returns calibrated answers for all
of them in ONE forward pass, generating no tokens. Three answer forms:
  choice  -> probability distribution over a declared option set
  score   -> a point on a declared ordinal rubric
  noul    -> a probability in [0, 1] for an asserted proposition

The single-forward-pass property is the whole reason the hybrid can be cheap, so
`JevBackend.query` takes a *batch* of questions by design: a pipeline that asks
Jev three things on one tick must pay for one call, not three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, Sequence, runtime_checkable

from ..types import Observation, Phase, SymbolicState, TargetEstimate, Waypoint

QuestionKind = Literal["choice", "score", "noul"]

DEFAULT_RUBRIC = ("zero", "low", "medium", "high", "very_high", "unknown")

#: Numeric reading of the ordinal rubric, for thresholding only. The Jev paper
#: is explicit that score answers are ordinal and do not support arithmetic, so
#: these values are used for comparisons and never summed or averaged.
RUBRIC_ORDER = {name: i for i, name in enumerate(DEFAULT_RUBRIC[:-1])}
RUBRIC_ORDER["unknown"] = -1


@dataclass
class Question:
    id: str
    kind: QuestionKind
    instruction: str
    options: Optional[Sequence[str]] = None  # choice only
    rubric: Sequence[str] = DEFAULT_RUBRIC  # score only
    proposition: str = ""  # noul only

    def render(self) -> str:
        if self.kind == "choice":
            opts = ", ".join(self.options or ())
            return f"[choice:{self.id}] {self.instruction} options=({opts})"
        if self.kind == "score":
            return (
                f"[score:{self.id}] {self.instruction} "
                f"rubric=({', '.join(self.rubric)})"
            )
        return f"[noul:{self.id}] {self.proposition}"


@dataclass
class Answer:
    id: str
    kind: QuestionKind
    distribution: dict[str, float] = field(default_factory=dict)  # choice
    score: str = "unknown"  # score
    probability: float = 0.0  # noul

    def argmax(self) -> str:
        if not self.distribution:
            raise ValueError(f"answer {self.id} has no distribution")
        return max(self.distribution.items(), key=lambda kv: kv[1])[0]

    def top_prob(self) -> float:
        return max(self.distribution.values()) if self.distribution else 0.0

    def score_rank(self) -> int:
        return RUBRIC_ORDER.get(self.score, -1)

    def score_fraction(self) -> float:
        """Normalised position on the rubric, for threshold comparisons."""
        r = self.score_rank()
        if r < 0:
            return 0.0
        return r / (len(DEFAULT_RUBRIC) - 2)


@dataclass
class JevResponse:
    answers: dict[str, Answer]
    tokens_in: int
    forward_passes: int
    wall_ms: float

    def __getitem__(self, qid: str) -> Answer:
        return self.answers[qid]


@dataclass
class AstraGlobalResponse:
    target: TargetEstimate
    tokens_in: int
    tokens_out: int
    images: int
    wall_ms: float


@dataclass
class AstraLocalResponse:
    waypoints: list[Waypoint]
    tokens_in: int
    tokens_out: int
    images: int
    wall_ms: float
    #: Astra-Local's own estimate of how well it can service the request.
    #: Used by the Astra-only pipeline to decide whether to re-localise.
    confidence: float = 1.0


@runtime_checkable
class AstraBackend(Protocol):
    name_global: str
    name_local: str

    def localize(self, obs: Observation) -> AstraGlobalResponse:
        """Astra-Global: pixels + instruction -> 3D target estimate."""
        ...

    def plan_local(
        self,
        obs: Observation,
        target: TargetEstimate,
        phase: Phase,
        goal: Waypoint,
        chunk_len: int,
    ) -> AstraLocalResponse:
        """Astra-Local: pixels + state + goal -> end-effector waypoint chunk."""
        ...


#: Canonical question ids. Both pipelines and both Jev backends agree on
#: these so a mock run and a real run ask literally the same questions.
QID_COARSE = "coarse_action"
QID_READY = "ready_for_fine"
QID_STALE = "belief_stale"
QID_RETRACT_RISK = "retract_risk"
QID_ALIGN_OK = "align_ok"


@runtime_checkable
class JevBackend(Protocol):
    name: str

    def query(
        self,
        situation: str,
        questions: Sequence[Question],
        *,
        hint: Optional[dict] = None,
    ) -> JevResponse:
        """Answer every question in one forward pass.

        `hint` carries non-privileged context the caller already has (the
        symbolic state, the coarse step size, a proposed waypoint). Real
        backends ignore it entirely -- they see only `situation` and
        `questions`, which is also all that is billed. Mock backends use it,
        together with their oracle, to synthesise calibrated answers. Keeping
        it out of the billed text is what stops the mock from accidentally
        making the hybrid look cheaper than it would be in deployment.
        """
        ...


def render_request(situation: str, questions: Sequence[Question]) -> str:
    """The exact text billed for a Jev call.

    Kept here so that the HTTP backend and the mock backend charge for byte-for
    byte the same prompt -- otherwise the mock run's cost numbers would not
    transfer to a real deployment, which is the whole point of measuring them.
    """
    lines = ["situation:", situation, "questions:"]
    lines.extend(q.render() for q in questions)
    return "\n".join(lines)
