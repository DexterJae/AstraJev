"""Backend protocols.

Two real, hosted models sit behind these interfaces.

**Astra** -- `gpt-6-astra`, a frontier vision LLM. $10 per million input
tokens and $50 per million *output* tokens, so the number of responses it
produces, not the number of images it is shown, is what the bill tracks. It
does three jobs here, each a separate call:

  plan_task   decompose the instruction into an ordered list of subgoals
  localize    ground every reference the plan names, plus nearby obstacles
  act_fine    correct a commanded end-effector pose by servoing on the
              current wrist view -- the millimetre work

Nothing cheaper can do the first two: turning "pick the Nivea Body Milk" plus
a cluttered shelf image into a 3D frame is exactly what a frontier VLM is for.

**Jev** -- `jev-1.14`, a closed decision model. $0.042 per million input
tokens, output free. It consumes a *textual* situation plus a list of typed
questions and returns calibrated answers for all of them in ONE shared read of
the state. Four answer forms:

  choice  a probability distribution over a declared option set
  score   a probability-weighted position on a declared ordinal scale
  noul    a probability in [0, 1] for an asserted proposition
  box     image coordinates (not used here)

One Astra call costs about as much as two thousand Jev calls. That ratio, not
any architectural subtlety, is what the hybrid pipeline exploits, and the
single-shared-read property is why `JevBackend.query` takes a *batch*: a
pipeline that asks Jev three things on one tick must pay for one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, Sequence, runtime_checkable

from ..plan import Frame, Subgoal, TaskPlan  # noqa: F401
from ..types import Observation, Waypoint

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
    score: str = "unknown"  # score, as a rubric level
    probability: float = 0.0  # noul
    #: Score as a fraction of the rubric. The real Jev returns a numeric,
    #: probability-weighted score that can land between levels, so the
    #: fraction is carried directly when available rather than being recovered
    #: from the nearest level name.
    numeric_score: Optional[float] = None

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
        if self.numeric_score is not None:
            return float(self.numeric_score)
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
    #: The provider's own cost figure for this call, in USD, when it supplies
    #: one. Authoritative: it replaces our computed figure in the ledger, so
    #: for Jev the cost side of the comparison is measured rather than assumed.
    reported_cost_usd: Optional[float] = None

    def __getitem__(self, qid: str) -> Answer:
        return self.answers[qid]


@dataclass
class AstraPlanResponse:
    """Astra-Global acting as a planner.

    Charged to both pipelines, once per distinct instruction. Generating the
    decomposition instead of hard-coding it is the price of generality, and it
    is a real price -- this call is as expensive as several localisations.
    """

    plan: TaskPlan
    tokens_in: int = 0
    tokens_out: int = 0
    images: int = 0
    #: Input tokens the provider served from its prompt cache, billed at a
    #: tenth of the fresh rate on the frontier model.
    cached_in: int = 0
    wall_ms: float = 0.0
    #: The provider's own USD figure, when the API returns one.
    reported_cost_usd: Optional[float] = None


@dataclass
class AstraLocalizeResponse:
    """Every reference the plan names, grounded in one call."""

    frames: dict[str, Frame]
    #: Centres of nearby items that must not be displaced. Returned by the
    #: same call, so the coarse controller needs no per-tick perception of its
    #: own: it dead-reckons from here until Jev judges the belief stale.
    obstacles: list = field(default_factory=list)
    confidence: float = 1.0
    tokens_in: int = 0
    tokens_out: int = 0
    images: int = 0
    cached_in: int = 0
    wall_ms: float = 0.0
    reported_cost_usd: Optional[float] = None


@dataclass
class AstraActResponse:
    """One fine-motion correction."""

    waypoint: Waypoint
    tokens_in: int = 0
    tokens_out: int = 0
    images: int = 0
    cached_in: int = 0
    wall_ms: float = 0.0
    reported_cost_usd: Optional[float] = None


#: Canonical question ids. Both pipelines and both Jev backends agree on
#: these so a mock run and a real run ask literally the same questions.
QID_COARSE = "coarse_action"
QID_READY = "ready_for_fine"
QID_STALE = "belief_stale"
QID_RETRACT_RISK = "retract_risk"
QID_ALIGN_OK = "align_ok"


@runtime_checkable
class AstraBackend(Protocol):
    """The expensive model. One name, three jobs."""

    name: str

    def plan_task(self, obs: Observation) -> AstraPlanResponse:
        """pixels + instruction -> an ordered list of subgoals."""
        ...

    def localize(
        self, obs: Observation, refs: Sequence[str]
    ) -> AstraLocalizeResponse:
        """pixels + instruction -> a frame per named reference, plus obstacles."""
        ...

    def act_fine(
        self, obs: Observation, subgoal: Subgoal, goal: Waypoint
    ) -> AstraActResponse:
        """pixels + commanded pose -> the corrected pose. Millimetre work."""
        ...


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
