"""Real Jev client.

Jev is a hosted, closed model: no weights, so this is the only way to run the
hybrid pipeline for real. Two routes, same request shape:

  OpenRouter  POST https://openrouter.ai/api/alpha/decisions
              model "typesafe/jev-1.14", key in OPENROUTER_API_KEY
  TypeSafe    POST https://api.typesafe.ai/v1/systemone
              model "jev-latest", key in TYPESAFE_API_KEY (waitlisted)

Request shape (verified against the published API docs, 2026-11):

    {"model": "...",
     "state": "<text>"  |  [ "<text>", {"type": "image_url", "image_url": {...}} ],
     "questions": {
        "<id>": {"type": "choice", "instructions": "...", "options": [...]},
        "<id>": {"type": "noul",   "instructions": "..."},
        "<id>": {"type": "score",  "instructions": "...", "levels": [...]}}}

Response shape:

    {"<id>": {"type": "choice", "choice": "left",
              "probabilities": {...}, "confidence": 0.9},
     "<id>": {"type": "noul",   "noul": 0.98},
     "<id>": {"type": "score",  "score": 1.21, "legend": [...],
              "confidence": 0.8},
     "usage": {"input_tokens": 183, "cost": 1.4994e-05}}

Two things about this that matter for the experiment:

  * `usage.cost` is the provider's own figure in USD, so for Jev the cost side
    of the comparison stops being an assumption entirely. The ledger records
    it as authoritative and keeps the locally computed figure alongside, so a
    divergence between the two is visible rather than silent.

  * All questions in one call share a single read of the state, which is the
    property the whole cost argument rests on. The batch interface is
    therefore not an optimisation here; it is the thing being measured.

Known Jev limitations worth keeping in mind when writing questions
(from the published model card): it is unreliable at counting, it treats dates
as plain text, and it needs bounded answer spaces. The questions this project
asks are comparisons and bounded choices, which is inside that envelope.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence

from ..tokens import count_text_tokens
from .base import Answer, JevResponse, Question, render_request

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

PROVIDERS = {
    "openrouter": {
        "url": OPENROUTER_URL,
        "model": "typesafe/jev-1.14",
        "key_env": "OPENROUTER_API_KEY",
    },
    "typesafe": {
        "url": TYPESAFE_URL,
        "model": "jev-latest",
        "key_env": "TYPESAFE_API_KEY",
    },
}


class JevApiError(RuntimeError):
    pass


class JevApiBackend:
    """Calls the hosted Jev model."""

    name = "jev"

    def __init__(
        self,
        *,
        provider: str = "openrouter",
        model: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}"
            )
        spec = PROVIDERS[provider]
        self.provider = provider
        self.url = url or spec["url"]
        self.model = model or spec["model"]
        self.api_key = api_key or os.environ.get(spec["key_env"], "")
        if not self.api_key:
            raise JevApiError(
                f"no API key: set {spec['key_env']} or pass api_key="
            )
        self.timeout = timeout
        self.max_retries = int(max_retries)
        #: Sum of the provider's own cost figures, in USD. Authoritative.
        self.reported_cost_usd = 0.0

    # -- request building --------------------------------------------------
    def _questions_payload(self, questions: Sequence[Question]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for q in questions:
            if q.kind == "choice":
                out[q.id] = {
                    "type": "choice",
                    "instructions": q.instruction,
                    "options": list(q.options or ()),
                }
            elif q.kind == "noul":
                # Jev's noul takes the proposition as its instruction.
                out[q.id] = {
                    "type": "noul",
                    "instructions": q.proposition or q.instruction,
                }
            elif q.kind == "score":
                out[q.id] = {
                    "type": "score",
                    "instructions": q.instruction,
                    "levels": list(q.rubric),
                }
            else:
                raise ValueError(f"unsupported question kind {q.kind!r}")
        return out

    # -- the Jev interface -------------------------------------------------
    def query(
        self,
        situation: str,
        questions: Sequence[Question],
        *,
        hint: Optional[dict] = None,
    ) -> JevResponse:
        # `hint` carries privileged context the mock backends use to synthesise
        # answers. A real Jev must never see it, and it is never billed.
        del hint

        payload = {
            "model": self.model,
            "state": situation,
            "questions": self._questions_payload(questions),
        }
        body, wall_ms = self._post(payload)

        answers: dict[str, Answer] = {}
        for q in questions:
            raw = body.get(q.id)
            if raw is None:
                raise JevApiError(
                    f"Jev returned no answer for question {q.id!r}; got keys "
                    f"{sorted(k for k in body if k != 'usage')}"
                )
            answers[q.id] = _parse_answer(q, raw)

        usage = body.get("usage") or {}
        reported = usage.get("cost")
        if reported is not None:
            self.reported_cost_usd += float(reported)

        return JevResponse(
            answers=answers,
            tokens_in=int(
                usage.get(
                    "input_tokens",
                    count_text_tokens(render_request(situation, questions)),
                )
            ),
            forward_passes=1,
            wall_ms=wall_ms,
            reported_cost_usd=float(reported) if reported is not None else None,
        )

    # -- transport ---------------------------------------------------------
    def _post(self, payload: dict) -> tuple[dict, float]:
        req_data = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.url,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read()), (time.perf_counter() - t0) * 1000.0
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:400]
                last = JevApiError(f"HTTP {exc.code} from {self.url}: {detail}")
                # Retry only what is worth retrying.
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise last from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = JevApiError(f"{type(exc).__name__} calling {self.url}: {exc}")
            if attempt < self.max_retries - 1:
                time.sleep(0.5 * (2**attempt))
        raise last if last else JevApiError("request failed")


def _parse_answer(q: Question, raw: dict) -> Answer:
    """Map one Jev answer onto our Answer type.

    Jev's score primitive returns a *numeric* probability-weighted level, not
    an ordinal word, so it is converted to the rubric fraction the pipelines
    threshold on. Getting this wrong would silently change every accept/reject
    decision in the fine subgoals, so the conversion lives in one place.
    """
    if q.kind == "choice":
        probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
        if not probs and raw.get("choice") is not None:
            probs = {str(raw["choice"]): 1.0}
        return Answer(id=q.id, kind="choice", distribution=probs)

    if q.kind == "noul":
        return Answer(id=q.id, kind="noul", probability=float(raw.get("noul", 0.0)))

    levels = list(q.rubric)
    score = float(raw.get("score", 0.0))
    n = max(1, len(levels) - 1)
    fraction = max(0.0, min(1.0, score / n))
    idx = int(round(fraction * n))
    return Answer(
        id=q.id,
        kind="score",
        score=levels[min(idx, len(levels) - 1)],
        numeric_score=fraction,
    )
