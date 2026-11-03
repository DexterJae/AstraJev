"""Real Jev backend over HTTP.

Contract, following the typed-question interface of arXiv:2610.11692:

  POST {base_url}/query
      {"situation": str,
       "questions": [{"id": str, "type": "choice"|"score"|"noul",
                      "instruction": str, "options": [...], "rubric": [...],
                      "proposition": str}, ...]}
   -> {"answers": {"<id>": {"distribution": {...}}       # choice
                           {"score": "high"}             # score
                           {"probability": 0.81}},       # noul
       "usage": {"input_tokens": int, "forward_passes": int}}

The whole batch must be answered in one request -- that single-forward-pass
property is the reason the hybrid pipeline is cheap, so sending one request per
question would quietly destroy the result being measured.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Optional, Sequence

from ..tokens import count_text_tokens
from .base import Answer, JevResponse, Question, render_request


class HttpJevBackend:
    name = "jev"

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def query(
        self,
        situation: str,
        questions: Sequence[Question],
        *,
        hint: Optional[dict] = None,
    ) -> JevResponse:
        # `hint` is intentionally dropped: a real Jev sees only the situation
        # text and the typed questions, and that is exactly what we bill for.
        del hint
        payload = {
            "situation": situation,
            "questions": [
                {
                    "id": q.id,
                    "type": q.kind,
                    "instruction": q.instruction,
                    "options": list(q.options) if q.options else None,
                    "rubric": list(q.rubric),
                    "proposition": q.proposition,
                }
                for q in questions
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/query",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        wall_ms = (time.perf_counter() - t0) * 1000.0

        raw = body["answers"]
        answers: dict[str, Answer] = {}
        for q in questions:
            a = raw[q.id]
            answers[q.id] = Answer(
                id=q.id,
                kind=q.kind,
                distribution={k: float(v) for k, v in (a.get("distribution") or {}).items()},
                score=str(a.get("score", "unknown")),
                probability=float(a.get("probability", 0.0)),
            )

        usage = body.get("usage") or {}
        return JevResponse(
            answers=answers,
            tokens_in=int(
                usage.get("input_tokens", count_text_tokens(render_request(situation, questions)))
            ),
            forward_passes=int(usage.get("forward_passes", 1)),
            wall_ms=wall_ms,
        )
