"""Real Astra backend: an OpenAI-compatible vision LLM (default gpt-6-astra).

One model, three jobs, each a separate chat call -- the same contract as
MockAstraBackend (backends/base.py, AstraBackend):

  plan_task   instruction + cameras -> ordered subgoals (prompts.PLAN_SYSTEM)
  localize    cameras -> a frame per named reference + obstacles
  act_fine    wrist view + commanded pose -> corrected pose

Prompts come from astraxjev/prompts.py, the same text the mock counts tokens
for, so a mock run and a real run are billed for the same prompts. Token
counts come from the provider's `usage` (cached input included); a provider
`usage.cost`, when present (OpenRouter), is recorded as authoritative.

Works with any chat-completions endpoint: OpenAI directly
(--vlm-base-url https://api.openai.com/v1 --vlm-key-env OPENAI_API_KEY) or
OpenRouter (model ids prefixed "openai/"). The model is a CLI choice; when it
is not gpt-6-astra, pass a matching --pricing file or the computed dollar
figure is at the wrong rate.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence

import numpy as np

from .. import prompts
from ..plan import Frame, Subgoal, TaskPlan
from ..types import Observation, Waypoint
from .base import AstraActResponse, AstraLocalizeResponse, AstraPlanResponse


class AstraApiError(RuntimeError):
    pass


def _png_b64(img: np.ndarray) -> str:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise AstraApiError(
            "pillow is required to send images; pip install pillow"
        ) from exc
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a chat reply.

    Vision models wrap JSON in prose or fences often enough that insisting on
    a bare object would make the backend fail for no good reason.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise AstraApiError(f"no JSON object in model reply: {text[:200]!r}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise AstraApiError(f"unterminated JSON in model reply: {text[:200]!r}")


class GptAstraBackend:
    """The expensive model behind the AstraBackend protocol."""

    name = "astra"

    def __init__(
        self,
        *,
        model: str = "gpt-6-astra",
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        max_retries: int = 3,
        max_completion_tokens: int = 2000,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise AstraApiError(f"no API key: set {api_key_env} or pass api_key=")
        self.timeout = timeout
        self.max_retries = int(max_retries)
        self.max_completion_tokens = int(max_completion_tokens)
        #: (job, parsed reply) per call, for inspection; never billed.
        self.log: list[tuple[str, dict]] = []

    # -- transport ---------------------------------------------------------
    def _chat(self, system: str, user_text: str, images: Sequence[np.ndarray]):
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in images:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_png_b64(img)}"}})
        payload = {
            "model": self.model,
            # max_completion_tokens, not max_tokens: the gpt-6 family rejects
            # the old name, and it also bounds hidden reasoning tokens.
            "max_completion_tokens": self.max_completion_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
        }
        data = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.loads(r.read())
                return body, (time.perf_counter() - t0) * 1000.0
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:400]
                last = AstraApiError(f"HTTP {exc.code}: {detail}")
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise last from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = AstraApiError(f"{type(exc).__name__}: {exc}")
            if attempt < self.max_retries - 1:
                time.sleep(0.5 * (2**attempt))
        raise last if last else AstraApiError("request failed")

    @staticmethod
    def _usage(body: dict) -> dict:
        u = body.get("usage") or {}
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        return dict(
            tokens_in=int(u.get("prompt_tokens", 0)),
            tokens_out=int(u.get("completion_tokens", 0)),
            cached_in=cached,
            reported_cost_usd=float(u["cost"]) if u.get("cost") is not None else None,
        )

    @staticmethod
    def _json(body: dict) -> dict:
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise AstraApiError(f"unexpected reply shape: {str(body)[:300]}") from exc
        if not text.strip():
            fin = body["choices"][0].get("finish_reason")
            raise AstraApiError(f"empty reply (finish_reason={fin}); raise max_completion_tokens?")
        return _extract_json(text)

    # -- the three jobs ----------------------------------------------------
    def plan_task(self, obs: Observation) -> AstraPlanResponse:
        imgs = obs.images()
        body, wall_ms = self._chat(prompts.PLAN_SYSTEM, prompts.plan_prompt(obs), imgs)
        raw = self._json(body)
        self.log.append(("plan", raw))
        try:
            plan = TaskPlan.from_dict({**raw, "instruction": obs.prompt})
        except (KeyError, ValueError, TypeError) as exc:
            raise AstraApiError(f"plan does not match the schema: {exc}; got {str(raw)[:300]}") from exc
        return AstraPlanResponse(plan=plan, images=len(imgs), wall_ms=wall_ms, **self._usage(body))

    def localize(self, obs: Observation, refs: Sequence[str]) -> AstraLocalizeResponse:
        refs = list(refs) or ["target"]
        imgs = obs.images()
        body, wall_ms = self._chat(prompts.LOCALIZE_SYSTEM, prompts.localize_prompt(obs, refs), imgs)
        raw = self._json(body)
        self.log.append(("localize", raw))
        frames: dict[str, Frame] = {}
        for name in refs:
            d = raw.get(name)
            # A reference the model could not find stays absent rather than
            # defaulted, so the pipeline re-localises or fails loudly.
            if not isinstance(d, dict) or "origin" not in d:
                continue
            # A model that cannot see the reference tends to answer with
            # zeros. A zero normal has no direction (normalising it is what
            # produced NaNs downstream), so treat it as "not found".
            try:
                if float(np.linalg.norm(np.asarray(d.get("normal", [0, 0, 1]), float))) < 1e-6:
                    continue
            except (ValueError, TypeError):
                continue
            try:
                frames[name] = Frame(
                    origin=d["origin"],
                    normal=d.get("normal", [0.0, 0.0, 1.0]),
                    extent=d.get("extent", [0.05, 0.05, 0.05]),
                    axis=d.get("axis", [0.0, 0.0, 1.0]),
                )
            except (ValueError, TypeError):
                continue
        obstacles = []
        for p in raw.get("obstacles") or []:
            try:
                obstacles.append(np.asarray(p, dtype=np.float64).reshape(3))
            except (ValueError, TypeError):
                pass
        return AstraLocalizeResponse(
            frames=frames, obstacles=obstacles,
            confidence=float(raw.get("confidence", 1.0)),
            images=len(imgs), wall_ms=wall_ms, **self._usage(body))

    def act_fine(self, obs: Observation, subgoal: Subgoal, goal: Waypoint) -> AstraActResponse:
        imgs = [obs.wrist_image, obs.image]  # same pair prompts.act_cost counts
        body, wall_ms = self._chat(prompts.ACT_SYSTEM, prompts.act_prompt(obs, subgoal, goal), imgs)
        raw = self._json(body)
        self.log.append(("act", raw))

        def vec(key, fallback):
            try:
                v = np.asarray(raw.get(key, fallback), dtype=np.float64).reshape(3)
                return v if np.all(np.isfinite(v)) else np.asarray(fallback, float)
            except (ValueError, TypeError):
                return np.asarray(fallback, float)

        def unit(v, fallback):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-6 else np.asarray(fallback, float)

        wp = Waypoint(
            position=vec("position", goal.position),
            approach=unit(vec("approach", goal.approach), goal.approach),
            closing=unit(vec("closing", goal.closing), goal.closing),
            gripper=goal.gripper,
            torso_delta=goal.torso_delta,
        )
        return AstraActResponse(waypoint=wp, images=len(imgs), wall_ms=wall_ms, **self._usage(body))
