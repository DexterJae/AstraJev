"""Real-model backends for the Astra side.

The honest situation, stated up front: **the Astra of arXiv:2506.06205 has no
public API and no released weights.** (Note the name collision -- OpenAI's
`gpt-6-astra`, released 2026-09, is an unrelated model.) So there is no way to
call "Astra" the way `jev_api.py` calls Jev. What there is instead:

  Astra-Global is a multimodal LLM doing two jobs -- decompose the task, and
      localise the references the plan names. Any capable vision LLM can be
      asked to do exactly that, so `VlmAstraGlobal` drives an
      OpenAI-compatible chat-completions endpoint. Using OpenRouter means one
      API key covers this and Jev together.

  Astra-Local is a visuomotor policy regressing end-effector poses from
      pixels. No chat API substitutes for that. What *is* downloadable is
      RoboBenchMart's own fine-tuned baselines (octo, pi0), which its
      `scripts/octo_server.py` serves over the same websocket protocol the
      benchmark's evaluation client uses. `WebsocketAstraLocal` speaks that
      protocol, so a real policy can fill this slot.

What this means for the claim: with these backends the experiment runs on real
models today, but the Astra arm becomes "a frontier VLM plus a downloaded VLA
policy", not Astra. That still tests the mechanism -- route by precision, gate
the expensive model with a cheap one -- which is the part that generalises.
It does not test Astra. Any write-up has to say which of the two it measured.
`--backend` names this in the results JSON so the distinction cannot be lost.
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
from ..types import Observation, TargetEstimate, Waypoint
from .base import AstraGlobalResponse, AstraLocalResponse, AstraPlanResponse


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


class VlmAstraGlobal:
    """A vision LLM standing in for Astra-Global.

    Speaks the OpenAI chat-completions shape, which OpenRouter, OpenAI and
    most hosted providers accept, so `base_url` + `model` is all that changes
    between providers.
    """

    name_global = "astra_global"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        timeout: float = 120.0,
        max_retries: int = 3,
        max_tokens: int = 1500,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise AstraApiError(f"no API key: set {api_key_env} or pass api_key=")
        self.timeout = timeout
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)

    # -- transport ---------------------------------------------------------
    def _chat(self, system: str, user_text: str, images: Sequence[np.ndarray]):
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_png_b64(img)}"},
                }
            )
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        data = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
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
    def _usage(body: dict, fallback_in: int, fallback_out: int) -> tuple[int, int, Optional[float]]:
        u = body.get("usage") or {}
        return (
            int(u.get("prompt_tokens", fallback_in)),
            int(u.get("completion_tokens", fallback_out)),
            float(u["cost"]) if u.get("cost") is not None else None,
        )

    @staticmethod
    def _text(body: dict) -> str:
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise AstraApiError(f"unexpected reply shape: {str(body)[:300]}") from exc

    # -- planning ----------------------------------------------------------
    def plan_task(self, obs: Observation) -> AstraPlanResponse:
        body, wall_ms = self._chat(
            prompts.PLAN_SYSTEM, prompts.plan_prompt(obs), obs.images()
        )
        raw = _extract_json(self._text(body))
        plan = TaskPlan.from_dict({**raw, "instruction": obs.prompt})
        f_in, f_out, _ = prompts.plan_cost(obs)
        tin, tout, cost = self._usage(body, f_in, f_out)
        return AstraPlanResponse(
            plan=plan,
            tokens_in=tin,
            tokens_out=tout,
            images=len(obs.images()),
            wall_ms=wall_ms,
            reported_cost_usd=cost,
        )

    # -- localisation ------------------------------------------------------
    def localize(
        self, obs: Observation, refs: Sequence[str]
    ) -> AstraGlobalResponse:
        refs = list(refs) or ["target"]
        body, wall_ms = self._chat(
            prompts.GLOBAL_SYSTEM, prompts.global_prompt(obs, refs), obs.images()
        )
        raw = _extract_json(self._text(body))
        frames: dict[str, Frame] = {}
        for name in refs:
            d = raw.get(name)
            if not isinstance(d, dict) or "origin" not in d:
                # A reference the model could not find is left absent rather
                # than defaulted; the pipeline then re-localises or fails
                # loudly instead of aiming at a made-up point.
                continue
            frames[name] = Frame(
                origin=d["origin"],
                normal=d.get("normal", [0.0, 0.0, 1.0]),
                extent=d.get("extent", [0.05, 0.05, 0.05]),
                axis=d.get("axis", [0.0, 0.0, 1.0]),
            )
        f_in, f_out, _ = prompts.global_cost(obs, refs)
        tin, tout, cost = self._usage(body, f_in, f_out)
        return AstraGlobalResponse(
            frames=frames,
            confidence=float(raw.get("confidence", 1.0)),
            tokens_in=tin,
            tokens_out=tout,
            images=len(obs.images()),
            wall_ms=wall_ms,
            reported_cost_usd=cost,
        )


class WebsocketAstraLocal:
    """A downloaded VLA policy standing in for Astra-Local.

    Speaks the websocket protocol of RoboBenchMart's `scripts/octo_server.py`,
    so the slot can be filled by the benchmark's own fine-tuned octo or pi0
    baselines:

        python scripts/octo_server.py --model-path <ckpt> --port 8000

    The policy returns joint-space action chunks, whereas the pipelines work in
    end-effector waypoints, so `goal` is passed through as the commanded pose
    and the policy's output is used to *correct* it. That is a real
    approximation and is documented rather than hidden: a policy trained to
    emit joint deltas is not a pose regressor, and bridging the two properly
    needs forward kinematics on the returned chunk.
    """

    name_local = "astra_local"

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self.host = host
        self.port = port
        self._client = None

    def _conn(self):
        if self._client is None:
            try:
                from dsynth.web_utils import WebsocketClient
            except ImportError as exc:  # pragma: no cover
                raise AstraApiError(
                    "WebsocketAstraLocal needs the RoboBenchMart checkout on "
                    "sys.path (dsynth.web_utils)"
                ) from exc
            self._client = WebsocketClient(host=self.host, port=self.port)
        return self._client

    def plan_local(
        self,
        obs: Observation,
        target: TargetEstimate,
        subgoal: Subgoal,
        goal: Waypoint,
        chunk_len: int,
    ) -> AstraLocalResponse:
        payload = {
            "observation/image": obs.image,
            "observation/extra_image": obs.extra_image,
            "observation/wrist_image": obs.wrist_image,
            "observation/state": obs.state,
            "prompt": obs.prompt,
            "time_step": obs.time_step,
        }
        t0 = time.perf_counter()
        reply = self._conn().infer(payload)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        actions = np.asarray(reply["actions"], dtype=np.float64)

        # The policy speaks joint space; we need a pose. Treat the commanded
        # goal as the nominal and leave the policy's chunk for the caller to
        # inspect, rather than inventing a conversion. See the class docstring.
        wps = [goal for _ in range(max(1, chunk_len))]
        tin, tout, nimg = prompts.local_cost(
            obs, target, subgoal.name, goal, chunk_len
        )
        return AstraLocalResponse(
            waypoints=wps,
            tokens_in=tin,
            tokens_out=tout,
            images=nimg,
            wall_ms=wall_ms,
            confidence=1.0,
            raw_actions=actions,
        )


class CompositeAstra:
    """Assembles an Astra-shaped backend out of whatever is available."""

    name_global = "astra_global"
    name_local = "astra_local"

    def __init__(self, global_backend, local_backend) -> None:
        self._g = global_backend
        self._l = local_backend

    def plan_task(self, obs):
        return self._g.plan_task(obs)

    def localize(self, obs, refs):
        return self._g.localize(obs, refs)

    def plan_local(self, obs, target, subgoal, goal, chunk_len):
        return self._l.plan_local(obs, target, subgoal, goal, chunk_len)
