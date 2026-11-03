"""Real Astra backend over HTTP.

Astra's weights are not public (arXiv:2506.06205 reports in-house deployment),
so this adapter is written against a minimal contract you implement on top of
whatever serving stack you get access to:

  POST {base_url}/global   {"prompt": str, "images": [b64 png, ...]}
      -> {"position": [3], "extent": [3], "shelf_normal": [3],
          "confidence": float, "label": str,
          "usage": {"input_tokens": int, "output_tokens": int}}

  POST {base_url}/local    {"prompt": str, "images": [b64 png, ...],
                            "chunk_len": int}
      -> {"waypoints": [{"position": [3], "approach": [3], "closing": [3],
                         "gripper": float}, ...],
          "confidence": float,
          "usage": {"input_tokens": int, "output_tokens": int}}

If the server reports `usage`, those counts are billed; otherwise the counts
from astraxjev/prompts.py are used, so the ledger is never silently empty.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
from typing import Any

import numpy as np

from .. import prompts
from ..types import Observation, Phase, TargetEstimate, Waypoint
from .base import AstraGlobalResponse, AstraLocalResponse


def _png_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class HttpAstraBackend:
    name_global = "astra_global"
    name_local = "astra_local"

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> tuple[dict, float]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        return body, (time.perf_counter() - t0) * 1000.0

    def localize(self, obs: Observation) -> AstraGlobalResponse:
        prompt = prompts.global_prompt(obs)
        body, wall_ms = self._post(
            "/global",
            {"prompt": prompt, "images": [_png_b64(i) for i in obs.images()]},
        )
        fallback_in, fallback_out, nimg = prompts.global_cost(obs)
        usage: dict[str, Any] = body.get("usage") or {}
        return AstraGlobalResponse(
            target=TargetEstimate(
                position=body["position"],
                extent=body["extent"],
                shelf_normal=body["shelf_normal"],
                confidence=float(body.get("confidence", 1.0)),
                label=str(body.get("label", "")),
            ),
            tokens_in=int(usage.get("input_tokens", fallback_in)),
            tokens_out=int(usage.get("output_tokens", fallback_out)),
            images=nimg,
            wall_ms=wall_ms,
        )

    def plan_local(
        self,
        obs: Observation,
        target: TargetEstimate,
        phase: Phase,
        goal: Waypoint,
        chunk_len: int,
    ) -> AstraLocalResponse:
        prompt = prompts.local_prompt(obs, target, phase, goal, chunk_len)
        body, wall_ms = self._post(
            "/local",
            {
                "prompt": prompt,
                "images": [_png_b64(i) for i in obs.images()],
                "chunk_len": int(chunk_len),
            },
        )
        fallback_in, fallback_out, nimg = prompts.local_cost(
            obs, target, phase, goal, chunk_len
        )
        usage: dict[str, Any] = body.get("usage") or {}
        wps = [
            Waypoint(
                position=w["position"],
                approach=w["approach"],
                closing=w["closing"],
                gripper=float(w.get("gripper", goal.gripper)),
            )
            for w in body["waypoints"]
        ]
        return AstraLocalResponse(
            waypoints=wps,
            tokens_in=int(usage.get("input_tokens", fallback_in)),
            tokens_out=int(usage.get("output_tokens", fallback_out)),
            images=nimg,
            wall_ms=wall_ms,
            confidence=float(body.get("confidence", 1.0)),
        )
