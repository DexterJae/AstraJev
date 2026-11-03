"""Mock Astra backend.

The capability model is the substantive part of this file, because it is what
decides whether the division of labour in the hybrid pipeline makes sense at
all. It encodes three claims about Astra, each taken from what
arXiv:2506.06205 says each component is for:

  Astra-Global is semantically strong but spatially coarse.
      It reads the brand off the packaging and returns a 3D centre with a few
      centimetres of error. Its error shrinks as the gripper gets closer,
      because the item then fills more of the wrist image -- which is exactly
      why re-localising is worth anything, and therefore why "should I
      re-localise?" is a question worth paying to answer.

  Astra-Local is spatially precise.
      It is a visuomotor regressor, so in the fine phases it servos on what it
      currently sees rather than on a stale global estimate: its error is
      millimetre-scale *relative to the true item*, not relative to whatever
      Astra-Global last said. This is the capability the hybrid pays for and
      never delegates.

  Neither component is free, and Astra-Global is roughly an order of magnitude
  dearer than Astra-Local per call.

Token counts come from astraxjev/prompts.py, i.e. from the prompts really
constructed, so the cost ledger a mock run produces is the cost ledger a real
run would produce.
"""

from __future__ import annotations

import numpy as np

from .. import prompts
from ..geometry import grasp_waypoint
from ..oracle import Oracle
from ..types import (
    FINE_PRECISION_MM,
    Observation,
    PHASE_SPEC,
    Phase,
    TargetEstimate,
    Waypoint,
)
from .base import AstraGlobalResponse, AstraLocalResponse


class MockAstraBackend:
    name_global = "astra_global"
    name_local = "astra_local"

    def __init__(
        self,
        oracle: Oracle,
        *,
        global_sigma_m: float = 0.025,
        local_sigma_m: float = 0.004,
        rng: np.random.Generator | None = None,
        global_wall_ms: float = 420.0,
        local_wall_ms: float = 35.0,
        finger_length_m: float = 0.03,
        #: Distance at which Astra-Global performs at its nominal sigma. Closer
        #: than this it does better, further it does worse.
        nominal_view_distance_m: float = 0.5,
    ) -> None:
        self.oracle = oracle
        self.global_sigma_m = float(global_sigma_m)
        self.local_sigma_m = float(local_sigma_m)
        self.rng = rng or np.random.default_rng(0)
        self.global_wall_ms = global_wall_ms
        self.local_wall_ms = local_wall_ms
        self.finger_length_m = float(finger_length_m)
        self.nominal_view_distance_m = float(nominal_view_distance_m)

    # -- Astra-Global ------------------------------------------------------
    def localize(self, obs: Observation) -> AstraGlobalResponse:
        truth = self.oracle.target_position()
        sigma = self._view_dependent_sigma(truth)
        noise = self.rng.normal(0.0, sigma, size=3)
        target = TargetEstimate(
            position=truth + noise,
            extent=self.oracle.target_extent(),
            shelf_normal=self.oracle.shelf_normal(),
            # Confidence reflects viewing conditions, not the realised error --
            # a model does not know its own error. This is what makes
            # confidence a weak re-localisation trigger for the Astra-only
            # pipeline, and why the hybrid uses a Jev noul instead.
            confidence=float(
                np.clip(self.global_sigma_m / max(sigma, 1e-6), 0.0, 1.0)
            ),
            label=self.oracle.target_label(),
        )
        tin, tout, nimg = prompts.global_cost(obs)
        return AstraGlobalResponse(
            target=target,
            tokens_in=tin,
            tokens_out=tout,
            images=nimg,
            wall_ms=self.global_wall_ms,
        )

    def _view_dependent_sigma(self, truth: np.ndarray) -> float:
        d = float(np.linalg.norm(self.oracle.tcp_position() - truth))
        scale = float(np.clip(d / self.nominal_view_distance_m, 0.3, 2.0))
        return self.global_sigma_m * scale

    # -- Astra-Local -------------------------------------------------------
    def plan_local(
        self,
        obs: Observation,
        target: TargetEstimate,
        phase: Phase,
        goal: Waypoint,
        chunk_len: int,
    ) -> AstraLocalResponse:
        tcp = self.oracle.tcp_position()
        fine = PHASE_SPEC[phase]["precision_mm"] <= FINE_PRECISION_MM

        if fine:
            # Visual servoing: regress the grasp pose from the wrist view, so
            # the error is relative to the real item rather than to the stale
            # global estimate.
            truth_target = TargetEstimate(
                position=self.oracle.target_position(),
                extent=self.oracle.target_extent(),
                shelf_normal=self.oracle.shelf_normal(),
            )
            anchor = grasp_waypoint(truth_target, self.finger_length_m).position
        else:
            anchor = goal.position

        wps: list[Waypoint] = []
        for i in range(max(1, chunk_len)):
            alpha = (i + 1) / max(1, chunk_len)
            nominal = tcp + (anchor - tcp) * alpha
            jitter = self.rng.normal(0.0, self.local_sigma_m, size=3)
            wps.append(
                Waypoint(
                    position=nominal + jitter,
                    approach=goal.approach,
                    closing=goal.closing,
                    gripper=goal.gripper,
                    torso_delta=goal.torso_delta,
                )
            )

        tin, tout, nimg = prompts.local_cost(obs, target, phase, goal, chunk_len)
        return AstraLocalResponse(
            waypoints=wps,
            tokens_in=tin,
            tokens_out=tout,
            images=nimg,
            wall_ms=self.local_wall_ms,
            confidence=float(target.confidence),
        )
