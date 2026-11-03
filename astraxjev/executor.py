"""Shared low-level waypoint executor.

Both pipelines emit `Waypoint`s and both execute them through the *same*
executor. That is the single most important control in this experiment: if the
two pipelines used different low-level controllers, any difference in success
rate or episode length would be uninterpretable.

The ManiSkill executor drives the arm with RoboBenchMart's own motion-planning
solver (`FetchMotionPlanningSapienSolver.static_manipulation`), which holds the
body joints at their current value and writes zeros into the base action dims.
So the mobile base never moves, and the task is solved with arm + torso lift +
gripper only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .types import GRIPPER_CLOSED, GRIPPER_OPEN, Waypoint


@dataclass
class ExecResult:
    ok: bool
    env_steps: int
    reason: str = ""


@runtime_checkable
class WaypointExecutor(Protocol):
    def execute(self, wp: Waypoint) -> ExecResult: ...
    def set_gripper(self, gripper: float) -> ExecResult: ...
    def idle(self, steps: int) -> ExecResult: ...
    def lift(self, delta_m: float) -> ExecResult: ...
    @property
    def env_steps(self) -> int: ...
    def episode_done(self) -> bool: ...
    def success(self) -> bool: ...


class ManiSkillExecutor:
    """Executes waypoints in a live RoboBenchMart environment."""

    def __init__(self, env, planner, *, max_env_steps: int = 500) -> None:
        self.env = env
        self.planner = planner
        self.max_env_steps = int(max_env_steps)
        self._last_info: dict = {}
        self._terminated = False

    # -- WaypointExecutor --------------------------------------------------
    @property
    def env_steps(self) -> int:
        return int(self.planner.elapsed_steps)

    def execute(self, wp: Waypoint) -> ExecResult:
        import sapien

        before = self.env_steps
        if abs(wp.torso_delta) > 1e-4:
            if self.planner.lift_body(float(wp.torso_delta)) == -1:
                return ExecResult(False, self.env_steps - before, "lift_body failed")

        pose = self.env.unwrapped.agent.build_grasp_pose(
            wp.approach, wp.closing, wp.position
        )
        if not self.planner.check_IK(pose):
            return ExecResult(False, self.env_steps - before, "no IK solution")

        res = self.planner.static_manipulation(pose)
        if res == -1:
            return ExecResult(False, self.env_steps - before, "motion plan failed")
        self._absorb(res)
        return ExecResult(True, self.env_steps - before)

    def set_gripper(self, gripper: float) -> ExecResult:
        before = self.env_steps
        res = (
            self.planner.close_gripper()
            if gripper <= 0
            else self.planner.open_gripper()
        )
        if res == -1:
            return ExecResult(False, self.env_steps - before, "gripper failed")
        self._absorb(res)
        return ExecResult(True, self.env_steps - before)

    def lift(self, delta_m: float) -> ExecResult:
        """Raise or lower the torso.

        Not mobile-base motion: a prismatic lift, which is what lets a
        fixed-base arm cover the height range of a retail shelf.
        """
        before = self.env_steps
        if self.planner.lift_body(float(delta_m)) == -1:
            return ExecResult(False, self.env_steps - before, "lift_body failed")
        return ExecResult(True, self.env_steps - before)

    def idle(self, steps: int) -> ExecResult:
        before = self.env_steps
        res = self.planner.idle_steps(t=int(steps))
        if res == -1:
            return ExecResult(False, self.env_steps - before, "idle failed")
        self._absorb(res)
        return ExecResult(True, self.env_steps - before)

    def episode_done(self) -> bool:
        return self._terminated or self.env_steps >= self.max_env_steps

    def success(self) -> bool:
        s = self._last_info.get("success")
        if s is None:
            return False
        try:
            return bool(s[0].item())
        except (TypeError, AttributeError, IndexError):
            return bool(s)

    # -- internals ---------------------------------------------------------
    def _absorb(self, res) -> None:
        """Keep the latest info dict from whatever the planner returned."""
        if isinstance(res, tuple) and len(res) == 5:
            _, _, terminated, truncated, info = res
            self._last_info = info or {}
            try:
                self._terminated = bool(terminated[0].item()) or bool(
                    truncated[0].item()
                )
            except (TypeError, AttributeError, IndexError):
                self._terminated = bool(terminated) or bool(truncated)
