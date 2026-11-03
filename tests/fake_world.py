"""A tiny kinematics-free stand-in for the RoboBenchMart environment.

This is NOT a substitute for the benchmark. It exists for two reasons:

  1. The real evaluation needs ManiSkill, Vulkan and ~GBs of RoboCasa assets on
     a Linux GPU box. The cost-accounting logic, the phase machine and the
     fusion policy must be testable without that.
  2. It makes the cost ledger exercisable end to end, and the cost half of the
     comparison does not depend on the physics at all -- it depends only on
     which prompts get built and how many times.

Treat any success rate produced here as a property of this toy world, never as
a benchmark result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from astraxjev.executor import ExecResult
from astraxjev.types import GRIPPER_CLOSED, Observation, Waypoint


@dataclass
class FakeWorld:
    rng: np.random.Generator
    shelf_normal: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, 0.0, 0.0])
    )
    target_position: np.ndarray = field(
        default_factory=lambda: np.array([1.20, 0.05, 1.05])
    )
    target_extent: np.ndarray = field(
        default_factory=lambda: np.array([0.07, 0.065, 0.19])
    )
    tcp: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.0, 0.95]))
    basket: np.ndarray = field(default_factory=lambda: np.array([0.30, 0.25, 0.60]))
    #: Non-target items standing on the same board, 7 cm apart.
    neighbours: np.ndarray = field(
        default_factory=lambda: np.array(
            [[1.20, -0.08, 1.05], [1.20, 0.18, 1.05], [1.20, 0.05, 1.30]]
        )
    )
    gripper_closed: bool = False
    #: Approach direction of the most recent commanded waypoint. The TCP sits
    #: behind the fingertips, so a correct grasp pose is deliberately ~one
    #: finger length short of the item centre (see geometry.grasp_waypoint);
    #: without tracking the approach direction this toy would score every
    #: correct grasp as a miss.
    approach: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    holding: bool = False
    neighbour_displaced: bool = False
    item_position: np.ndarray | None = None
    steps: int = 0

    # Physics constants of the toy world.
    exec_sigma_m: float = 0.003
    grasp_tol_m: float = 0.025
    finger_length_m: float = 0.03
    neighbour_clearance_m: float = 0.035
    #: Matches PickToBasketContEnv.TARGET_POS_THRESH in the real benchmark.
    basket_tol_m: float = 0.2
    #: Lateral radius of the basket mouth. Release outside it and the item
    #: lands on the floor instead of in the basket, as it would in the sim.
    basket_mouth_m: float = 0.18

    def __post_init__(self) -> None:
        if self.item_position is None:
            self.item_position = self.target_position.copy()

    # -- dynamics ----------------------------------------------------------
    def move_to(self, position: np.ndarray, approach: np.ndarray | None = None) -> None:
        """Straight-line move with small tracking error, swept for collisions."""
        if approach is not None:
            self.approach = np.asarray(approach, dtype=np.float64).reshape(3)
        start = self.tcp.copy()
        end = np.asarray(position, dtype=np.float64).reshape(3)
        end = end + self.rng.normal(0.0, self.exec_sigma_m, size=3)
        n = max(2, int(np.linalg.norm(end - start) / 0.01))
        for i in range(1, n + 1):
            p = start + (end - start) * (i / n)
            if not self.gripper_closed or not self.holding:
                self._check_neighbours(p)
            else:
                # While carrying, the item itself can clip a neighbour.
                self._check_neighbours(p)
            self.steps += 1
        self.tcp = end
        if self.holding:
            self.item_position = self.tcp.copy()

    def _check_neighbours(self, p: np.ndarray) -> None:
        d = np.linalg.norm(self.neighbours - p.reshape(1, 3), axis=1)
        if np.any(d < self.neighbour_clearance_m):
            self.neighbour_displaced = True

    def fingertip(self) -> np.ndarray:
        return self.tcp + self.approach * self.finger_length_m

    def close_gripper(self) -> None:
        self.gripper_closed = True
        self.steps += 6
        if np.linalg.norm(self.fingertip() - self.item_position) <= self.grasp_tol_m:
            self.holding = True

    def open_gripper(self) -> None:
        """Release, and let the item fall.

        The reference solution releases ~40 cm above the basket's success
        centre and lets the item drop in, so the toy has to model the drop or
        every correct release would score as a miss.
        """
        self.gripper_closed = False
        self.steps += 6
        if self.holding:
            lateral = float(
                np.linalg.norm(self.item_position[:2] - self.basket[:2])
            )
            if lateral <= self.basket_mouth_m:
                # Falls into the basket and settles at its centre height.
                self.item_position = np.array(
                    [self.item_position[0], self.item_position[1], self.basket[2]]
                )
            # Otherwise it falls wherever it was released -- a dropped item.
            self.steps += 10
        self.holding = False

    def success(self) -> bool:
        if self.neighbour_displaced:
            return False
        return bool(
            np.all(np.abs(self.item_position - self.basket) <= self.basket_tol_m)
        )

    # -- observation -------------------------------------------------------
    def observation(self, prompt: str) -> Observation:
        """Synthetic images of the right shapes, so token counts are real."""
        return Observation(
            image=np.zeros((256, 256, 3), dtype=np.uint8),
            extra_image=np.zeros((256, 256, 3), dtype=np.uint8),
            wrist_image=np.zeros((128, 128, 3), dtype=np.uint8),
            state=np.zeros(15, dtype=np.float32),
            prompt=prompt,
            time_step=self.steps,
        )


class FakeOracle:
    def __init__(self, w: FakeWorld) -> None:
        self.w = w

    def target_position(self):
        return self.w.item_position.copy()

    def target_extent(self):
        return self.w.target_extent.copy()

    def shelf_normal(self):
        return self.w.shelf_normal.copy()

    def tcp_position(self):
        return self.w.tcp.copy()

    def basket_position(self):
        return self.w.basket.copy()

    def nearest_neighbour_distance(self):
        return float(
            np.min(np.linalg.norm(self.w.neighbours - self.w.tcp.reshape(1, 3), axis=1))
        )

    def gripper_closed(self):
        return self.w.gripper_closed

    def holding_item(self):
        return self.w.holding

    def target_label(self):
        return "Nivea Body Milk"


class FakeExecutor:
    def __init__(self, w: FakeWorld, *, max_env_steps: int = 500) -> None:
        self.w = w
        self.max_env_steps = max_env_steps

    @property
    def env_steps(self) -> int:
        return self.w.steps

    def execute(self, wp: Waypoint) -> ExecResult:
        before = self.w.steps
        # Crude reachability check: the toy arm cannot reach behind itself.
        if wp.position[0] < 0.1 or wp.position[2] < 0.2:
            self.w.steps += 2
            return ExecResult(False, self.w.steps - before, "no IK solution")
        self.w.move_to(wp.position, wp.approach)
        if wp.gripper <= 0 and not self.w.gripper_closed:
            self.w.close_gripper()
        return ExecResult(True, self.w.steps - before)

    def set_gripper(self, gripper: float) -> ExecResult:
        before = self.w.steps
        if gripper <= 0:
            self.w.close_gripper()
        else:
            self.w.open_gripper()
        return ExecResult(True, self.w.steps - before)

    def lift(self, delta_m: float) -> ExecResult:
        before = self.w.steps
        self.w.move_to(self.w.tcp + np.array([0.0, 0.0, float(delta_m)]))
        return ExecResult(True, self.w.steps - before)

    def idle(self, steps: int) -> ExecResult:
        self.w.steps += int(steps)
        return ExecResult(True, int(steps))

    def episode_done(self) -> bool:
        return self.w.steps >= self.max_env_steps

    def success(self) -> bool:
        return self.w.success()
