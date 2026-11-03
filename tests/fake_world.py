"""A tiny kinematics-free stand-in for the RoboBenchMart environments.

This is NOT a substitute for the benchmark. It exists for two reasons:

  1. The real evaluation needs ManiSkill, Vulkan and GBs of RoboCasa assets on
     a Linux GPU box. The cost accounting, the plan driver and the routing
     policy must be testable without that.
  2. It makes the generality claim checkable. The whole point of generating
     plans instead of hard-coding them is that one pipeline should drive
     several task shapes; a world that only models shelf picking could not
     show that. So this world models four shapes, including an articulated
     door, which is structurally unlike pick-and-place: rotational motion, no
     free object carried, and a grip that tears off if you leave the hinge
     arc.

Treat any success rate produced here as a property of this toy world, never as
a benchmark result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from astraxjev.executor import ExecResult
from astraxjev.plan import Frame, TaskCapabilities
from astraxjev.types import Waypoint

#: The four task shapes, with the instruction strings RoboBenchMart actually
#: emits, so the planner stand-in is exercised on real text.
TASKS = {
    "pick_to_basket": "move to shelf and pick Nivea Body Milk to basket",
    "board_to_board": "pick Duff Beer Can and place on empty board",
    "pick_from_floor": "pick Heinz Beans from floor and place it on shelf",
    "open_fridge": "open the fridge",
}


@dataclass
class FakeWorld:
    rng: np.random.Generator
    task: str = "pick_to_basket"

    shelf_normal: np.ndarray = field(default_factory=lambda: np.array([-1.0, 0.0, 0.0]))
    tcp: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.0, 0.95]))
    approach: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    # pick-and-place state
    target_position: np.ndarray = field(
        default_factory=lambda: np.array([1.20, 0.05, 1.05])
    )
    target_extent: np.ndarray = field(
        default_factory=lambda: np.array([0.07, 0.065, 0.19])
    )
    basket: np.ndarray = field(default_factory=lambda: np.array([0.30, 0.25, 0.60]))
    goal_site: np.ndarray = field(default_factory=lambda: np.array([1.20, 0.30, 1.35]))
    item_position: np.ndarray | None = None

    # articulated-door state
    hinge: np.ndarray = field(default_factory=lambda: np.array([1.25, -0.35, 1.00]))
    handle_radius: float = 0.55
    door_angle: float = 0.0

    gripper_closed: bool = False
    holding: bool = False
    holding_handle: bool = False
    neighbour_displaced: bool = False
    steps: int = 0

    # physics constants of the toy world
    exec_sigma_m: float = 0.003
    finger_length_m: float = 0.03
    #: Matches PickToBasketContEnv.TARGET_POS_THRESH.
    place_tolerance_m: float = 0.2
    basket_mouth_m: float = 0.18
    #: Centre-to-centre spacing of items on a board. The RoboBenchMart scene
    #: configs use y_gap 0.02 between items ~0.065 wide.
    neighbour_spacing_m: float = 0.085
    neighbour_clearance_m: float = 0.035
    #: How far off the hinge arc a held handle tolerates before slipping.
    arc_slip_m: float = 0.04
    door_open_deg: float = 55.0

    def __post_init__(self) -> None:
        if self.item_position is None:
            self.item_position = self.target_position.copy()
        if self.task == "pick_from_floor":
            # The item is on the floor and the destination is its shelf slot:
            # the vertical relationship is inverted relative to a shelf pick.
            self.target_position = np.array([1.05, 0.05, 0.08])
            self.item_position = self.target_position.copy()
            self.goal_site = np.array([1.20, 0.05, 1.05])
        self.neighbours = self._neighbours()

    def _neighbours(self) -> np.ndarray:
        if self.task == "open_fridge":
            # Nothing to knock over; the risk in a door task is the door.
            return np.zeros((0, 3))
        g = self.neighbour_spacing_m
        c = self.target_position
        if self.task == "pick_from_floor":
            # Items on a floor have neighbours beside them, not stacked above:
            # anything 25 cm overhead would be a shelf board, not something
            # knockable, and placing one there would just make the approach
            # path unwinnable for reasons that have nothing to do with routing.
            return np.array([c + [0, -g, 0], c + [0, g, 0], c + [g, 0, 0]])
        return np.array([c + [0, -g, 0], c + [0, g, 0], c + [0, 0, 0.25]])

    @property
    def handle(self) -> np.ndarray:
        """Handle position, which moves as the door swings."""
        r = np.array([0.0, self.handle_radius, 0.0])
        return self.hinge + _rot_z(r, np.deg2rad(self.door_angle))

    # -- dynamics ----------------------------------------------------------
    def move_to(self, position, approach=None) -> None:
        if approach is not None:
            self.approach = np.asarray(approach, dtype=np.float64).reshape(3)
        start = self.tcp.copy()
        end = np.asarray(position, dtype=np.float64).reshape(3)
        end = end + self.rng.normal(0.0, self.exec_sigma_m, size=3)
        n = max(2, int(np.linalg.norm(end - start) / 0.01))
        for i in range(1, n + 1):
            p = start + (end - start) * (i / n)
            self._check_neighbours(p)
            if self.holding_handle:
                self._drag_door(p)
            self.steps += 1
        self.tcp = end
        if self.holding:
            self.item_position = self.tcp.copy()

    def _check_neighbours(self, p: np.ndarray) -> None:
        if len(self.neighbours) == 0:
            return
        d = np.linalg.norm(self.neighbours - p.reshape(1, 3), axis=1)
        if np.any(d < self.neighbour_clearance_m):
            self.neighbour_displaced = True

    def _drag_door(self, p: np.ndarray) -> None:
        """A held handle follows the hinge arc, or the grip slips off it."""
        r = p - self.hinge
        radial = np.array([r[0], r[1], 0.0])
        dist = float(np.linalg.norm(radial))
        if abs(dist - self.handle_radius) > self.arc_slip_m:
            self.holding_handle = False
            return
        self.door_angle = float(np.rad2deg(np.arctan2(radial[0], radial[1])))

    def fingertip(self) -> np.ndarray:
        return self.tcp + self.approach * self.finger_length_m

    def close_gripper(self) -> None:
        self.gripper_closed = True
        self.steps += 6
        tol = self.grasp_tolerance_m()
        if self.task == "open_fridge":
            if np.linalg.norm(self.fingertip() - self.handle) <= tol:
                self.holding_handle = True
        elif np.linalg.norm(self.fingertip() - self.item_position) <= tol:
            self.holding = True

    def open_gripper(self) -> None:
        """Release, and let a carried item fall."""
        self.gripper_closed = False
        self.steps += 6
        self.holding_handle = False
        if self.holding:
            dest = self.basket if self.task == "pick_to_basket" else self.goal_site
            mouth = self.basket_mouth_m if self.task == "pick_to_basket" else 0.12
            if float(np.linalg.norm(self.item_position[:2] - dest[:2])) <= mouth:
                self.item_position = np.array(
                    [self.item_position[0], self.item_position[1], dest[2]]
                )
            self.steps += 10
        self.holding = False

    def grasp_tolerance_m(self) -> float:
        return self.capabilities().grasp_tolerance_m()

    # -- scoring -----------------------------------------------------------
    def success(self) -> bool:
        if self.neighbour_displaced:
            return False
        if self.task == "open_fridge":
            return abs(self.door_angle) >= self.door_open_deg
        dest = self.basket if self.task == "pick_to_basket" else self.goal_site
        return bool(
            np.all(np.abs(self.item_position - dest) <= self.place_tolerance_m)
        )

    # -- oracle inputs -----------------------------------------------------
    def capabilities(self) -> TaskCapabilities:
        width = float(np.min(self.target_extent[:2]))
        return TaskCapabilities(
            place_tolerance_m=self.place_tolerance_m,
            finger_length_m=self.finger_length_m,
            gripper_stroke_m=0.1,
            neighbour_gap_m=self.neighbour_spacing_m
            if len(self.neighbours)
            else 0.3,
            item_width_m=width,
        )

    def frames(self) -> dict[str, Frame]:
        up = np.array([0.0, 0.0, 1.0])
        if self.task == "open_fridge":
            return {
                "handle": Frame(
                    origin=self.handle, normal=self.shelf_normal,
                    extent=np.array([0.03, 0.03, 0.12]),
                ),
                "hinge": Frame(
                    origin=self.hinge, normal=self.shelf_normal, axis=up,
                ),
            }
        out = {
            "target": Frame(
                origin=self.item_position,
                # An item on the floor is approached from above, not from the
                # side: the frame, not the pipeline, carries that.
                normal=up if self.task == "pick_from_floor" else self.shelf_normal,
                extent=self.target_extent,
            )
        }
        if self.task == "pick_to_basket":
            out["basket"] = Frame(origin=self.basket, normal=up)
        else:
            out["goal_site"] = Frame(origin=self.goal_site, normal=up)
        return out

    # -- observation -------------------------------------------------------
    def observation(self, prompt: str):
        """Synthetic images of the right shapes, so token counts are real."""
        from astraxjev.types import Observation

        return Observation(
            image=np.zeros((256, 256, 3), dtype=np.uint8),
            extra_image=np.zeros((256, 256, 3), dtype=np.uint8),
            wrist_image=np.zeros((128, 128, 3), dtype=np.uint8),
            state=np.zeros(15, dtype=np.float32),
            prompt=prompt,
            time_step=self.steps,
        )


def _rot_z(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


class FakeOracle:
    def __init__(self, w: FakeWorld) -> None:
        self.w = w

    def frames(self):
        return self.w.frames()

    def free_ref_names(self):
        # The basket is bolted to the robot; a shelf destination is not.
        return ("basket",)

    def capabilities(self):
        return self.w.capabilities()

    def tcp_position(self):
        return self.w.tcp.copy()

    def obstacles(self):
        return [p.copy() for p in self.w.neighbours]

    def nearest_neighbour_distance(self):
        if len(self.w.neighbours) == 0:
            return 10.0
        return float(
            np.min(np.linalg.norm(self.w.neighbours - self.w.tcp.reshape(1, 3), axis=1))
        )

    def gripper_closed(self):
        return self.w.gripper_closed

    def holding_item(self):
        return self.w.holding or self.w.holding_handle

    def target_label(self):
        return "target"


class FakeExecutor:
    def __init__(self, w: FakeWorld, *, max_env_steps: int = 500) -> None:
        self.w = w
        self.max_env_steps = max_env_steps

    @property
    def env_steps(self) -> int:
        return self.w.steps

    def execute(self, wp: Waypoint) -> ExecResult:
        before = self.w.steps
        # Crude reachability: the toy arm cannot reach behind or below itself.
        if wp.position[0] < 0.1 or wp.position[2] < 0.02:
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
