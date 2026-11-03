"""Shared datatypes for the Astra / Astra+Jev pipelines.

The benchmark (RoboBenchMart, arXiv:2511.10276) drives a ds_fetch_basket robot
in ManiSkill with a 13-dim `pd_joint_pos` action:

    action = [ arm(0:7) | gripper(7) | head_pan(8) head_tilt(9) torso_lift(10)
               | base_vel(11) base_omega(12) ]

We pin the mobile-base dims (11, 12) and the head dims (8, 9) to zero, so both
pipelines solve the task with the arm + torso lift + gripper only.  That makes
this a *gripper task* in the sense asked for: no mobile manipulation, and the
only thing that differs between the two pipelines is who decides the end
effector waypoints and how often.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

ACTION_DIM = 13
ARM_SLICE = slice(0, 7)
GRIPPER_IDX = 7
HEAD_PAN_IDX = 8
HEAD_TILT_IDX = 9
TORSO_LIFT_IDX = 10
BASE_SLICE = slice(11, 13)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0


#: A subgoal at or below this tolerance is a "fine" one: it needs a visuomotor
#: controller servoing on current pixels, and its completion cannot be judged
#: against a stale global estimate.
FINE_PRECISION_MM = 10.0


@dataclass
class Observation:
    """What the benchmark's policy server contract actually hands us.

    Mirrors `prepare_obs` in RoboBenchMart/scripts/eval_policy_client.py.
    """

    image: np.ndarray  # left base camera, (256, 256, 3) uint8
    extra_image: np.ndarray  # right base camera, (256, 256, 3) uint8
    wrist_image: np.ndarray  # fetch_hand, (128, 128, 3) uint8
    state: np.ndarray  # agent qpos, (15,)
    prompt: str
    time_step: int

    @classmethod
    def from_server_dict(cls, d: dict) -> "Observation":
        return cls(
            image=d["observation/image"],
            extra_image=d["observation/extra_image"],
            wrist_image=d["observation/wrist_image"],
            state=d["observation/state"],
            prompt=d["prompt"],
            time_step=int(d["time_step"]),
        )

    def images(self) -> list[np.ndarray]:
        return [self.image, self.extra_image, self.wrist_image]


@dataclass
class Waypoint:
    """A target TCP pose plus gripper command.

    Both pipelines emit waypoints; a single shared executor turns a waypoint
    into joint-space actions.  Keeping the executor common is what makes the
    cost comparison fair -- the pipelines differ only in *decisions*, not in
    low-level control quality.
    """

    position: np.ndarray  # (3,) world-frame TCP position
    approach: np.ndarray  # (3,) unit vector the gripper points along
    closing: np.ndarray  # (3,) unit vector the fingers close along
    gripper: float = GRIPPER_OPEN
    # Relative height change of the torso lift to apply before the reach.
    torso_delta: float = 0.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.approach = _unit(self.approach)
        self.closing = _unit(self.closing)

    def with_offset(self, offset: np.ndarray) -> "Waypoint":
        return Waypoint(
            position=self.position + np.asarray(offset, dtype=np.float64).reshape(3),
            approach=self.approach,
            closing=self.closing,
            gripper=self.gripper,
            torso_delta=self.torso_delta,
        )


def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero-length direction vector")
    return v / n


@dataclass
class TargetEstimate:
    """Astra-Global's answer to 'where is the item named in the prompt'."""

    position: np.ndarray  # (3,) world-frame centre of the target item
    extent: np.ndarray  # (3,) bounding-box size, used to pick a grasp width
    shelf_normal: np.ndarray  # (3,) unit vector pointing out of the shelf
    confidence: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.extent = np.asarray(self.extent, dtype=np.float64).reshape(3)
        self.shelf_normal = _unit(self.shelf_normal)


@dataclass
class SymbolicState:
    """The coarse, text-serialisable state Jev is given.

    Jev consumes a *textual* situation description, not pixels (see
    arXiv:2610.11692), so the hybrid pipeline needs a cheap state verbaliser.
    Everything here is obtainable from a small open-vocabulary detector plus
    proprioception -- deliberately no sub-centimetre quantities, because the
    whole premise is that Jev only drives the low-precision subgoals.

    Note what is *not* here: the name of a task phase. The fields describe a
    generic "move the gripper from here to there without touching that", which
    is why the same verbaliser serves a shelf pick, a floor pick and a door.
    """

    subgoal: str
    goal_kind: str
    tcp_position: np.ndarray
    goal_position: np.ndarray
    frame_normal: np.ndarray
    gripper_closed: bool
    holding_item: bool
    nearest_neighbour_cm: float
    steps_in_subgoal: int
    steps_total: int

    def to_goal(self) -> np.ndarray:
        return np.asarray(self.goal_position, dtype=np.float64).reshape(
            3
        ) - np.asarray(self.tcp_position, dtype=np.float64).reshape(3)


@dataclass
class EpisodeOutcome:
    seed: int
    success: bool
    env_steps: int
    #: Name of the subgoal the episode stopped on, and how far through the
    #: generated plan it got.
    subgoal_reached: str
    subgoals_completed: int
    subgoals_total: int
    cost_usd: float
    astra_calls: int
    jev_calls: int
    failure_reason: str = ""
