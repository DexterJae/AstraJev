"""Grasp and waypoint geometry.

Mirrors the conventions RoboBenchMart's own motion-planning solution uses
(dsynth/planning/fetch_skills.py): the gripper approaches along the direction
pointing into the shelf, and the fingers close along the horizontal direction
perpendicular to it, which is the only way to get a gripper between two items
standing side by side on a board.
"""

from __future__ import annotations

import numpy as np

from .types import GRIPPER_CLOSED, GRIPPER_OPEN, TargetEstimate, Waypoint


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def approach_and_closing(shelf_normal) -> tuple[np.ndarray, np.ndarray]:
    """Approach points *into* the shelf; closing is horizontal across it."""
    normal = _unit(shelf_normal)
    approach = -normal
    closing = np.cross(approach, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(closing) < 1e-6:
        closing = np.array([0.0, 1.0, 0.0])
    return approach, _unit(closing)


def standoff_waypoint(
    target: TargetEstimate, standoff_m: float, *, gripper: float = GRIPPER_OPEN
) -> Waypoint:
    """Pre-grasp pose: out in front of the item, gripper open.

    This is the goal of the cheap APPROACH phase. It is intentionally a loose
    target -- anywhere within `handoff_radius_m` of it is a fine place for the
    expensive model to take over.
    """
    approach, closing = approach_and_closing(target.shelf_normal)
    return Waypoint(
        position=target.position + _unit(target.shelf_normal) * standoff_m,
        approach=approach,
        closing=closing,
        gripper=gripper,
    )


def grasp_waypoint(
    target: TargetEstimate, finger_length_m: float, *, gripper: float = GRIPPER_OPEN
) -> Waypoint:
    """The pose the gripper must hit to within a few millimetres.

    The TCP is placed slightly short of the item centre along the approach
    direction so the fingers straddle the item rather than colliding with it.
    """
    approach, closing = approach_and_closing(target.shelf_normal)
    depth = min(finger_length_m, float(target.extent[0]) * 0.5)
    return Waypoint(
        position=target.position - approach * depth,
        approach=approach,
        closing=closing,
        gripper=gripper,
    )


def retract_waypoint(
    grasp: Waypoint, target: TargetEstimate, retract_m: float
) -> Waypoint:
    """Straight back out along the shelf normal, still holding the item."""
    return Waypoint(
        position=grasp.position + _unit(target.shelf_normal) * retract_m,
        approach=grasp.approach,
        closing=grasp.closing,
        gripper=GRIPPER_CLOSED,
    )


#: Offset from the basket's success centre to the pose the gripper releases
#: from. Copied from `drop_to_basket` in the benchmark's own reference solution
#: (dsynth/planning/fetch_skills.py), which releases well above the rim and
#: lets the item drop in; `TARGET_POS_THRESH` for PickToBasket is a generous
#: 0.2 m per axis, so the drop lands comfortably inside it.
BASKET_RELEASE_OFFSET = np.array([0.05, 0.0, 0.40])


def place_waypoint(
    basket_position,
    *,
    gripper: float = GRIPPER_CLOSED,
    release_offset=None,
) -> Waypoint:
    """Over the basket, gripper pointing down."""
    basket_position = np.asarray(basket_position, dtype=np.float64).reshape(3)
    offset = (
        BASKET_RELEASE_OFFSET
        if release_offset is None
        else np.asarray(release_offset, dtype=np.float64).reshape(3)
    )
    return Waypoint(
        position=basket_position + offset,
        approach=np.array([0.0, 0.0, -1.0]),
        closing=np.array([0.0, 1.0, 0.0]),
        gripper=gripper,
    )


def distance(a: Waypoint | np.ndarray, b: Waypoint | np.ndarray) -> float:
    pa = a.position if isinstance(a, Waypoint) else np.asarray(a, dtype=np.float64)
    pb = b.position if isinstance(b, Waypoint) else np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(pa.reshape(3) - pb.reshape(3)))
