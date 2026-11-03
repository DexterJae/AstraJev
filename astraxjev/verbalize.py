"""Turn the coarse symbolic state into the text Jev reads.

Three deliberate choices, all load-bearing for the experiment:

1. Everything is quantised to whole centimetres. Jev structurally *cannot*
   drive a 6 mm-tolerance grasp off this description, so the division of labour
   is enforced by the interface rather than by a promise in a paper.

2. The encoding is terse key=value, not prose. Jev is billed per prompt token
   and its interface is *typed* -- the question kind and the declared option
   set already carry the semantics -- so English sentences explaining the
   situation would be pure cost. This matters more than it looks: Jev calls are
   the largest single line item in the hybrid pipeline, so prompt length is one
   of the handful of parameters the cost claim is actually sensitive to. The
   token count is measured by astraxjev/cost.py, never assumed.

3. The fields are fixed and ordered, so the prompt length barely varies across
   ticks and the per-episode cost is predictable.
"""

from __future__ import annotations

import numpy as np

from .types import Phase, SymbolicState

#: Jev's coarse action vocabulary in the TCP frame. The axes are expressed
#: relative to the shelf so the option names stay meaningful across layouts.
COARSE_ACTIONS = (
    "toward_goal",
    "away_from_goal",
    "up",
    "down",
    "left",
    "right",
    "hold",
)

PHASE_GOAL = {
    Phase.APPROACH: "standoff",
    Phase.RETRACT: "standoff",
    Phase.PLACE: "basket",
}


def _cm(x: float) -> int:
    return int(round(float(x) * 100.0))


def _vec_cm(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return f"({_cm(v[0])}, {_cm(v[1])}, {_cm(v[2])})"


#: One line, sent once per query, covering everything that is constant across
#: ticks: units, frame, and the two hard rules of the task.
PREAMBLE = (
    "units=cm int; frame=shelf; base=locked(arm+gripper only); "
    "rule=displacing a non-target item voids the episode"
)


def describe(state: SymbolicState, instruction: str) -> str:
    """Render the situation string for a Jev query.

    Terse by design -- see the module docstring. Roughly 110 tokens, against
    ~250 for an equivalent prose rendering, and Jev is the hybrid pipeline's
    largest cost line.
    """
    goal_name = PHASE_GOAL.get(state.phase, "standoff")
    to_goal = state.vector_to(goal_name)

    return (
        f"{PREAMBLE}\n"
        f"task={instruction}\n"
        f"phase={state.phase.value} goal={goal_name}\n"
        f"grip={'closed' if state.gripper_closed else 'open'} "
        f"held={'target' if state.holding_item else 'none'}\n"
        f"d_goal={_vec_cm(to_goal)} |d_goal|={_cm(np.linalg.norm(to_goal))}\n"
        f"d_nearest_nontarget={int(round(state.nearest_neighbour_cm))}\n"
        f"shelf_normal={_vec_cm(state.shelf_normal)}\n"
        f"tick={state.steps_in_phase}/{state.steps_total}"
    )


def resolve_coarse_action(
    choice: str, state: SymbolicState, step_m: float
) -> np.ndarray:
    """Map one of COARSE_ACTIONS to a TCP translation in world coordinates.

    `toward_goal` is a *direction*, not a teleport: the step is clipped to
    `step_m`, so reaching the goal takes several Jev ticks. That keeps the
    comparison honest -- the hybrid pays for every tick it uses.
    """
    goal_name = PHASE_GOAL.get(state.phase, "standoff")
    to_goal = state.vector_to(goal_name)
    dist = float(np.linalg.norm(to_goal))

    normal = np.asarray(state.shelf_normal, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(up, normal)
    nl = np.linalg.norm(lateral)
    lateral = lateral / nl if nl > 1e-9 else np.array([0.0, 1.0, 0.0])

    if choice == "hold":
        return np.zeros(3)
    if choice == "toward_goal":
        if dist < 1e-6:
            return np.zeros(3)
        return to_goal / dist * min(step_m, dist)
    if choice == "away_from_goal":
        if dist < 1e-6:
            return normal * step_m
        return -to_goal / dist * step_m
    if choice == "up":
        return up * step_m
    if choice == "down":
        return -up * step_m
    if choice == "left":
        return lateral * step_m
    if choice == "right":
        return -lateral * step_m
    raise ValueError(f"unknown coarse action {choice!r}")
