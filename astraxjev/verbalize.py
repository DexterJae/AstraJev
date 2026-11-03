"""Turn the coarse symbolic state into the text Jev reads.

Three deliberate choices, all load-bearing for the experiment:

1. Everything is quantised to whole centimetres. Jev structurally *cannot*
   drive a millimetre-tolerance contact off this description, so the division
   of labour is enforced by the interface rather than by a promise in a paper.

2. The encoding is terse key=value, not prose. Jev is billed per prompt token
   and its interface is *typed* -- the question kind and the declared option
   set already carry the semantics -- so English sentences explaining the
   situation would be pure cost. Jev calls are the largest single line item in
   the hybrid pipeline, so prompt length is one of the few parameters the cost
   claim is sensitive to. The token count is measured, never assumed.

3. Nothing here names a task phase. The description is "move the gripper from
   here to there without touching that", which is why the same verbaliser
   serves a shelf pick, a floor pick, a board-to-board transfer and a fridge
   door without modification.
"""

from __future__ import annotations

import numpy as np

from .types import SymbolicState

#: Jev's coarse action vocabulary, in the frame of the current subgoal's
#: reference. Generic directions, not task steps.
COARSE_ACTIONS = (
    "toward_goal",
    "away_from_goal",
    "up",
    "down",
    "left",
    "right",
    "hold",
)

#: One line, sent once per query, covering what is constant across ticks.
PREAMBLE = (
    "units=cm int; frame=goal-relative; base=locked(arm+gripper only); "
    "rule=displacing a non-target item voids the episode"
)


def _cm(x: float) -> int:
    return int(round(float(x) * 100.0))


def _vec_cm(v) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return f"({_cm(v[0])}, {_cm(v[1])}, {_cm(v[2])})"


def describe(state: SymbolicState, instruction: str) -> str:
    """Render the situation string for a Jev query (~110 tokens)."""
    to_goal = state.to_goal()
    return (
        f"{PREAMBLE}\n"
        f"task={instruction}\n"
        f"subgoal={state.subgoal} kind={state.goal_kind}\n"
        f"grip={'closed' if state.gripper_closed else 'open'} "
        f"held={'target' if state.holding_item else 'none'}\n"
        f"d_goal={_vec_cm(to_goal)} |d_goal|={_cm(np.linalg.norm(to_goal))}\n"
        f"d_nearest_nontarget={int(round(state.nearest_neighbour_cm))}\n"
        f"frame_normal={_vec_cm(state.frame_normal)}\n"
        f"tick={state.steps_in_subgoal}/{state.steps_total}"
    )


def resolve_coarse_action(
    choice: str, state: SymbolicState, step_m: float
) -> np.ndarray:
    """Map one of COARSE_ACTIONS to a TCP translation in world coordinates.

    `toward_goal` is a *direction*, not a teleport: the step is clipped to
    `step_m`, so reaching the goal takes several Jev ticks and the hybrid pays
    for every tick it uses.
    """
    to_goal = state.to_goal()
    dist = float(np.linalg.norm(to_goal))

    normal = np.asarray(state.frame_normal, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(up, normal)
    nl = float(np.linalg.norm(lateral))
    lateral = lateral / nl if nl > 1e-9 else np.array([0.0, 1.0, 0.0])

    if choice == "hold":
        return np.zeros(3)
    if choice == "toward_goal":
        return np.zeros(3) if dist < 1e-6 else to_goal / dist * min(step_m, dist)
    if choice == "away_from_goal":
        return normal * step_m if dist < 1e-6 else -to_goal / dist * step_m
    if choice == "up":
        return up * step_m
    if choice == "down":
        return -up * step_m
    if choice == "left":
        return lateral * step_m
    if choice == "right":
        return -lateral * step_m
    raise ValueError(f"unknown coarse action {choice!r}")
