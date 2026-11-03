"""Prompt construction for the Astra backend.

These functions exist so the cost of an Astra call is *measured*. Both the API
backend and the mock build their prompts through here, so a mock run reports
the token counts a real run would, and the dollar figures carry over when the
real endpoint is plugged in.

Astra here is a frontier vision LLM (`gpt-6-astra`), which bills $10 per
million input tokens and **$50 per million output tokens**. Output is five
times dearer than input, so these prompts are written to make the model emit
as little as possible: strict JSON, no prose, no explanation. Every field the
schema asks for is a field that gets billed at the output rate.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .tokens import VisionTokenizerSpec, count_image_tokens, count_text_tokens
from .plan import Subgoal
from .types import Observation, Waypoint

#: A frontier VLM tower: ViT patches with a pooling resampler. Vision tokens on
#: `gpt-6-astra` bill at the ordinary input rate, so they are counted into
#: tokens_in rather than priced separately.
VISION = VisionTokenizerSpec(patch_size=14, pool_factor=2)

NO_PROSE = (
    "Reply with one JSON object and nothing else: no prose, no explanation, no "
    "code fence. Output tokens are the dominant cost of this system, so every "
    "character you emit beyond the schema is waste."
)

PLAN_SYSTEM = (
    "You are a robot task planner for a fixed-base arm with a parallel "
    "gripper. The mobile base is locked; you may use the arm, the torso lift "
    "and the gripper only. Decompose the instruction into an ordered list of "
    "subgoals. Each subgoal must use exactly one primitive:\n"
    "  standoff(ref, distance)  hold off in front of ref along its normal\n"
    "  contact(ref)             bring the gripper onto ref\n"
    "  withdraw(distance)       back off along the axis you came in on\n"
    "  over(ref, height)        position above ref, approaching downward\n"
    "  arc(ref, angle_deg)      swing about ref's hinge axis\n"
    "  set_gripper(gripper)     actuate only; gripper 1 open, -1 closed\n"
    "  hold(steps)              stay still and let the scene settle\n"
    'Schema: {"subgoals": [{"name": str, "kind": str, "ref": str, '
    '"tolerance_mm": number, "needs_pixels": bool, "gripper": number, '
    '"params": object}]}\n'
    "Set needs_pixels true only where the gripper must be placed relative to "
    "something currently seen rather than to a stored coordinate. Name every "
    "ref you need localised. A long swing must be split into several short "
    "arcs: a gripper holding a handle has to stay on the hinge arc, and a "
    "coarse controller will chase a single large arc in a straight line. "
    + NO_PROSE
)

LOCALIZE_SYSTEM = (
    "You are a robot perception module. From the robot's head cameras and "
    "wrist camera, localise every named reference in the world frame, in "
    "metres. Items on a retail shelf are densely packed: identify the one the "
    "instruction names by brand text and packaging, not by position alone.\n"
    'Schema: {"<ref>": {"origin": [x,y,z], "normal": [x,y,z], '
    '"extent": [w,d,h], "axis": [x,y,z]}, "obstacles": [[x,y,z], ...], '
    '"confidence": number}\n'
    "`normal` is the unit vector pointing out of the surface the reference "
    "rests on or against, so the gripper approaches along its negation. "
    "`axis` is the hinge axis, for articulated references only. `obstacles` "
    "are the centres of nearby items that must not be displaced. "
    + NO_PROSE
)

ACT_SYSTEM = (
    "You are a robot fine-motion controller. Given the cameras, the robot's "
    "joint positions and a commanded end-effector pose, output the corrected "
    "pose that will actually achieve the subgoal, in the world frame, in "
    "metres. Servo on what you can see now rather than on the commanded pose: "
    "the commanded pose comes from an earlier localisation and may be "
    "centimetres off, while this subgoal needs millimetres.\n"
    'Schema: {"position": [x,y,z], "approach": [x,y,z], "closing": [x,y,z]}\n'
    + NO_PROSE
)


def _fmt(v) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"


# -- plan ---------------------------------------------------------------------
def plan_prompt(obs: Observation) -> str:
    return (
        f"instruction: {obs.prompt}\n"
        f"robot_qpos: {_fmt(obs.state)}\n"
        f"cameras: head_left, head_right, wrist"
    )


_PLAN_EXAMPLE = (
    '{"subgoals": [{"name": "approach_item", "kind": "standoff", "ref": '
    '"target", "tolerance_mm": 40, "needs_pixels": false, "gripper": 1, '
    '"params": {"distance": 0.18}}, {"name": "grasp_item", "kind": "contact", '
    '"ref": "target", "tolerance_mm": 6, "needs_pixels": true, "gripper": 1, '
    '"params": {}}, {"name": "close", "kind": "set_gripper", "gripper": -1, '
    '"params": {}}, {"name": "extract", "kind": "withdraw", "tolerance_mm": '
    '25, "gripper": -1, "params": {"distance": 0.3}}, {"name": "over_basket", '
    '"kind": "over", "ref": "basket", "tolerance_mm": 60, "gripper": -1, '
    '"params": {"height": 0.4, "torso_lift": 0.3}}, {"name": "release", '
    '"kind": "set_gripper", "gripper": 1, "params": {}}, {"name": "settle", '
    '"kind": "hold", "params": {"steps": 20}}]}'
)


def plan_cost(obs: Observation) -> tuple[int, int, int, int]:
    """(tokens_in, tokens_out, n_images, cacheable_in) for one planning call.

    Charged to *both* pipelines. Generating the decomposition instead of
    hard-coding it is not free, and pretending otherwise would hide the price
    of the generality. It is charged once per distinct instruction rather than
    once per episode, because the decomposition does not change between
    episodes of the same task -- see PlanCache.
    """
    system = count_text_tokens(PLAN_SYSTEM)
    text = count_text_tokens(plan_prompt(obs))
    imgs = obs.images()
    return (
        system + text + count_image_tokens(imgs, VISION),
        count_text_tokens(_PLAN_EXAMPLE),
        len(imgs),
        system,
    )


# -- localize -----------------------------------------------------------------
def localize_prompt(obs: Observation, refs: Sequence[str]) -> str:
    return (
        f"instruction: {obs.prompt}\n"
        f"localise: {', '.join(refs) if refs else 'target'}\n"
        f"robot_qpos: {_fmt(obs.state)}\n"
        f"cameras: head_left, head_right, wrist\n"
        f"step: {obs.time_step}"
    )


_REF_EXAMPLE = (
    '"target": {"origin": [1.234, -0.567, 1.102], "normal": [-1.000, 0.000, '
    '0.000], "extent": [0.070, 0.065, 0.190], "axis": [0.000, 0.000, 1.000]}'
)
_OBSTACLES_EXAMPLE = (
    '"obstacles": [[1.23, -0.65, 1.10], [1.23, -0.48, 1.10], '
    '[1.23, -0.57, 1.35]], "confidence": 0.87'
)


def localize_cost(
    obs: Observation, refs: Sequence[str]
) -> tuple[int, int, int, int]:
    """Cost of grounding every reference the plan names, in one call.

    Grounding several references together is both realistic and much cheaper
    than one call each, given that each call re-sends three images.
    """
    system = count_text_tokens(LOCALIZE_SYSTEM)
    text = count_text_tokens(localize_prompt(obs, refs))
    imgs = obs.images()
    out = count_text_tokens(_REF_EXAMPLE) * max(1, len(refs)) + count_text_tokens(
        _OBSTACLES_EXAMPLE
    )
    return (
        system + text + count_image_tokens(imgs, VISION),
        out,
        len(imgs),
        system,
    )


# -- fine action --------------------------------------------------------------
def act_prompt(obs: Observation, subgoal: Subgoal, goal: Waypoint) -> str:
    return (
        f"instruction: {obs.prompt}\n"
        f"subgoal: {subgoal.name} ({subgoal.kind.value})\n"
        f"tolerance_mm: {subgoal.tolerance_mm:.1f}\n"
        f"commanded_position: {_fmt(goal.position)}\n"
        f"commanded_approach: {_fmt(goal.approach)}\n"
        f"commanded_closing: {_fmt(goal.closing)}\n"
        f"gripper: {goal.gripper:+.0f}\n"
        f"robot_qpos: {_fmt(obs.state)}"
    )


_ACT_EXAMPLE = (
    '{"position": [1.168, 0.049, 1.051], "approach": [1.000, 0.000, 0.000], '
    '"closing": [0.000, 1.000, 0.000]}'
)


def act_cost(
    obs: Observation, subgoal: Subgoal, goal: Waypoint
) -> tuple[int, int, int, int]:
    """Cost of one fine-motion correction.

    The wrist camera is what matters for a millimetre correction, so only it
    and one head camera are sent. At $50 per million output tokens a frontier
    model's cost here is dominated by the short JSON pose it returns, not by
    the images.
    """
    system = count_text_tokens(ACT_SYSTEM)
    text = count_text_tokens(act_prompt(obs, subgoal, goal))
    imgs = [obs.wrist_image, obs.image]
    return (
        system + text + count_image_tokens(imgs, VISION),
        count_text_tokens(_ACT_EXAMPLE),
        len(imgs),
        system,
    )
