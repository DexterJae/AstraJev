"""Prompt construction for the Astra backends.

These functions exist so the cost of an Astra call is *measured*. Both the
HTTP backend and the mock backend build the prompt through here, so a mock run
reports the same token counts a real run would, and the dollar figures carry
over unchanged when real weights are plugged in.
"""

from __future__ import annotations

import numpy as np

from .tokens import VisionTokenizerSpec, count_image_tokens, count_text_tokens
from .types import Observation, Phase, TargetEstimate, Waypoint

#: Astra-Global is an MLLM, so its images go through a pooled ViT tower.
GLOBAL_VISION = VisionTokenizerSpec(patch_size=14, pool_factor=2)

#: Astra-Local's 4D spatial-temporal encoder is small but unpooled, so it sees
#: more feature tokens per frame while costing far fewer FLOPs per token.
LOCAL_VISION = VisionTokenizerSpec(patch_size=16, pool_factor=1, extra_tokens_per_image=0)

GLOBAL_SYSTEM = (
    "You are Astra-Global. Given the robot's head cameras and wrist camera and a "
    "language instruction, localise the referenced retail item on the shelf. "
    "Return JSON with keys: position (3 floats, world frame, metres), extent "
    "(3 floats, bounding box size in metres), shelf_normal (3 floats, unit "
    "vector pointing out of the shelf face), confidence (0-1), label (string). "
    "Items are densely packed; identify the one matching the instruction by "
    "brand text and packaging, not by position alone."
)

LOCAL_SYSTEM = (
    "You are Astra-Local. Given the current observation, the localised target "
    "and a goal end-effector pose, emit the next end-effector waypoint chunk "
    "that makes progress toward the goal without colliding with the shelf or "
    "neighbouring items. The mobile base is locked."
)


def _fmt(v) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"


def global_prompt(obs: Observation) -> str:
    return (
        f"{GLOBAL_SYSTEM}\n"
        f"instruction: {obs.prompt}\n"
        f"robot_qpos: {_fmt(obs.state)}\n"
        f"cameras: head_left, head_right, wrist\n"
        f"step: {obs.time_step}"
    )


def global_output_tokens() -> int:
    """Token count of the structured answer Astra-Global returns.

    Counted off a representative response rather than guessed, so that changing
    the schema changes the cost.
    """
    example = (
        '{"position": [1.234, -0.567, 1.102], "extent": [0.070, 0.065, 0.190], '
        '"shelf_normal": [-1.000, 0.000, 0.000], "confidence": 0.87, '
        '"label": "Nivea Body Milk"}'
    )
    return count_text_tokens(example)


def global_cost(obs: Observation) -> tuple[int, int, int]:
    """(tokens_in, tokens_out, n_images) for one Astra-Global call."""
    text = count_text_tokens(global_prompt(obs))
    imgs = obs.images()
    return text + count_image_tokens(imgs, GLOBAL_VISION), global_output_tokens(), len(imgs)


def local_prompt(
    obs: Observation, target: TargetEstimate, phase: Phase, goal: Waypoint, chunk_len: int
) -> str:
    return (
        f"{LOCAL_SYSTEM}\n"
        f"instruction: {obs.prompt}\n"
        f"phase: {phase.value}\n"
        f"target_position: {_fmt(target.position)}\n"
        f"target_extent: {_fmt(target.extent)}\n"
        f"shelf_normal: {_fmt(target.shelf_normal)}\n"
        f"goal_position: {_fmt(goal.position)}\n"
        f"goal_approach: {_fmt(goal.approach)}\n"
        f"goal_closing: {_fmt(goal.closing)}\n"
        f"gripper: {goal.gripper:+.0f}\n"
        f"robot_qpos: {_fmt(obs.state)}\n"
        f"chunk_len: {chunk_len}"
    )


def local_cost(
    obs: Observation,
    target: TargetEstimate,
    phase: Phase,
    goal: Waypoint,
    chunk_len: int,
) -> tuple[int, int, int]:
    """(tokens_in, tokens_out, n_images) for one Astra-Local call.

    Output tokens are zero: Astra-Local is a multitask network with regression
    heads, not an autoregressive decoder, so there is nothing to decode. Its
    cost is pure prefill over feature tokens.
    """
    text = count_text_tokens(
        local_prompt(obs, target, phase, goal, chunk_len)
    )
    imgs = obs.images()
    return text + count_image_tokens(imgs, LOCAL_VISION), 0, len(imgs)


DETECTOR_PROMPT = (
    "open-vocabulary detector: locate the target item and the nearest "
    "non-target item; return coarse centroids in centimetres"
)


def detector_cost(obs: Observation) -> tuple[int, int, int]:
    """Cost of the cheap perception the hybrid needs to feed Jev text.

    Jev reads text, not pixels, so the hybrid is charged for a small detector
    pass on every decision tick. Leaving this out would be the easiest way to
    accidentally overstate the hybrid's advantage.
    """
    text = count_text_tokens(DETECTOR_PROMPT)
    imgs = [obs.image, obs.wrist_image]
    spec = VisionTokenizerSpec(patch_size=16, pool_factor=2, extra_tokens_per_image=0)
    return text + count_image_tokens(imgs, spec), 0, len(imgs)
