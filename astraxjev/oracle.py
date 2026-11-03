"""Privileged-state access, used only by the mock backends.

The real backends (astra_http, jev_http) never touch this: they see exactly the
observation dict the benchmark's policy-server contract provides. The oracle
exists so the pipelines can be run end to end, and the cost ledger exercised,
on a machine that does not have Astra or Jev weights.

What this means for the claim:
  * the COST half of the comparison is exact under the mock backends, because
    cost is computed from the prompts the pipelines really construct;
  * the SUCCESS-RATE half is a controlled simulation whose outcome is set by
    the noise parameters in config/default.yaml `mock:`, and must be replaced
    by real backends before it is reported as a result.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Oracle(Protocol):
    def target_position(self) -> np.ndarray: ...
    def target_extent(self) -> np.ndarray: ...
    def shelf_normal(self) -> np.ndarray: ...
    def tcp_position(self) -> np.ndarray: ...
    def basket_position(self) -> np.ndarray: ...
    def nearest_neighbour_distance(self) -> float: ...
    def gripper_closed(self) -> bool: ...
    def holding_item(self) -> bool: ...
    def target_label(self) -> str: ...


class ManiSkillOracle:
    """Reads privileged state out of a live RoboBenchMart environment.

    Depends on ManiSkill/sapien and on the env internals used by
    dsynth/planning/fetch_skills.py, so it is imported lazily.
    """

    def __init__(self, env) -> None:
        self.env = env
        self._unwrapped = getattr(env, "unwrapped", env)
        self._target_actor = None

    # -- helpers -----------------------------------------------------------
    def _env(self):
        return self._unwrapped

    def _target(self):
        if self._target_actor is None:
            env = self._env()
            df = env.target_products_df
            rows = df[df["scene_idx"] == 0]
            if len(rows) == 0:
                raise RuntimeError("no target product in scene 0")
            self._target_actor = env.actors["products"][rows["actor_name"].iloc[0]]
        return self._target_actor

    # -- Oracle ------------------------------------------------------------
    def target_position(self) -> np.ndarray:
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            get_actor_obb,
        )

        obb = get_actor_obb(self._target())
        return np.array(obb.primitive.transform)[:3, 3].astype(np.float64)

    def target_extent(self) -> np.ndarray:
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            get_actor_obb,
        )

        obb = get_actor_obb(self._target())
        return np.asarray(obb.primitive.extents, dtype=np.float64)

    def shelf_normal(self) -> np.ndarray:
        # `directions_to_shelf` points from the robot into the shelf; the
        # outward normal is its negation.
        d = np.asarray(self._env().directions_to_shelf[0], dtype=np.float64).reshape(3)
        return -d

    def tcp_position(self) -> np.ndarray:
        from dsynth.planning.utils import get_tcp_pose

        return np.asarray(get_tcp_pose(self.env).p, dtype=np.float64).reshape(3)

    def basket_position(self) -> np.ndarray:
        return np.asarray(
            self._env().calc_target_pose().sp.p, dtype=np.float64
        ).reshape(3)

    def nearest_neighbour_distance(self) -> float:
        env = self._env()
        tcp = self.tcp_position()
        target_names = set(
            env.target_products_df[env.target_products_df["scene_idx"] == 0][
                "actor_name"
            ]
        )
        best = float("inf")
        for name, actor in env.actors["products"].items():
            if name in target_names:
                continue
            p = np.asarray(actor.pose.p.cpu().numpy(), dtype=np.float64).reshape(-1, 3)[0]
            best = min(best, float(np.linalg.norm(p - tcp)))
        return best if np.isfinite(best) else 10.0

    def gripper_closed(self) -> bool:
        qpos = np.asarray(self._env().agent.robot.get_qpos().cpu().numpy()).reshape(-1)
        return bool(qpos[-1] < 0.01)

    def holding_item(self) -> bool:
        return bool(self._env().agent.is_grasping(self._target())[0].item())

    def target_label(self) -> str:
        return str(self._env().target_product_names[0])
