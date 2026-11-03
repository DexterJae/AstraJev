"""Privileged-state access, used only by the mock backends.

The real backends (jev_api, astra_api) never touch this: they see exactly
the observation dict the benchmark's policy-server contract provides.

What this means for the claim:
  * the COST half of the comparison is exact under the mock backends, because
    cost is computed from the prompts the pipelines really construct;
  * the SUCCESS-RATE half is a controlled simulation whose outcome is set by
    the noise parameters in config/default.yaml `mock:`, and must be replaced
    by real backends before being reported as a result.

The interface is plan-shaped rather than task-shaped: `frames()` returns every
scene reference a generated plan might name, so a new task needs a new entry
here, not new pipeline code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .plan import Frame, TaskCapabilities


@runtime_checkable
class Oracle(Protocol):
    def frames(self) -> dict[str, Frame]:
        """Every scene reference a plan may name, keyed by name."""
        ...

    def free_ref_names(self) -> tuple[str, ...]:
        """Which of those are obtainable without a model call."""
        ...

    def capabilities(self) -> TaskCapabilities:
        """Tolerance-setting quantities measured from the environment."""
        ...

    def obstacles(self) -> list:
        """Centres of items that must not be displaced."""
        ...

    def tcp_position(self) -> np.ndarray: ...
    def nearest_neighbour_distance(self) -> float: ...
    def gripper_closed(self) -> bool: ...
    def holding_item(self) -> bool: ...
    def target_label(self) -> str: ...


class ManiSkillOracle:
    """Reads privileged state out of a live RoboBenchMart environment.

    Depends on ManiSkill/sapien and on the env internals used by
    dsynth/planning/fetch_skills.py, so imports are lazy.

    Which references exist depends on the task, and that is read off the
    environment rather than hard-coded: a PnP env exposes a target item and a
    destination, an articulated env exposes a handle and a hinge.
    """

    #: Frames the robot knows for free: rigidly attached to its own base.
    FREE_REFS = ("basket",)

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

    def _obb(self, actor):
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            get_actor_obb,
        )

        return get_actor_obb(actor)

    def _shelf_normal(self) -> np.ndarray:
        # `directions_to_shelf` points from the robot into the shelf; the
        # outward normal is its negation.
        d = np.asarray(self._env().directions_to_shelf[0], dtype=np.float64).reshape(3)
        return -d

    # -- Oracle ------------------------------------------------------------
    def frames(self) -> dict[str, Frame]:
        env = self._env()
        out: dict[str, Frame] = {}

        # The manipulation target, when the task has one.
        try:
            obb = self._obb(self._target())
            out["target"] = Frame(
                origin=np.array(obb.primitive.transform)[:3, 3].astype(np.float64),
                normal=self._shelf_normal(),
                extent=np.asarray(obb.primitive.extents, dtype=np.float64),
            )
        except Exception:
            pass

        # The basket, rigidly attached to the base, hence free.
        if hasattr(env, "calc_target_pose"):
            try:
                p = np.asarray(env.calc_target_pose().sp.p, dtype=np.float64).reshape(3)
                out["basket"] = Frame(origin=p, normal=np.array([0.0, 0.0, 1.0]))
            except TypeError:
                # Some envs take an actor name; those tasks use "goal_site".
                pass

        # A destination on a shelf board, for board-to-board and floor tasks.
        for attr in ("goal_positions", "target_positions"):
            if hasattr(env, attr):
                try:
                    g = np.asarray(getattr(env, attr)[0], dtype=np.float64).reshape(3)
                    out["goal_site"] = Frame(
                        origin=g, normal=np.array([0.0, 0.0, 1.0])
                    )
                    break
                except Exception:
                    pass

        # Articulated tasks: handle and hinge.
        for name, key in (("handle", "handle_link"), ("hinge", "hinge_link")):
            link = getattr(env, key, None)
            if link is not None:
                try:
                    p = np.asarray(link.pose.p.cpu().numpy()).reshape(-1, 3)[0]
                    out[name] = Frame(
                        origin=p.astype(np.float64),
                        normal=self._shelf_normal(),
                        axis=np.array([0.0, 0.0, 1.0]),
                    )
                except Exception:
                    pass

        return out

    def free_ref_names(self) -> tuple[str, ...]:
        return self.FREE_REFS

    def capabilities(self) -> TaskCapabilities:
        env = self._env()
        extent = np.array([0.065, 0.065, 0.19])
        try:
            extent = np.asarray(
                self._obb(self._target()).primitive.extents, dtype=np.float64
            )
        except Exception:
            pass
        return TaskCapabilities(
            # The task's own success threshold, declared by the environment.
            place_tolerance_m=float(getattr(env, "TARGET_POS_THRESH", 0.2)),
            finger_length_m=0.03,
            # Fetch's parallel gripper stroke, from its URDF joint limits.
            gripper_stroke_m=0.1,
            neighbour_gap_m=self.neighbour_gap(),
            item_width_m=float(np.min(extent[:2])),
        )

    def neighbour_gap(self) -> float:
        """Centre-to-centre distance from the target to its closest neighbour."""
        env = self._env()
        try:
            obb = self._obb(self._target())
            origin = np.array(obb.primitive.transform)[:3, 3].astype(np.float64)
        except Exception:
            return 0.085
        target_names = set(
            env.target_products_df[env.target_products_df["scene_idx"] == 0][
                "actor_name"
            ]
        )
        best = float("inf")
        for name, actor in env.actors["products"].items():
            if name in target_names:
                continue
            p = np.asarray(actor.pose.p.cpu().numpy()).reshape(-1, 3)[0]
            best = min(best, float(np.linalg.norm(p - origin)))
        return best if np.isfinite(best) else 0.085

    def tcp_position(self) -> np.ndarray:
        from dsynth.planning.utils import get_tcp_pose

        return np.asarray(get_tcp_pose(self.env).p, dtype=np.float64).reshape(3)

    def obstacles(self) -> list:
        env = self._env()
        target_names = set(
            env.target_products_df[env.target_products_df["scene_idx"] == 0][
                "actor_name"
            ]
        )
        out = []
        for name, actor in env.actors["products"].items():
            if name in target_names:
                continue
            p = np.asarray(actor.pose.p.cpu().numpy()).reshape(-1, 3)[0]
            out.append(p.astype(np.float64))
        return out

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
            p = np.asarray(actor.pose.p.cpu().numpy()).reshape(-1, 3)[0]
            best = min(best, float(np.linalg.norm(p - tcp)))
        return best if np.isfinite(best) else 10.0

    def gripper_closed(self) -> bool:
        qpos = np.asarray(self._env().agent.robot.get_qpos().cpu().numpy()).reshape(-1)
        return bool(qpos[-1] < 0.01)

    def holding_item(self) -> bool:
        try:
            return bool(self._env().agent.is_grasping(self._target())[0].item())
        except Exception:
            # Fallback if the agent has no is_grasping: fingers closed and the
            # target still within a finger length of the gripper.
            if not self.gripper_closed():
                return False
            f = self.frames().get("target")
            if f is None:
                return False
            return float(np.linalg.norm(f.origin - self.tcp_position())) < 0.08

    def target_label(self) -> str:
        return str(self._env().target_product_names[0])
