"""Stationary Franka Panda on a fixed pedestal in front of a RoboBenchMart mini shelf.

The benchmark's PickToBasket task assumes a Fetch with a basket attached to its
base. For a stationary arm there is no robot-mounted basket, so this env:

  * spawns ``panda_wristcam`` (Panda + wrist RGB-D camera) on top of a static
    pedestal, a fixed distance in front of the active shelf, with no pose
    randomisation (the robot never moves its base, by construction);
  * puts a fixed basket on the pedestal's side shelf, inside the arm's reach;
  * redefines the goal (``calc_target_pose``) as that basket.

Everything else -- scene layout, shelves, products, textures, the success
checks (target in goal box, other items untouched, robot static) -- is the
benchmark's own ``PickToBasketContEnv``. Nothing in RoboBenchMart is edited.

Register by importing this module after ``dsynth.envs``.
"""
from __future__ import annotations

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from dsynth.envs.pick_to_basket import PickToBasketContEnv

# Geometry (metres). Mini shelf: 1.07 wide x 0.83 deep x 1.55 tall, origin at
# the bottom centre; its +y axis points away from the robot side.
SHELF_HALF_DEPTH = 0.415
PEDESTAL_HEIGHT = 0.55          # Panda base height above the floor
STANDOFF = 0.40                 # shelf front face -> Panda base
# The column top sits 2 mm below the Panda base. Flush contact would make the
# motion planner see the start state as colliding (link0 vs pedestal) and
# refuse every plan; the base is a fixed root, so the gap is physically inert.
PEDESTAL_GAP = 0.002
PEDESTAL_HALF = (0.22, 0.22)    # top plate of the robot column
SIDE_TABLE_HALF = (0.20, 0.22)  # basket table, to the robot's right
BASKET_HALF = (0.14, 0.11, 0.07)
BASKET_WALL = 0.008
# Basket centre in the robot base frame (x forward towards the shelf, y left).
BASKET_IN_BASE = (0.05, -0.42, 0.0)


@register_env("PickToBasketPandaPedestal-v0", max_episode_steps=200000)
class PickToBasketPandaPedestalEnv(PickToBasketContEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda"]
    ROBOT_INIT_POSE_RANDOM_ENABLED = False

    # "Ready" pose: TCP ~9 cm in front of the shelf face, 0.5 m above the
    # base, pitched ~33 deg down so the wrist camera sees the middle board.
    # Found by grid search over (q2, q4, q6) -- sim/probe_setup.py.
    INIT_QPOS = np.array([0.0, -1.4, 0.0, -3.0, 0.0, 2.6, np.pi / 4, 0.04, 0.04])

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- cameras
    @property
    def _default_human_render_camera_configs(self):
        from mani_skill.sensors.camera import CameraConfig
        base = super()._default_human_render_camera_configs
        base = base if isinstance(base, (list, tuple)) else [base]
        # free camera for overview videos; the script moves it every frame
        return list(base) + [CameraConfig("overview", pose=sapien.Pose([0, 0, 3]),
                                          width=self.OVERVIEW_WH[0], height=self.OVERVIEW_WH[1],
                                          fov=np.deg2rad(55), near=0.05, far=100)]

    OVERVIEW_WH = (1280, 720)

    @property
    def _default_sensor_configs(self):
        """Two fixed third-person cameras, the Panda analogue of Fetch's
        left/right base cameras (256x256): mounted on panda_link0, behind and
        above the shoulder, looking at the middle board. Rigid with the base,
        so they never move -- like the pedestal.
        """
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils import sapien_utils
        mount = self.agent.robot.links_map["panda_link0"]
        cfgs = []
        for uid, y in (("scene_left", 0.45), ("scene_right", -0.45)):
            pose = sapien_utils.look_at(eye=[-0.35, y, 0.75], target=[0.55, 0.0, 0.1])
            cfgs.append(CameraConfig(uid, pose=pose, width=256, height=256,
                                     fov=np.deg2rad(70), near=0.01, far=10, mount=mount))
        return cfgs

    # ------------------------------------------------------------------ scene
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.pedestal = self._build_box_actor(
            "pedestal",
            [(PEDESTAL_HALF[0], PEDESTAL_HALF[1], (PEDESTAL_HEIGHT - PEDESTAL_GAP) / 2, (0, 0, 0))],
            color=[0.22, 0.24, 0.27, 1],
        )
        table_h = PEDESTAL_HEIGHT - PEDESTAL_GAP
        self.side_table = self._build_box_actor(
            "side_table",
            [(SIDE_TABLE_HALF[0], SIDE_TABLE_HALF[1], table_h / 2, (0, 0, 0))],
            color=[0.30, 0.32, 0.35, 1],
        )
        bx, by, bz = BASKET_HALF
        w = BASKET_WALL
        self.basket = self._build_box_actor(
            "basket",
            [
                (bx, by, w, (0, 0, -bz + w)),             # bottom
                (bx, w, bz, (0, by - w, 0)),              # left
                (bx, w, bz, (0, -by + w, 0)),             # right
                (w, by, bz, (bx - w, 0, 0)),              # front
                (w, by, bz, (-bx + w, 0, 0)),             # back
            ],
            color=[0.85, 0.15, 0.12, 1],
        )

    def _build_box_actor(self, name, boxes, color):
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=color, roughness=0.6)
        for hx, hy, hz, off in boxes:
            pose = sapien.Pose(p=list(off))
            builder.add_box_collision(pose=pose, half_size=[hx, hy, hz])
            builder.add_box_visual(pose=pose, half_size=[hx, hy, hz], material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, -10])
        return builder.build_kinematic(name=name)

    # ------------------------------------------------------------------ robot
    def _compute_robot_init_pose(self, env_idx=None):
        origins, angles, dirs = [], [], []
        for idx in env_idx:
            idx = idx.cpu().item()
            shelf = self.actors["fixtures"]["shelves"][self.active_shelves[idx][0]]
            sp = shelf.pose.sp
            d = sp.to_transformation_matrix()[:3, 1]
            d = d / np.linalg.norm(d)
            origin = sp.p - (SHELF_HALF_DEPTH + STANDOFF) * d
            origin[2] = PEDESTAL_HEIGHT
            angle = np.arctan2(d[1], d[0])
            origins.append(origin)
            angles.append(angle)
            dirs.append(d)
        return np.array(origins), np.array(angles), np.array(dirs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.agent.reset(self.INIT_QPOS)
        q = np.array([euler2quat(0, 0, a) for a in self.robot_angles])
        base = Pose.create_from_pq(p=self.robot_origins, q=q)
        self.agent.robot.set_pose(base)

        floor = self.robot_origins.copy()
        floor[:, 2] = 0.0
        floor_pose = Pose.create_from_pq(p=floor, q=q)
        self.pedestal.set_pose(floor_pose * Pose.create_from_pq(p=[0, 0, (PEDESTAL_HEIGHT - PEDESTAL_GAP) / 2]))
        # side table: right of the robot, flush with the pedestal top
        self.side_table.set_pose(
            floor_pose * Pose.create_from_pq(
                p=[BASKET_IN_BASE[0], BASKET_IN_BASE[1], (PEDESTAL_HEIGHT - PEDESTAL_GAP) / 2]))
        self.basket.set_pose(base * Pose.create_from_pq(
            p=[BASKET_IN_BASE[0], BASKET_IN_BASE[1], BASKET_HALF[2]]))

    # ------------------------------------------------------------------- goal
    REACH_MAX_X = 0.75  # base-frame depth limit; the far face of the shelf is ~0.95
    # Base-frame height band. Measured with sim/debug_plan.py (collision-aware
    # IK to a 10 cm standoff): the middle board (z +0.11) is reachable, the
    # bottom board (z -0.34) almost never is from a 0.55 m pedestal -- Fetch
    # covers it with its torso lift, the Panda has none.
    REACH_Z = (-0.10, 0.35)

    def setup_target_objects(self, env_idxs):
        """Sample the target only among items on the robot's face of the shelf.

        The mini shelf is two-sided and the benchmark samples the target from
        every item in the scene, so about half the episodes would ask a fixed
        arm for an item behind the back panel; and the bottom board is below
        what a pedestal Panda can reach without a torso. Same rule otherwise: uniform over
        product names, success counts any instance of that name.
        """
        if self.TARGET_PRODUCT_NAME is not None:
            return super().setup_target_objects(env_idxs)
        origins, angles, _ = self._compute_robot_init_pose(env_idxs)
        import pandas as pd
        self.target_product_names, frames = {}, []
        for k, idx in enumerate(env_idxs):
            idx = idx.cpu().item()
            T = np.linalg.inv(sapien.Pose(origins[k], euler2quat(0, 0, angles[k])).to_transformation_matrix())
            df = self.products_df[self.products_df["scene_idx"] == idx]
            def in_reach(n):
                x, _, z = (T @ np.r_[self.actors["products"][n].pose.p[0].cpu().numpy(), 1])[:3]
                return x < self.REACH_MAX_X and self.REACH_Z[0] < z < self.REACH_Z[1]
            near = [n for n in df["actor_name"] if in_reach(n)]
            names = sorted(df[df["actor_name"].isin(near)]["product_name"].unique())
            if not names:
                raise RuntimeError(f"scene #{idx}: no product on the robot side of the shelf")
            name = self._batched_episode_rng[idx].choice(names)
            self.target_product_names[idx] = name
            frames.append(df[df["product_name"] == name])
        self.target_products_df = pd.concat(frames)

    def calc_target_pose(self):
        return self.basket.pose

    def setup_language_instructions(self, env_idx):
        self.language_instructions = [
            f"pick {self.target_product_names[i.cpu().item()]} from the shelf and place it in the basket"
            for i in env_idx
        ]
