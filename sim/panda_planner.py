"""Collision-aware motion planner for the pedestal Panda, on mplib 0.2.1.

Why not ManiSkill's PandaArmMotionPlanningSolver: ManiSkill 3.0.1's solver is
written against mplib 0.1.1, while RoboBenchMart's dsynth imports mplib 0.2.1
(``mplib.sapien_utils``) -- the two cannot share an environment. This mirrors
dsynth's own ``FetchMotionPlanningSapienSolver`` instead: the planning world is
built from the live SAPIEN scene, so the shelf, its boards, the neighbouring
products, the pedestal and the basket are all obstacles. Only the target items
are allowed to touch the robot (it has to grasp one of them).

Same method surface the executor needs as the Fetch solver:
move_to_pose_with_screw / move_to_pose_with_RRTConnect / open_gripper /
close_gripper / idle_steps / elapsed_steps / gripper_state.
"""
from __future__ import annotations

import mplib
import numpy as np
import sapien
from mani_skill.utils.structs.pose import to_sapien_pose
from mplib.sapien_utils import SapienPlanner
from mplib.sapien_utils.conversion import convert_object_name

import inspect

import dsynth.planning.utils as _dsu


def _make_panda_planning_world():
    """dsynth's SapienPlanningWorldV2 (scaled-mesh fix) minus its Fetch-only
    assumptions: it seeds and refreshes the planned model with ``qpos[4:]``
    (skipping Fetch's leading base/torso joints), and update_from_simulation
    re-roots the model at Fetch's torso link (``links[7]``, asserted by name).
    For the Panda the planned qpos is the full vector and the root is link0.
    Built from the upstream source with those expressions replaced, so
    RoboBenchMart stays unedited; the asserts fail loudly if upstream changes."""
    src = inspect.getsource(_dsu.SapienPlanningWorldV2)
    patches = [
        # (fetch-only expression, panda replacement, expected occurrences)
        ("articulation.qpos[4:]", "articulation.qpos", 2),
        ("root_pose = articulation.links[7].pose", "root_pose = articulation.root_pose", 1),
        ("assert articulation.links[7].name == 'scene-0-ds_fetch_basket_torso_lift_link'", "pass", 1),
    ]
    for old, new, n in patches:
        assert src.count(old) == n, f"dsynth SapienPlanningWorldV2 changed ({old!r}); re-check the patch"
        src = src.replace(old, new)
    src = src.replace("class SapienPlanningWorldV2(", "class PandaPlanningWorld(", 1)
    ns = dict(vars(_dsu))
    exec(compile(src, _dsu.__file__ + ":PandaPlanningWorld", "exec"), ns)
    return ns["PandaPlanningWorld"]


PandaPlanningWorld = _make_panda_planning_world()


class PandaPedestalPlanner:
    OPEN = 1
    CLOSED = -1
    MOVE_GROUP = "panda_hand_tcp"

    def __init__(self, env, *, joint_vel_limits=0.9, joint_acc_limits=0.9,
                 allow_target_contact=True, obstacles=True, verbose=False):
        self.env = env
        self.base_env = env.unwrapped
        self.agent = self.base_env.agent
        self.robot = self.agent.robot
        self.control_mode = self.base_env.control_mode
        assert self.control_mode == "pd_joint_pos", self.control_mode
        self.verbose = verbose
        self.elapsed_steps = 0
        self.gripper_state = self.OPEN

        scene = self.base_env.scene.sub_scenes[0]
        art = self.robot._objs[0]
        world = PandaPlanningWorld(
            scene,
            user_link_names=[l.get_name() for l in self.robot.get_links()],
            user_joint_names=[j.get_name() for j in self.robot.get_active_joints()],
            planned_articulations=[art],
            planned_urdf_paths=[self.agent.urdf_path],
            disable_actors_collision=not obstacles,
            new_package_keyword="",
            use_convex=False,
            verbose=False,
        )
        if allow_target_contact and obstacles:
            acm = world.get_allowed_collision_matrix()
            df = self.base_env.target_products_df
            for name in df[df["scene_idx"] == 0]["actor_name"]:
                actor = self.base_env.actors["products"][name]
                acm.set_default_entry(convert_object_name(actor._objs[0]), True)
        self.planner = SapienPlanner(
            world, self.MOVE_GROUP,
            joint_vel_limits=np.ones(7) * joint_vel_limits,
            joint_acc_limits=np.ones(7) * joint_acc_limits,
        )
        bp = self.robot.pose.sp
        self.planner.set_base_pose(mplib.Pose(bp.p, bp.q))

    # ---------------------------------------------------------------- motion
    def _arm_qpos(self):
        return self.robot.get_qpos().cpu().numpy()[0]

    def follow_path(self, result):
        res = None
        for qpos in result["position"]:
            res = self.env.step(np.hstack([qpos, self.gripper_state]))
            self.elapsed_steps += 1
        self.planner.update_from_simulation()
        return res

    def move_to_pose_with_screw(self, pose):
        pose = to_sapien_pose(pose)
        for _ in range(2):  # dsynth's solver retries screw once
            result = self.planner.plan_screw(
                mplib.Pose(p=pose.p, q=pose.q), self._arm_qpos(),
                time_step=self.base_env.control_timestep)
            if result["status"] == "Success":
                return self.follow_path(result)
        if self.verbose:
            print("screw:", result["status"])
        return -1

    def move_to_pose_with_RRTConnect(self, pose):
        pose = to_sapien_pose(pose)
        result = self.planner.plan_pose(
            mplib.Pose(p=pose.p, q=pose.q), self._arm_qpos(),
            time_step=self.base_env.control_timestep, wrt_world=True,
            planning_time=2, rrt_range=0.1, simplify=True)
        if result["status"] != "Success":
            if self.verbose:
                print("rrt:", result["status"])
            return -1
        return self.follow_path(result)

    # --------------------------------------------------------------- gripper
    def _hold(self, t):
        arm = self.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        res = None
        for _ in range(int(t)):
            res = self.env.step(np.hstack([arm, self.gripper_state]))
            self.elapsed_steps += 1
        return res

    def open_gripper(self, t=6):
        self.gripper_state = self.OPEN
        return self._hold(t)

    def close_gripper(self, t=6):
        self.gripper_state = self.CLOSED
        return self._hold(t)

    def idle_steps(self, t=20):
        return self._hold(t)
