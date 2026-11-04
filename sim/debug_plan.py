"""Why does the first standoff fail? Plan to a few poses in front of the target."""
import sys, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "RoboBenchMart"), str(R / "sim"), str(R)]
import os; os.chdir(R / "RoboBenchMart")
import gymnasium as gym, mani_skill.envs, dsynth.envs, dsynth.robots, panda_pedestal_env  # noqa
from panda_planner import PandaPedestalPlanner
from astraxjev.oracle import ManiSkillOracle
env = gym.make("PickToBasketPandaPedestal-v0", config_dir_path="generated_envs/panda_mini_shelf",
               num_envs=1, control_mode="pd_joint_pos", obs_mode="none", render_mode="rgb_array", sim_backend="cpu")
env.reset(seed=0, options={"reconfigure": True}); u = env.unwrapped
pl = PandaPedestalPlanner(env, verbose=True)
w = pl.planner.planning_world
print("start collisions:", [(c.object_name1, c.link_name1, c.object_name2, c.link_name2, round(c.depth,4) if hasattr(c,"depth") else None) for c in w.check_collision()][:8]); print("link0 z", u.agent.robot.links_map["panda_link0"].pose.p, "pedestal top", u.pedestal.pose.p)
f = ManiSkillOracle(env).frames()["target"]
print("target", np.round(f.origin, 3), "normal", np.round(f.normal, 2), "extent", np.round(f.extent, 3))
closing = np.cross([0, 0, 1], -f.normal); closing /= np.linalg.norm(closing)
for dist in (0.18, 0.10, 0.05):
    pose = u.agent.build_grasp_pose(-f.normal, closing, f.origin + dist * f.normal)
    r = pl.planner.plan_screw(__import__("mplib").Pose(pose.p, pose.q), pl._arm_qpos(), time_step=u.control_timestep)
    ik = pl.planner.IK(__import__("mplib").Pose(pose.p, pose.q), pl._arm_qpos())
    print(f"standoff {dist}: screw={r['status']} ik={ik[0]}")

for label, pl in (("collisions ON", pl), ("collisions OFF", PandaPedestalPlanner(env, obstacles=False))):
  print(label)
  # Reachability map  : for every near-side product, can IK reach a grasp/standoff
  # pose, for a horizontal approach and for 30/45 deg downward approaches?
  import mplib, collections
  T = np.linalg.inv(u.agent.robot.pose.sp.to_transformation_matrix())
  n = -f.normal  # into the shelf
  tab = collections.defaultdict(lambda: collections.Counter())
  for name, a in u.actors["products"].items():
      p = a.pose.p[0].cpu().numpy(); pb = (T @ np.r_[p, 1])[:3]
      if pb[0] > 0.75: continue
      board = {-0.34: "bottom", 0.11: "middle", 0.56: "top"}[min((-0.34, 0.11, 0.56), key=lambda z: abs(z - pb[2]))]
      for deg in (0, 30, 45):
          th = np.deg2rad(deg); app = np.cos(th) * n + np.array([0, 0, -np.sin(th)])
          app /= np.linalg.norm(app); closing = np.cross([0, 0, 1], n); closing /= np.linalg.norm(closing)
          ok_all = True
          for dist in ((0.10,) if label == "collisions ON" else (0.0, 0.10)):
              pose = u.agent.build_grasp_pose(app, closing, p - dist * app)
              st, _ = pl.planner.IK(pl.planner._transform_goal_to_wrt_base(mplib.Pose(pose.p, pose.q)), pl._arm_qpos())
              ok_all &= (st == "Success")
          tab[board][f"{deg}deg"] += int(ok_all); tab[board]["n"] += 0
      tab[board]["items"] += 1
  for b, c in tab.items(): print("reach", b, dict(c))
