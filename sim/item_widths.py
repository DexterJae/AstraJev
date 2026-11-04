"""Graspable width of each reachable (middle-board, robot-side) product vs the Panda stroke."""
import sys, os, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "RoboBenchMart"), str(R / "sim")]; os.chdir(R / "RoboBenchMart")
import gymnasium as gym, mani_skill.envs, dsynth.envs, dsynth.robots, panda_pedestal_env  # noqa
from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb
env = gym.make("PickToBasketPandaPedestal-v0", config_dir_path="generated_envs/panda_mini_shelf",
               num_envs=1, obs_mode="none", render_mode="rgb_array", sim_backend="cpu")
env.reset(seed=0, options={"reconfigure": True}); u = env.unwrapped
T = np.linalg.inv(u.agent.robot.pose.sp.to_transformation_matrix())
seen = {}
for n, a in u.actors["products"].items():
    x, _, z = (T @ np.r_[a.pose.p[0].cpu().numpy(), 1])[:3]
    if x < 0.75 and -0.1 < z < 0.35:
        name = u.products_df[u.products_df.actor_name == n].product_name.iloc[0]
        seen[name] = np.round(sorted(get_actor_obb(a).primitive.extents), 3)
for k, v in seen.items():
    print(f"{k:40s} extents(sorted)={v}  min_horizontal_fits_0.08m_stroke={v[0] < 0.08}")
