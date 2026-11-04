"""Sanity: target reachability + success check wired to the fixed basket, per seed."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, str(Path(__file__).resolve().parent))
from mani_skill.utils.structs.pose import Pose
import gymnasium as gym, mani_skill.envs, dsynth.envs, dsynth.robots, panda_pedestal_env  # noqa
env = gym.make("PickToBasketPandaPedestal-v0", config_dir_path=sys.argv[1], num_envs=1,
               obs_mode="none", render_mode="rgb_array", sim_backend="cpu")
for seed in range(6):
    env.reset(seed=seed, options={"reconfigure": True}); u = env.unwrapped
    T = np.linalg.inv(u.agent.robot.pose.sp.to_transformation_matrix())
    df = u.target_products_df
    tgt = [(T @ np.r_[u.actors["products"][n].pose.p[0].cpu().numpy(), 1])[:3] for n in df["actor_name"]]
    near = [p for p in tgt if p[0] < 0.75]
    info = u.evaluate()
    # teleport one target into the basket -> success must flip to True
    a = u.actors["products"][df["actor_name"].iloc[0]]
    a.set_pose(u.basket.pose * Pose.create_from_pq(p=[0, 0, 0.02]))
    for _ in range(20): env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    info2 = u.evaluate()
    print(f"seed {seed}: target={u.target_product_names[0]!r:40s} n={len(tgt)} reachable_side={len(near)} "
          f"closest={np.round(min(tgt, key=lambda p: p[0]), 2)} | success@start={bool(info['success'])} "
          f"| success after placing in basket={bool(info2['success'])}")
