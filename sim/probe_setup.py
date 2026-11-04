"""Probe: product heights on the active shelf + TCP pose for candidate Panda qpos."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, str(Path(__file__).resolve().parent))
import gymnasium as gym, mani_skill.envs, dsynth.envs, dsynth.robots, panda_pedestal_env  # noqa
env = gym.make("PickToBasketPandaPedestal-v0", config_dir_path=sys.argv[1], num_envs=1,
               obs_mode="none", render_mode="rgb_array", sim_backend="cpu")
env.reset(seed=0, options={"reconfigure": True}); u = env.unwrapped
base = u.agent.robot.pose.sp; T = np.linalg.inv(base.to_transformation_matrix())
print("control_mode", u.control_mode, "action", env.action_space.shape)
print("ceiling-like actors:", [n for n in u.scene.actors if "ceil" in n.lower()][:10])
rows = []
for name, a in u.actors["products"].items():
    p = (T @ np.r_[a.pose.p[0].cpu().numpy(), 1])[:3]
    rows.append((round(p[2], 2), round(p[0], 2), round(p[1], 2), name.split(":")[-1][:40]))
rows.sort()
print("products in robot base frame (z, x fwd, y left):")
for r in rows: print("  ", r)
print("target:", u.target_product_names[0])
best=[]
import itertools
for q2,q4,q6 in itertools.product(np.linspace(-1.6,0.6,23),np.linspace(-3.0,-1.2,19),np.linspace(1.0,3.7,28)):
    q=[0,q2,0,q4,0,q6,np.pi/4]
    u.agent.robot.set_qpos(np.array(q+[0.04,0.04]))
    m = T @ u.agent.tcp.pose.sp.to_transformation_matrix()
    p, z = m[:3,3], m[:3,2]
    cost = np.linalg.norm(p-np.array([0.10,0,0.30])) + 0.5*np.linalg.norm(z-np.array([0.87,0,-0.5]))
    best.append((cost, np.round(q,3).tolist(), np.round(p,2).tolist(), np.round(z,2).tolist()))
best.sort(key=lambda r:r[0])
for b in best[:3]: print("ready", b)
