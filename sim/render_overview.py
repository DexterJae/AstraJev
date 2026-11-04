"""Video 1: overview of the RoboBenchMart scene with the stationary Panda setup.

Shots (one continuous MP4):
  1. "dollhouse" orbit over the whole store room (ceiling hidden),
  2. descent / push-in to the pedestal Panda in front of the mini shelf,
  3. side-by-side: third-person close-up arc | Panda wrist camera.

Run from RoboBenchMart/ (dsynth loads assets by relative path):
  python ../sim/render_overview.py --scene-dir generated_envs/panda_mini_shelf \
      --out ../results/videos/01_overview.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def look_at(eye, target):
    import sapien
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0, 0, 1], fwd)
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = fwd, left, up, eye
    return sapien.Pose(m)


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def set_visible(actor, visible: bool):
    import sapien
    for obj in actor._objs:
        rb = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        if rb is not None:
            rb.visibility = 1.0 if visible else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="generated_envs/panda_mini_shelf")
    ap.add_argument("--out", default="../results/videos/01_overview.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--shader", default="default", choices=["minimal", "default", "rt-fast", "rt"])
    ap.add_argument("--stills", action="store_true", help="also dump a PNG per shot")
    args = ap.parse_args()

    import gymnasium as gym
    import imageio.v2 as imageio
    import PIL.Image as Image
    import mani_skill.envs  # noqa: F401
    import dsynth.envs  # noqa: F401  (registers benchmark envs)
    import dsynth.robots  # noqa: F401
    import panda_pedestal_env
    panda_pedestal_env.PickToBasketPandaPedestalEnv.OVERVIEW_WH = (args.width, args.height)

    env = gym.make(
        "PickToBasketPandaPedestal-v0",
        robot_uids="panda_wristcam",
        config_dir_path=args.scene_dir,
        num_envs=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="cpu",
        enable_shadow=True,
        sensor_configs=dict(hand_camera=dict(width=args.height, height=args.height)),
        human_render_camera_configs=dict(overview=dict(shader_pack=args.shader)),
    )
    env.reset(seed=args.seed, options={"reconfigure": True})
    u = env.unwrapped
    assert "delta" in u.control_mode, u.control_mode
    for _ in range(10):  # let products settle; zero delta action = hold
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    for name, actor in u.scene.actors.items():
        if "ceiling" in name.lower():
            set_visible(actor, False)

    cam = u._human_render_cameras["overview"]

    def shoot(pose):
        cam.camera.set_local_pose(pose)
        u.scene.update_render(update_sensors=False, update_human_render_cameras=True)
        cam.capture()
        rgb = cam.get_obs(rgb=True, depth=False, position=False, segmentation=False)["rgb"]
        return rgb[0].cpu().numpy().astype(np.uint8)

    def wrist():
        o = u.get_obs()
        return o["sensor_data"]["hand_camera"]["rgb"][0].cpu().numpy().astype(np.uint8)

    xs, ys = u.scene_builder.x_size[0], u.scene_builder.y_size[0]
    room_c = np.array([xs / 2, ys / 2, 0.0])
    shelf = u.actors["fixtures"]["shelves"][u.active_shelves[0][0]].pose.sp
    base = u.agent.robot.pose.sp
    d = shelf.to_transformation_matrix()[:3, 1]
    side = np.cross([0, 0, 1], d)
    work_c = 0.5 * (shelf.p + base.p)
    work_c[2] = 0.75
    print(f"room {xs:.1f}x{ys:.1f} m | shelf at {np.round(shelf.p, 2)} | panda base at {np.round(base.p, 3)}")
    print("target product:", u.target_product_names[0], "|", u.language_instructions[0])

    frames, stills = [], {}
    n1, n2, n3 = 8 * args.fps, 4 * args.fps, 5 * args.fps
    R, Z = 0.55 * max(xs, ys), 6.0

    # 1. dollhouse orbit around the room, ending behind-right of the robot
    a_end = np.arctan2(*(base.p - room_c)[[1, 0]]) - 0.6
    a_start = a_end - 2 * np.pi
    for i in range(n1):
        a = a_start + (a_end - a_start) * i / (n1 - 1)
        eye = room_c + np.array([R * np.cos(a), R * np.sin(a), Z])
        frames.append(shoot(look_at(eye, room_c)))
    stills["1_orbit"] = frames[n1 // 3]
    eye_a, tgt_a = eye, room_c

    # 2. push-in to an over-the-shoulder view of the robot and shelf
    eye_b = base.p - 1.3 * d + 0.9 * side
    eye_b[2] = 1.6
    for i in range(n2):
        t = ease(i / (n2 - 1))
        frames.append(shoot(look_at((1 - t) * eye_a + t * eye_b, (1 - t) * tgt_a + t * work_c)))
    stills["2_closeup"] = frames[-1]

    # 3. slow arc around the robot | wrist camera on the right
    H, W = args.height, args.width
    wr = np.asarray(Image.fromarray(wrist()).resize((H, H)))
    lw = W - H
    for i in range(n3):
        t = ease(i / (n3 - 1))
        eye = base.p - 1.3 * d + (0.9 - 1.8 * t) * side
        eye[2] = 1.6
        tp = shoot(look_at(eye, work_c))
        frames.append(np.concatenate([tp[:, (W - lw) // 2:(W + lw) // 2], wr], axis=1))
    stills["3_wrist"] = frames[-1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=8)
    print(f"wrote {out} ({len(frames)} frames, {len(frames) / args.fps:.1f}s)")
    if args.stills:
        for k, im in stills.items():
            imageio.imwrite(out.with_name(f"{out.stem}_{k}.png"), im)
    env.close()


if __name__ == "__main__":
    main()
