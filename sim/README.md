# Stationary Panda setup on RoboBenchMart

A Franka Panda with a wrist camera (`panda_wristcam`), mounted on a fixed pedestal
in front of the original RoboBenchMart mini shelf (`fixtures.small_shelf_two_sided`).
The scene, products, textures and success checks are RoboBenchMart's own
(`PickToBasketContEnv`). Nothing in RoboBenchMart is edited.

| file | what |
|---|---|
| `panda_pedestal_env.py` | `PickToBasketPandaPedestal-v0`: pedestal, side table, fixed basket as goal, target sampled from the robot's side of the shelf |
| `conf/panda_mini_shelf.yaml` | scene config: one active mini shelf with 3 stocked boards, in a 10x8 m store |
| `render_overview.py` | video 1: overview of the whole scene (orbit, push-in, wrist camera) |
| `probe_setup.py`, `check_task.py` | geometry probe and task sanity check (target reachable, basket success check) |

Geometry: Panda base 0.55 m above the floor, 0.40 m from the shelf face. Boards sit at
0.21 / 0.66 / 1.11 m (base-frame z −0.34 / +0.11 / +0.56). Items on the robot's face
are at base-frame x 0.40–0.60.

```bash
cp sim/conf/panda_mini_shelf.yaml RoboBenchMart/conf/ds_continuous/
cd RoboBenchMart
python scripts/generate_scene_continuous.py ds_continuous=panda_mini_shelf
python ../sim/check_task.py generated_envs/panda_mini_shelf
python ../sim/render_overview.py --scene-dir generated_envs/panda_mini_shelf --out ../results/videos/01_overview.mp4 --stills
```
Requires `mani_skill==3.0.1` (3.0.0b5 has no RoboCasa scene builder).
