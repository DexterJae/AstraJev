#!/usr/bin/env python3
"""Preflight for the sim path.

Every environment and planner API that `scripts/run_eval.py --world sim`
touches is exercised here, in order, with a named check per API. The point is
that a mismatch surfaces in seconds with the exact attribute named, instead of
thirty minutes into a 30-episode run.

This matters because the sim path was written by reading the RoboBenchMart
source, not by running it: Vulkan and mplib are Linux-only, so it could not be
executed on the machine where it was written. Run this first.

    python scripts/check_sim_env.py --env-id PickToBasketContNiveaEnv \
        --scene-dir demo_envs/test_unseen_items_pick_to_basket

Exit code 0 means `run_eval.py --world sim` should work. Anything else prints
which check failed and what to look at.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []


def check(name: str, fn, *, fatal: bool = False):
    """Run one check, record the outcome, keep going unless it is fatal."""
    try:
        detail = fn()
        _results.append((PASS, name, "" if detail is None else str(detail)))
        return True
    except Exception as exc:  # noqa: BLE001 - a preflight reports, never raises
        _results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
        if fatal:
            print(f"\nFATAL in {name!r}:\n")
            traceback.print_exc()
            report()
            sys.exit(2)
        return False


def report() -> int:
    print()
    print(f"{'result':<7} {'check':<42} detail")
    print("-" * 100)
    for status, name, detail in _results:
        print(f"{status:<7} {name:<42} {detail[:50]}")
    failed = [r for r in _results if r[0] == FAIL]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print()
        print("Fix these before running the experiment. The most likely causes:")
        print("  * ManiSkill version drift renaming an env attribute")
        print("  * assets not downloaded (RoboCasa / RoboBenchMart_assets)")
        print("  * scene dir does not contain generated scenes")
        return 1
    print(f"all {len(_results)} checks passed -- run_eval.py --world sim should work")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default="PickToBasketContNiveaEnv")
    p.add_argument("--scene-dir", default="demo_envs/test_unseen_items_pick_to_basket")
    p.add_argument("--robot-uids", default="ds_fetch_basket")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rbm", default=str(ROOT / "RoboBenchMart"))
    p.add_argument("--no-plan", action="store_true",
                   help="skip the motion-planner checks (they are the slowest)")
    args = p.parse_args(argv)

    rbm = Path(args.rbm)
    check("benchmark checkout present",
          lambda: rbm.exists() or _raise(f"not found: {rbm}"), fatal=True)
    sys.path.insert(0, str(rbm))

    scene_dir = Path(args.scene_dir)
    if not scene_dir.is_absolute():
        scene_dir = rbm / scene_dir

    # -- imports -----------------------------------------------------------
    check("import sapien", lambda: __import__("sapien").__version__, fatal=True)
    check("import mani_skill", lambda: __import__("mani_skill").__version__, fatal=True)
    check("import mplib", lambda: __import__("mplib").__version__, fatal=True)
    check("import torch", lambda: __import__("torch").__version__, fatal=True)

    def _render_device():
        import sapien
        s = sapien.Scene()
        del s
        return "rendering device found"

    check("sapien can create a render device", _render_device, fatal=True)

    def _register():
        import importlib
        importlib.import_module("dsynth.envs")
        importlib.import_module("dsynth.robots")
        import gymnasium as gym
        if args.env_id not in gym.registry:
            _raise(f"{args.env_id} not registered")
        return f"{args.env_id} registered"

    check("dsynth envs and robots register", _register, fatal=True)

    check("scene dir has scenes",
          lambda: _scene_dir_ok(scene_dir), fatal=True)

    # -- environment -------------------------------------------------------
    import gymnasium as gym

    env = None

    def _make():
        nonlocal env
        env = gym.make(
            args.env_id,
            robot_uids=args.robot_uids,
            config_dir_path=str(scene_dir),
            num_envs=1,
            control_mode="pd_joint_pos",
            obs_mode="rgb",
            render_mode="rgb_array",
            sim_backend="auto",
            enable_shadow=True,
            viewer_camera_configs={"shader_pack": "default"},
            human_render_camera_configs={"shader_pack": "default"},
            parallel_in_single_scene=False,
        )
        return "env constructed"

    check("gym.make", _make, fatal=True)
    check("env.reset", lambda: _reset(env, args.seed), fatal=True)

    u = env.unwrapped
    # Snapshot the base pose before anything moves, so the "base never drives"
    # check at the end has something to compare against.
    base_at_reset = _base_qpos(u)

    check("action space is 13-dim",
          lambda: _expect(env.action_space.shape[-1], 13, "action dim"))
    check("env.language_instructions[0]",
          lambda: u.language_instructions[0])
    check("env.unwrapped.get_obs() has the three cameras",
          lambda: _check_obs(u))
    check("env.elapsed_steps is indexable",
          lambda: int(u.elapsed_steps[0].item()))
    check("env.target_products_df has scene_idx",
          lambda: sorted(u.target_products_df.columns)[:4])
    check("env.actors['products'] non-empty",
          lambda: f"{len(u.actors['products'])} products")
    check("env.directions_to_shelf[0]", lambda: list(u.directions_to_shelf[0]))
    check("env.calc_target_pose().sp.p", lambda: list(u.calc_target_pose().sp.p))
    check("env.target_product_names[0]", lambda: u.target_product_names[0])
    check("agent.build_grasp_pose", lambda: _build_grasp_pose(u))
    check("agent.is_static", lambda: bool(u.agent.is_static(0.2)[0].item()))

    # -- our own adapters --------------------------------------------------
    oracle = None

    def _oracle():
        nonlocal oracle
        from astraxjev.oracle import ManiSkillOracle

        oracle = ManiSkillOracle(env)
        return "constructed"

    check("ManiSkillOracle constructs", _oracle)
    if oracle is not None:
        for name, fn in [
            ("oracle.frames() names the plan's references",
             lambda: sorted(oracle.frames())),
            ("oracle.frames() origins are finite",
             lambda: _frames_finite(oracle)),
            ("oracle.free_ref_names", lambda: list(oracle.free_ref_names())),
            ("oracle.capabilities (env-derived tolerances)",
             lambda: oracle.capabilities().to_dict()),
            ("oracle.obstacles", lambda: f"{len(oracle.obstacles())} obstacles"),
            ("oracle.tcp_position", lambda: list(oracle.tcp_position())),
            ("oracle.nearest_neighbour_distance",
             lambda: round(oracle.nearest_neighbour_distance(), 4)),
            ("oracle.gripper_closed", lambda: oracle.gripper_closed()),
            ("oracle.holding_item", lambda: oracle.holding_item()),
            ("oracle.target_label", lambda: oracle.target_label()),
        ]:
            check(name, fn)

        check("env declares its own success threshold",
              lambda: _place_tolerance(u))
        check("a plan generated for this instruction resolves",
              lambda: _plan_resolves(u, oracle))

    # -- planner and executor ---------------------------------------------
    if args.no_plan:
        _results.append((SKIP, "motion planner", "--no-plan"))
        env.close()
        return report()

    planner = None

    def _planner():
        nonlocal planner
        from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver

        planner = FetchMotionPlanningSapienSolver(
            env, vis=False, print_env_info=False, visualize_target_grasp_pose=False
        )
        return f"elapsed_steps={planner.elapsed_steps}"

    check("FetchMotionPlanningSapienSolver constructs", _planner)

    if planner is not None and oracle is not None:
        from astraxjev.executor import ManiSkillExecutor
        from astraxjev.plan import GoalKind, Subgoal, resolve

        ex = ManiSkillExecutor(env, planner, max_env_steps=500)
        frames = oracle.frames()
        wp = resolve(
            Subgoal("approach", GoalKind.STANDOFF, "target",
                    params={"distance": 0.18}),
            frames,
            tcp_position=oracle.tcp_position(),
            anchor=None,
            finger_length_m=0.03,
        )

        check("planner.check_IK on the standoff pose",
              lambda: planner.check_IK(
                  u.agent.build_grasp_pose(wp.approach, wp.closing, wp.position)
              ))
        check("executor.execute(standoff) moves the arm",
              lambda: _exec(ex, wp))
        check("executor.lift(+0.05)", lambda: _exec_lift(ex, 0.05))
        check("executor.set_gripper(close/open)", lambda: _exec_grip(ex))
        check("executor.idle(5)", lambda: ex.idle(5).ok)
        check("base never drove (no mobile manipulation)",
              lambda: _base_unmoved(u, base_at_reset))
        check("executor.success() readable", lambda: ex.success())

    env.close()
    return report()


# -- helpers -------------------------------------------------------------------
def _raise(msg: str):
    raise AssertionError(msg)


def _expect(got, want, what: str):
    if got != want:
        _raise(f"{what}: got {got}, want {want}")
    return got


def _scene_dir_ok(scene_dir: Path):
    if not scene_dir.exists():
        _raise(
            f"{scene_dir} missing -- run "
            "`bash bash/generate_test_scenes.sh` inside the benchmark checkout"
        )
    entries = [p.name for p in scene_dir.iterdir()]
    if not entries:
        _raise(f"{scene_dir} is empty")
    return f"{len(entries)} entries"


def _reset(env, seed: int):
    env.reset(seed=seed, options={"reconfigure": True})
    return f"reset(seed={seed})"


def _check_obs(u):
    obs = u.get_obs()
    sd = obs["sensor_data"]
    for cam in ("left_base_camera_link", "right_base_camera_link", "fetch_hand"):
        if cam not in sd:
            _raise(f"missing camera {cam}; have {sorted(sd)}")
    shapes = {c: tuple(sd[c]["rgb"].shape) for c in sd}
    if "agent" not in obs or "qpos" not in obs["agent"]:
        _raise("obs['agent']['qpos'] missing")
    return shapes


def _frames_finite(oracle):
    import numpy as np

    frames = oracle.frames()
    if not frames:
        _raise("frames() returned nothing; a plan could not be grounded")
    for name, f in frames.items():
        if not np.all(np.isfinite(f.origin)):
            _raise(f"{name}.origin is not finite: {f.origin}")
        if abs(float(np.linalg.norm(f.normal)) - 1.0) > 1e-3:
            _raise(f"{name}.normal is not a unit vector: {f.normal}")
    return f"{len(frames)} frames: {sorted(frames)}"


def _place_tolerance(u):
    """The task's own success threshold replaces a hand-chosen number."""
    thresh = getattr(u, "TARGET_POS_THRESH", None)
    if thresh is None:
        _raise(
            "env has no TARGET_POS_THRESH; plan.calibrate will fall back to "
            "its default place tolerance instead of the task's own"
        )
    return f"TARGET_POS_THRESH={float(thresh)}"


def _plan_resolves(u, oracle):
    """Every reference a generated plan names must be groundable here.

    The planner is asked for a decomposition from the instruction; if it names
    something perception cannot supply, episodes fail on the first tick. Better
    to find that out now.
    """
    from astraxjev.backends.astra_mock import synthesise_plan
    from astraxjev.plan import calibrate

    instruction = u.language_instructions[0]
    plan = calibrate(synthesise_plan(instruction), oracle.capabilities())
    available = set(oracle.frames()) | set(oracle.free_ref_names())
    missing = [r for r in plan.required_refs if r not in available]
    if missing:
        _raise(
            f"plan for {instruction!r} needs refs {missing} which this env "
            f"cannot ground (has {sorted(available)}); add them to "
            "ManiSkillOracle.frames()"
        )
    return f"{len(plan)} subgoals, refs {list(plan.required_refs)} all groundable"


def _build_grasp_pose(u):
    import numpy as np

    pose = u.agent.build_grasp_pose(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0])
    )
    return type(pose).__name__


def _exec(ex, wp):
    res = ex.execute(wp)
    if not res.ok:
        _raise(f"{res.reason} (this is the check most likely to need tuning: "
               "try a larger geometry.standoff_m)")
    return f"ok in {res.env_steps} env steps"


def _exec_lift(ex, d):
    res = ex.lift(d)
    if not res.ok:
        _raise(res.reason)
    return f"ok in {res.env_steps} env steps"


def _exec_grip(ex):
    for g in (-1.0, 1.0):
        res = ex.set_gripper(g)
        if not res.ok:
            _raise(f"gripper {g}: {res.reason}")
    return "ok"


def _base_qpos(u):
    """Fetch's first three qpos entries are its planar base joints."""
    import numpy as np

    return np.asarray(u.agent.robot.get_qpos().cpu().numpy()).reshape(-1)[:3].copy()


def _base_unmoved(u, baseline):
    """The whole task framing depends on the base never driving.

    RoboBenchMart's `follow_path` writes zeros into the last two action dims,
    so this should hold by construction -- which is exactly why it is worth
    asserting once against the real simulator.
    """
    import numpy as np

    now = _base_qpos(u)
    drift = float(np.linalg.norm(now - baseline))
    if drift > 1e-3:
        _raise(f"base moved by {drift:.4f} (qpos[:3] {baseline} -> {now})")
    return f"base qpos unchanged (drift {drift:.2e})"


if __name__ == "__main__":
    sys.exit(main())
