#!/usr/bin/env python3
"""Run one pipeline over N episodes and write a results JSON.

Two worlds:

  --world sim    the real RoboBenchMart environment. Needs ManiSkill, Vulkan,
                 the RoboCasa + RoboBenchMart assets, and a Linux GPU box.
                 This is the only mode whose success rate is a benchmark
                 result.

  --world fake   the toy world in tests/fake_world.py. No simulator needed.
                 Use it to exercise the pipelines and to produce the cost
                 ledger, which does not depend on the physics -- only on which
                 prompts get built and how often. Its success rate is a
                 property of the toy, not a benchmark number.

Two backend sets:

  --backend mock   Astra and Jev simulated from privileged state with the
                   noise in config/default.yaml `mock:`. Token counts and
                   therefore costs are the real ones.
  --backend http   real Astra and Jev endpoints (see astraxjev/backends/*_http).

Examples
--------
    python scripts/run_eval.py --pipeline astra      --world fake -n 30
    python scripts/run_eval.py --pipeline astra_jev  --world fake -n 30
    python scripts/run_eval.py --pipeline astra_jev  --world sim  -n 30 \
        --backend http --astra-url http://localhost:9001 --jev-url http://localhost:9002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from astraxjev.cost import CostLedger, PriceTable, summarise, write_result
from astraxjev.pipelines import (
    AstraJevPipeline,
    AstraOnlyPipeline,
    EpisodeContext,
    Geometry,
)
from astraxjev.senses import ProprioSenses


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", choices=["astra", "astra_jev"], required=True)
    p.add_argument("--world", choices=["fake", "sim"], default="fake")
    p.add_argument("--backend", choices=["mock", "http"], default="mock")
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--pricing", default=str(ROOT / "config" / "pricing.yaml"))
    p.add_argument("-n", "--num-traj", type=int, default=None)
    p.add_argument("--start-seed", type=int, default=None)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--astra-url", default=None)
    p.add_argument("--jev-url", default=None)
    # Overrides that the sweep script drives.
    p.add_argument("--global-every-n-ticks", type=int, default=None)
    p.add_argument("--env-id", default=None)
    p.add_argument("--scene-dir", default=None)
    p.add_argument("--save-video", action="store_true",
                   help="sim only: record one mp4 per episode")
    p.add_argument("--video-dir", default=None)
    p.add_argument("-b", "--sim-backend", default="auto",
                   choices=["auto", "cpu", "gpu"])
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def build_backends(args, cfg, oracle, rng):
    if args.backend == "http":
        from astraxjev.backends.astra_http import HttpAstraBackend
        from astraxjev.backends.jev_http import HttpJevBackend

        if not args.astra_url:
            raise SystemExit("--backend http requires --astra-url")
        astra = HttpAstraBackend(args.astra_url)
        jev = HttpJevBackend(args.jev_url) if args.jev_url else None
        if args.pipeline == "astra_jev" and jev is None:
            raise SystemExit("--pipeline astra_jev with --backend http requires --jev-url")
        return astra, jev

    from astraxjev.backends import MockAstraBackend, MockJevBackend

    m = cfg["mock"]
    astra = MockAstraBackend(
        oracle,
        global_sigma_m=m["astra_global_sigma_m"],
        local_sigma_m=m["astra_local_sigma_m"],
        rng=rng,
    )
    jev = MockJevBackend(
        oracle,
        choice_accuracy=m["jev_choice_accuracy"],
        noul_noise=m["jev_noul_noise"],
        rng=rng,
    )
    return astra, jev


def build_pipeline(args, cfg, geom, astra, jev):
    if args.pipeline == "astra":
        a = cfg["astra_only"]
        cadence = (
            args.global_every_n_ticks
            if args.global_every_n_ticks is not None
            else a["global_every_n_ticks"]
        )
        return AstraOnlyPipeline(
            astra,
            geom,
            global_every_n_ticks=cadence,
            chunk_len=a["chunk_len"],
            max_ticks=a["max_ticks"],
        )

    h = cfg["astra_jev"]
    return AstraJevPipeline(
        astra,
        jev,
        geom,
        jev_precision_threshold_mm=h["jev_precision_threshold_mm"],
        handoff_prob=h["handoff_prob"],
        max_align_resamples=h["max_align_resamples"],
        align_accept_score=h["align_accept_score"],
        retract_risk_escalate=h["retract_risk_escalate"],
        max_ticks=h["max_ticks"],
    )


# ---------------------------------------------------------------- fake world
def run_fake(args, cfg, prices):
    from fake_world import FakeExecutor, FakeOracle, FakeWorld

    task = cfg["task"]
    n = args.num_traj or task["num_traj"]
    seed0 = args.start_seed if args.start_seed is not None else task["start_seed"]
    geom = Geometry(**cfg["geometry"])
    ledger = CostLedger(prices)
    outcomes = []

    for i in range(n):
        seed = seed0 + i
        rng = np.random.default_rng(cfg["mock"]["seed"] + seed)
        world = FakeWorld(rng=rng)
        # Vary the scene the way the benchmark's test scenes do: item height,
        # lateral offset and shelf depth all shift per seed.
        world.target_position = world.target_position + rng.normal(0, 0.03, 3)
        world.item_position = world.target_position.copy()
        world.neighbours = world.neighbours + rng.normal(0, 0.02, world.neighbours.shape)

        oracle = FakeOracle(world)
        astra, jev = build_backends(args, cfg, oracle, rng)
        pipeline = build_pipeline(args, cfg, geom, astra, jev)
        executor = FakeExecutor(world, max_env_steps=task["max_horizon"])

        prompt = f"move to shelf and pick {oracle.target_label()} to basket"
        ctx = EpisodeContext(
            observe=lambda w=world, p=prompt: w.observation(p),
            senses=ProprioSenses(oracle),
            executor=executor,
            ledger=ledger,
            seed=seed,
            episode=i,
        )
        outcomes.append(pipeline.run_episode(ctx))
        if not args.quiet:
            o = outcomes[-1]
            print(
                f"[{i + 1}/{n}] seed={seed} success={o.success} "
                f"steps={o.env_steps} phase={o.phase_reached.value} "
                f"usd={o.cost_usd:.3e} astra={o.astra_calls} jev={o.jev_calls}"
                + (f"  <- {o.failure_reason}" if o.failure_reason else "")
            )
    return ledger, outcomes


# ----------------------------------------------------------------- sim world
def run_sim(args, cfg, prices):
    """Real RoboBenchMart evaluation.

    Imports are local because ManiSkill/sapien are heavy and Linux-only; the
    fake path must stay importable without them.
    """
    import gymnasium as gym

    rbm = ROOT / "RoboBenchMart"
    if not rbm.exists():
        raise SystemExit(
            f"expected the benchmark checkout at {rbm}; "
            "git clone https://github.com/emb-ai/RoboBenchMart"
        )
    sys.path.insert(0, str(rbm))

    # Importing these modules is what registers the environments and the
    # ds_fetch_basket agent with ManiSkill's gym registry.
    import importlib

    importlib.import_module("dsynth.envs")
    importlib.import_module("dsynth.robots")
    from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver

    from astraxjev.executor import ManiSkillExecutor
    from astraxjev.oracle import ManiSkillOracle

    task = cfg["task"]
    env_id = args.env_id or task["env_id"]
    scene_dir = args.scene_dir or task["scene_dir"]
    n = args.num_traj or task["num_traj"]
    seed0 = args.start_seed if args.start_seed is not None else task["start_seed"]
    geom = Geometry(**cfg["geometry"])

    env = gym.make(
        env_id,
        robot_uids=task["robot_uids"],
        config_dir_path=scene_dir,
        num_envs=1,
        control_mode=task["control_mode"],
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        enable_shadow=True,
        # Matches what eval_policy_client.py passes when no scene json is given.
        viewer_camera_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        parallel_in_single_scene=False,
    )

    if args.save_video:
        from mani_skill.utils.wrappers.record import RecordEpisode

        video_dir = Path(args.video_dir or (ROOT / "results" / "videos" / args.pipeline))
        env = RecordEpisode(
            env,
            output_dir=str(video_dir),
            trajectory_name=args.pipeline,
            save_video=True,
            save_trajectory=False,
            source_type="Policy",
            source_desc=f"{args.pipeline} pipeline",
            video_fps=30,
            record_reward=False,
            save_on_reset=False,
        )
        print(f"recording video to {video_dir}")

    ledger = CostLedger(prices)
    outcomes = []

    def observe(env=env, prompt_box=None):
        from astraxjev.types import Observation

        obs = env.unwrapped.get_obs()
        sd = obs["sensor_data"]
        return Observation(
            image=sd["left_base_camera_link"]["rgb"][0].cpu().numpy(),
            extra_image=sd["right_base_camera_link"]["rgb"][0].cpu().numpy(),
            wrist_image=sd["fetch_hand"]["rgb"][0].cpu().numpy(),
            state=obs["agent"]["qpos"][0].cpu().numpy(),
            prompt=prompt_box[0],
            time_step=int(env.unwrapped.elapsed_steps[0].item()),
        )

    for i in range(n):
        seed = seed0 + i
        rng = np.random.default_rng(cfg["mock"]["seed"] + seed)
        env.reset(seed=seed, options={"reconfigure": True})
        prompt_box = [env.unwrapped.language_instructions[0]]

        planner = FetchMotionPlanningSapienSolver(
            env, vis=False, print_env_info=False, visualize_target_grasp_pose=False
        )
        oracle = ManiSkillOracle(env)
        astra, jev = build_backends(args, cfg, oracle, rng)
        pipeline = build_pipeline(args, cfg, geom, astra, jev)
        executor = ManiSkillExecutor(env, planner, max_env_steps=task["max_horizon"])

        ctx = EpisodeContext(
            observe=lambda pb=prompt_box: observe(env, pb),
            senses=ProprioSenses(oracle),
            executor=executor,
            ledger=ledger,
            seed=seed,
            episode=i,
        )
        outcomes.append(pipeline.run_episode(ctx))
        o = outcomes[-1]
        if not args.quiet:
            print(
                f"[{i + 1}/{n}] seed={seed} success={o.success} "
                f"steps={o.env_steps} phase={o.phase_reached.value} "
                f"usd={o.cost_usd:.3e} astra={o.astra_calls} jev={o.jev_calls}"
                + (f"  <- {o.failure_reason}" if o.failure_reason else "")
            )
        if args.save_video:
            env.flush_video(str(seed))

    env.close()
    return ledger, outcomes


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    prices = PriceTable.from_yaml(args.pricing)

    runner = run_fake if args.world == "fake" else run_sim
    ledger, outcomes = runner(args, cfg, prices)

    result = summarise(args.pipeline, ledger, outcomes)
    result["world"] = args.world
    result["backend"] = args.backend
    result["config"] = cfg
    if args.global_every_n_ticks is not None:
        result["override_global_every_n_ticks"] = args.global_every_n_ticks

    out = args.out or str(
        ROOT / "results" / f"{args.pipeline}_{args.world}_{args.backend}.json"
    )
    write_result(out, result)

    print()
    print(f"pipeline        {result['pipeline']}  ({args.world}/{args.backend})")
    print(f"episodes        {result['episodes']}")
    print(f"success rate    {result['success_rate']:.3f}")
    print(f"USD / episode   {result['usd_per_episode']:.4e}")
    print(f"USD / success   {result['usd_per_success']:.4e}")
    print(f"GFLOP / episode {result['gflops_per_episode']:.1f}")
    print("calls by backend:")
    for b, agg in sorted(result["by_backend"].items()):
        print(
            f"  {b:14s} calls={agg['calls']:6d}  tok_in={agg['tokens_in']:9d}  "
            f"usd={agg['usd']:.4e}"
        )
    print(f"\nwrote {out}")
    return result


if __name__ == "__main__":
    main()
