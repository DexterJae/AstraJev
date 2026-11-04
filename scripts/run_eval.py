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
                   noise in config/default.yaml `mock:`. Token counts, and
                   therefore costs, are the real ones.
  --backend api    the real hosted jev-1.14, plus a vision LLM in the Astra
                   slot. Any OpenAI-compatible endpoint works via
                   --vlm-base-url. Bills your account.

Examples
--------
    python scripts/run_eval.py --pipeline astra      --world fake -n 30
    python scripts/run_eval.py --pipeline astra_jev  --world fake -n 30
    OPENROUTER_API_KEY=... python scripts/run_eval.py --pipeline astra_jev \
        --world fake -n 5 --backend api --vlm-model gpt-6-astra
"""

from __future__ import annotations

import argparse
import os
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
from astraxjev.pipelines.astra_only import PlanCache
from astraxjev.senses import ProprioSenses


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", choices=["astra", "astra_jev"], required=True)
    p.add_argument("--world", choices=["fake", "sim"], default="fake")
    p.add_argument("--backend", choices=["mock", "api"], default="mock",
                   help="mock: simulated from privileged state, real token "
                        "counts. api: the real hosted Jev plus a vision LLM in "
                        "the Astra slot.")
    p.add_argument("--jev-provider", choices=["openrouter", "typesafe"],
                   default="openrouter")
    p.add_argument("--jev-model", default=None,
                   help="override the provider's default Jev model id")
    p.add_argument("--vlm-model", default="gpt-6-astra",
                   help="chat-completions model filling the Astra slot; use a "
                        "matching --pricing file if it is not gpt-6-astra")
    p.add_argument("--vlm-base-url", default="https://api.openai.com/v1")
    p.add_argument("--vlm-key-env", default="OPENAI_API_KEY")
    p.add_argument("--local-policy-host", default="localhost")
    p.add_argument("--local-policy-port", type=int, default=0,
                   help="port of a RoboBenchMart policy server (octo/pi0) to "
                        "use as Astra-Local; 0 uses the mock")
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--pricing", default=str(ROOT / "config" / "pricing.yaml"))
    p.add_argument("-n", "--num-traj", type=int, default=None)
    p.add_argument("--start-seed", type=int, default=None)
    p.add_argument("-o", "--out", default=None)
    # Overrides that the sweep script drives.
    p.add_argument("--global-every-n-ticks", type=int, default=None)
    p.add_argument("--env-id", default=None)
    p.add_argument("--scene-dir", default=None)
    p.add_argument("--save-video", action="store_true",
                   help="sim only: record one mp4 per episode")
    p.add_argument("--video-dir", default=None)
    p.add_argument("-b", "--sim-backend", default="auto",
                   choices=["auto", "cpu", "gpu"])
    p.add_argument("--fake-task", default="pick_to_basket",
                   help="which task shape the toy world should model")
    p.add_argument("--no-plan-cache", action="store_true",
                   help="pay an Astra-Global planning call every episode")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def build_backends(args, cfg, oracle, rng):
    if args.backend == "api":
        # Real models: gpt-6-astra (or --vlm-model) for Astra, hosted Jev.
        from astraxjev.backends.astra_api import GptAstraBackend
        from astraxjev.backends.jev_api import JevApiBackend

        astra = GptAstraBackend(
            model=args.vlm_model,
            base_url=args.vlm_base_url,
            api_key_env=args.vlm_key_env,
        )
        jev = (
            JevApiBackend(provider=args.jev_provider, model=args.jev_model)
            if args.pipeline == "astra_jev"
            else None
        )
        return astra, jev

    from astraxjev.backends import MockAstraBackend, MockJevBackend

    m = cfg["mock"]
    astra = MockAstraBackend(
        oracle,
        localize_sigma_m=m["astra_localize_sigma_m"],
        act_sigma_m=m["astra_act_sigma_m"],
        rng=rng,
    )
    jev = MockJevBackend(
        oracle,
        choice_accuracy=m["jev_choice_accuracy"],
        noul_noise=m["jev_noul_noise"],
        rng=rng,
    )
    return astra, jev


def build_pipeline(args, cfg, geom, astra, jev, plan_cache, capabilities):
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
            max_ticks=a["max_ticks"],
            plan_cache=plan_cache,
            capabilities=capabilities,
        )

    h = cfg["astra_jev"]
    return AstraJevPipeline(
        astra,
        jev,
        geom,
        jev_precision_threshold_mm=h["jev_precision_threshold_mm"],
        handoff_prob=h["handoff_prob"],
        relocalize_prob=h["relocalize_prob"],
        max_align_resamples=h["max_align_resamples"],
        align_accept_score=h["align_accept_score"],
        obstacle_risk_escalate=h["obstacle_risk_escalate"],
        obstacle_attention_m=h["obstacle_attention_m"],
        max_relocalizes_per_subgoal=h["max_relocalizes_per_subgoal"],
        max_ticks=h["max_ticks"],
        plan_cache=plan_cache,
        capabilities=capabilities,
    )


# ---------------------------------------------------------------- fake world
def run_fake(args, cfg, prices):
    from fake_world import TASKS, FakeExecutor, FakeOracle, FakeWorld

    task = cfg["task"]
    n = args.num_traj or task["num_traj"]
    seed0 = args.start_seed if args.start_seed is not None else task["start_seed"]
    geom = Geometry(**cfg["geometry"])
    ledger = CostLedger(prices)
    outcomes = []
    # One cache for the whole run: the decomposition for a task does not
    # change between episodes, only the grounding does. Both pipelines use the
    # same class, so neither gains from it.
    plan_cache = PlanCache(enabled=not args.no_plan_cache)

    if args.fake_task not in TASKS:
        raise SystemExit(
            f"unknown --fake-task {args.fake_task!r}; choose from {sorted(TASKS)}"
        )
    instruction = TASKS[args.fake_task]

    for i in range(n):
        seed = seed0 + i
        rng = np.random.default_rng(cfg["mock"]["seed"] + seed)
        world = FakeWorld(rng=rng, task=args.fake_task)
        # Vary the scene the way the benchmark's test scenes do.
        jitter = rng.normal(0, 0.03, 3)
        world.target_position = world.target_position + jitter
        world.item_position = world.target_position.copy()
        world.hinge = world.hinge + rng.normal(0, 0.02, 3)
        world.neighbours = world._neighbours() + rng.normal(
            0, 0.01, world._neighbours().shape
        )

        oracle = FakeOracle(world)
        astra, jev = build_backends(args, cfg, oracle, rng)
        pipeline = build_pipeline(
            args, cfg, geom, astra, jev, plan_cache, oracle.capabilities()
        )
        executor = FakeExecutor(world, max_env_steps=task["max_horizon"])

        ctx = EpisodeContext(
            observe=lambda w=world, p=instruction: w.observation(p),
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
                f"steps={o.env_steps} "
                f"plan={o.subgoals_completed}/{o.subgoals_total}({o.subgoal_reached}) "
                f"usd={o.cost_usd:.3e} astra={o.astra_calls} jev={o.jev_calls}"
                + (f"  <- {o.failure_reason}" if o.failure_reason else "")
            )
    return ledger, outcomes, plan_cache


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
    sys.path.insert(0, str(ROOT / "sim"))
    # dsynth loads assets/ and textures by path relative to the benchmark
    # checkout, so run from there. Pin our own paths first.
    if args.out:
        args.out = str(Path(args.out).resolve())
    if args.video_dir:
        args.video_dir = str(Path(args.video_dir).resolve())
    os.chdir(rbm)

    # Importing these modules is what registers the environments and the
    # robot agents with ManiSkill's gym registry.
    import importlib

    importlib.import_module("dsynth.envs")
    importlib.import_module("dsynth.robots")

    from astraxjev.executor import ManiSkillExecutor, PandaExecutor
    from astraxjev.oracle import ManiSkillOracle

    task = cfg["task"]
    is_panda = "panda" in task["robot_uids"]
    if is_panda:
        # PickToBasketPandaPedestal-v0: stationary Panda + wrist camera on a
        # pedestal in front of the RoboBenchMart mini shelf.
        importlib.import_module("panda_pedestal_env")
        from panda_planner import PandaPedestalPlanner
        cams = ("scene_left", "scene_right", "hand_camera")
    else:
        from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver
        cams = ("left_base_camera_link", "right_base_camera_link", "fetch_hand")
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
    plan_cache = PlanCache(enabled=not args.no_plan_cache)

    def observe(env=env, prompt_box=None):
        from astraxjev.types import Observation

        obs = env.unwrapped.get_obs()
        sd = obs["sensor_data"]
        return Observation(
            image=sd[cams[0]]["rgb"][0].cpu().numpy(),
            extra_image=sd[cams[1]]["rgb"][0].cpu().numpy(),
            wrist_image=sd[cams[2]]["rgb"][0].cpu().numpy(),
            state=obs["agent"]["qpos"][0].cpu().numpy(),
            prompt=prompt_box[0],
            time_step=int(env.unwrapped.elapsed_steps[0].item()),
        )

    for i in range(n):
        seed = seed0 + i
        rng = np.random.default_rng(cfg["mock"]["seed"] + seed)
        env.reset(seed=seed, options={"reconfigure": True})
        prompt_box = [env.unwrapped.language_instructions[0]]

        if is_panda:
            planner = PandaPedestalPlanner(env)
        else:
            planner = FetchMotionPlanningSapienSolver(
                env, vis=False, print_env_info=False, visualize_target_grasp_pose=False
            )
        oracle = ManiSkillOracle(env)
        astra, jev = build_backends(args, cfg, oracle, rng)
        pipeline = build_pipeline(
            args, cfg, geom, astra, jev, plan_cache, oracle.capabilities()
        )
        executor = (PandaExecutor if is_panda else ManiSkillExecutor)(
            env, planner, max_env_steps=task["max_horizon"]
        )

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
                f"steps={o.env_steps} "
                f"plan={o.subgoals_completed}/{o.subgoals_total}({o.subgoal_reached}) "
                f"usd={o.cost_usd:.3e} astra={o.astra_calls} jev={o.jev_calls}"
                + (f"  <- {o.failure_reason}" if o.failure_reason else "")
            )
        if args.save_video:
            env.flush_video(str(seed))

    env.close()
    return ledger, outcomes, plan_cache


def main(argv=None):
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    prices = PriceTable.from_yaml(args.pricing)

    runner = run_fake if args.world == "fake" else run_sim
    ledger, outcomes, plan_cache = runner(args, cfg, prices)

    result = summarise(args.pipeline, ledger, outcomes)
    result["world"] = args.world
    result["backend"] = args.backend
    result["task"] = args.fake_task if args.world == "fake" else (
        args.env_id or cfg["task"]["env_id"]
    )
    result.update(plan_cache.stats())
    result["config"] = cfg
    if args.global_every_n_ticks is not None:
        result["override_global_every_n_ticks"] = args.global_every_n_ticks

    out = args.out or str(
        ROOT / "results" / f"{args.pipeline}_{args.world}_{args.backend}.json"
    )
    write_result(out, result)

    print()
    print(f"pipeline        {result['pipeline']}  ({args.world}/{args.backend})")
    print(f"task            {result['task']}")
    print(f"episodes        {result['episodes']}")
    print(f"success rate    {result['success_rate']:.3f}")
    print(f"plan progress   {result['mean_plan_progress']:.3f}")
    print(f"USD / episode   {result['usd_per_episode']:.4e}")
    print(f"USD / success   {result['usd_per_success']:.4e}")
    print(f"GFLOP / episode {result['gflops_per_episode']:.1f}")
    prov = result["cost_provenance"]
    print(
        f"cost measured    {prov['fraction_measured']:.1%} of USD came from "
        f"provider-reported usage ({prov['calls_with_provider_cost']}"
        f"/{prov['calls_total']} calls)"
    )
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
