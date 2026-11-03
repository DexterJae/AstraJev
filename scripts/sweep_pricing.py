#!/usr/bin/env python3
"""Sensitivity analysis for the cost claim.

A cost reduction reported at one price sheet and one pair of hand-picked
pipeline parameters is not a result; it is a choice. This script sweeps the
three things the claim actually depends on and reports where it survives and
where it does not.

  1. Jev's unit price. The weakest assumption in config/pricing.yaml. Reported
     as a break-even: the Jev price at which the hybrid stops being cheaper.
     Needs no re-running -- the *work* done is fixed by the recorded run, so
     the ledger is simply re-priced.

  2. Jev's parameter count, for the vendor-independent GFLOPs figure. The
     hybrid replaces cheap Astra-Local ticks with Jev ticks, so if Jev is much
     larger than Astra-Local the hybrid can be cheaper in dollars and dearer
     in FLOPs. Reported as a break-even too.

  3. The pipeline parameters: the baseline's re-localisation cadence
     (`global_every_n_ticks`) and the hybrid's coarse step scale
     (`coarse_step_scale`). These set how expensive each pipeline is, so a
     single value of either chooses the answer. Requires re-running, so it is
     opt-in via --grid.

Usage
-----
    # re-price a finished pair of runs (fast, no simulator)
    python scripts/sweep_pricing.py results/astra_fake_mock.json \
                                    results/astra_jev_fake_mock.json

    # also re-run both pipelines over the parameter grid
    python scripts/sweep_pricing.py results/astra_fake_mock.json \
                                    results/astra_jev_fake_mock.json --grid -n 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def usd_total(result: dict, prices: dict[str, dict]) -> float:
    """Re-price a recorded run.

    Exact, because `by_backend` records the token counts and call counts that
    actually occurred; only the unit prices change.
    """
    total = 0.0
    for backend, agg in result["by_backend"].items():
        p = prices[backend]
        total += (
            agg["tokens_in"] / 1e6 * p["usd_per_mtok_in"]
            + agg["tokens_out"] / 1e6 * p["usd_per_mtok_out"]
            + agg["calls"] * p.get("usd_per_call", 0.0)
        )
    return total


def gflops_total(result: dict, prices: dict[str, dict]) -> float:
    total = 0.0
    for backend, agg in result["by_backend"].items():
        p = prices[backend]
        total += 2.0 * p["params_b"] * (agg["tokens_in"] + agg["tokens_out"])
    return total


def per_success(total: float, result: dict) -> float:
    n = result["successes"]
    return total / n if n else float("inf")


def bisect_breakeven(f, lo: float, hi: float, iters: int = 60):
    """Find x in [lo, hi] where f changes sign, or None if it does not."""
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if (flo > 0) == (fhi > 0):
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sweep_price(a: dict, b: dict, key: str, label: str, unit: str) -> None:
    base_prices = a["prices"]
    nominal = base_prices["jev"][key]

    def with_jev(x: float) -> dict:
        p = {k: dict(v) for k, v in base_prices.items()}
        p["jev"][key] = x
        return p

    metric = usd_total if key.startswith("usd") else gflops_total

    print()
    print(f"sweep: Jev {label}  (nominal {nominal:g} {unit})")
    print(f"{'value':>14} {'baseline/success':>18} {'hybrid/success':>18} {'delta':>10}")
    print("-" * 64)
    multipliers = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    for m in multipliers:
        x = nominal * m if nominal else m
        p = with_jev(x)
        ca, cb = per_success(metric(a, p), a), per_success(metric(b, p), b)
        d = (cb - ca) / ca * 100.0 if math.isfinite(ca) and ca else float("nan")
        print(f"{x:14.6g} {ca:18.4e} {cb:18.4e} {d:+9.1f}%")

    def gap(x: float) -> float:
        p = with_jev(x)
        return per_success(metric(b, p), b) - per_success(metric(a, p), a)

    hi = (nominal if nominal else 1.0) * 1000.0
    be = bisect_breakeven(gap, 0.0, hi)
    if be is None:
        verdict = (
            "hybrid cheaper across the whole swept range"
            if gap(0.0) < 0
            else "hybrid NEVER cheaper in the swept range"
        )
        print(f"  break-even: none -- {verdict}")
    else:
        ratio = be / nominal if nominal else float("inf")
        print(
            f"  break-even: Jev {label} = {be:.6g} {unit}"
            f" ({ratio:.1f}x the assumed value)."
        )
        print(
            f"  The claim holds as long as Jev's {label} is below that."
        )


def run_grid(args) -> None:
    import run_eval

    print()
    print("grid: re-running both pipelines over their cost knobs")
    print("(baseline cadence = ticks between Astra-Global re-localisations;")
    print(" cadence 0 means localise once per episode, the cheapest baseline)")
    print()

    cadences = [1, 2, 4, 8, 0]
    base = {}
    for c in cadences:
        r = run_eval.main(
            [
                "--pipeline", "astra", "--world", args.world,
                "-n", str(args.num_traj), "--quiet",
                "--global-every-n-ticks", str(c),
                "-o", str(ROOT / "results" / f"sweep_astra_cadence{c}.json"),
            ]
        )
        base[c] = r
        print(
            f"  astra cadence={c or 'once':>4}  success={r['success_rate']:.3f} "
            f"usd/success={r['usd_per_success']:.4e} "
            f"global_calls/ep={r['by_backend'].get('astra_global', {}).get('calls', 0) / max(1, r['episodes']):.2f}"
        )

    scales = [2.0, 4.0, 8.0]
    hyb = {}
    cfg_path = ROOT / "config" / "default.yaml"
    import yaml

    original = cfg_path.read_text()
    try:
        for sc in scales:
            cfg = yaml.safe_load(original)
            cfg["geometry"]["coarse_step_scale"] = sc
            tmp = ROOT / "results" / f"_cfg_scale{sc}.yaml"
            tmp.write_text(yaml.safe_dump(cfg))
            r = run_eval.main(
                [
                    "--pipeline", "astra_jev", "--world", args.world,
                    "-n", str(args.num_traj), "--quiet",
                    "--config", str(tmp),
                    "-o", str(ROOT / "results" / f"sweep_astra_jev_scale{sc}.json"),
                ]
            )
            hyb[sc] = r
            print(
                f"  astra_jev step_scale={sc:<4}  success={r['success_rate']:.3f} "
                f"usd/success={r['usd_per_success']:.4e} "
                f"jev_calls/ep={r['by_backend'].get('jev', {}).get('calls', 0) / max(1, r['episodes']):.2f}"
            )
    finally:
        cfg_path.write_text(original)

    cheapest_base = min(base.values(), key=lambda r: r["usd_per_success"])
    dearest_hyb = max(hyb.values(), key=lambda r: r["usd_per_success"])
    best_hyb = min(hyb.values(), key=lambda r: r["usd_per_success"])

    print()
    print("worst case for the claim: cheapest baseline vs dearest hybrid")
    print(
        f"  baseline  usd/success={cheapest_base['usd_per_success']:.4e}"
        f"  (success={cheapest_base['success_rate']:.3f})"
    )
    print(
        f"  hybrid    usd/success={dearest_hyb['usd_per_success']:.4e}"
        f"  (success={dearest_hyb['success_rate']:.3f})"
    )
    if dearest_hyb["usd_per_success"] < cheapest_base["usd_per_success"]:
        d = (
            (dearest_hyb["usd_per_success"] - cheapest_base["usd_per_success"])
            / cheapest_base["usd_per_success"]
            * 100
        )
        print(f"  claim survives the worst case: {d:+.1f}%")
    else:
        print(
            "  claim does NOT survive the worst case. Report the grid, not a"
            " single pair of parameters."
        )

    breakeven_relocalisation_rate(base, best_hyb)


def breakeven_relocalisation_rate(base: dict, hybrid: dict) -> None:
    """The number this experiment actually turns on.

    The baseline's cost is almost entirely set by how many Astra-Global calls
    it makes per episode, and the hybrid's whole mechanism is to make that
    number 1 by answering "do I need to re-localise?" with a cheap model
    instead of an expensive one. So the honest statement of the result is not a
    percentage, it is a threshold: *above how many re-localisations per episode
    does adding Jev pay for itself?*

    Below the threshold the hybrid's Jev and detector overhead exceeds what it
    saves, and plain Astra is cheaper. That is a real limitation and belongs in
    any write-up next to the headline figure.
    """
    pts = sorted(
        (
            r["by_backend"].get("astra_global", {}).get("calls", 0)
            / max(1, r["episodes"]),
            r["usd_per_success"],
        )
        for r in base.values()
    )
    target = hybrid["usd_per_success"]
    hyb_rate = (
        hybrid["by_backend"].get("astra_global", {}).get("calls", 0)
        / max(1, hybrid["episodes"])
    )

    print()
    print("break-even re-localisation rate")
    print("-" * 72)
    print(
        f"  hybrid: {hyb_rate:.2f} Astra-Global calls/episode at"
        f" {target:.4e} USD/success"
    )

    crossing = None
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        if (c0 - target) * (c1 - target) <= 0 and c1 != c0:
            crossing = r0 + (target - c0) * (r1 - r0) / (c1 - c0)
            break

    if crossing is None:
        if pts[0][1] > target:
            print(
                "  the baseline is dearer than the hybrid at every cadence"
                " swept: the claim holds across the grid."
            )
        else:
            print(
                "  the baseline is cheaper than the hybrid at every cadence"
                " swept: on this task Jev does not pay for itself."
            )
        return

    print(
        f"  break-even: Astra alone becomes the dearer option above"
        f" ~{crossing:.2f} Astra-Global calls per episode."
    )
    print(
        "  Below that rate, plain Astra is cheaper -- the Jev and detector"
        " ticks cost more than the Astra-Global calls they avoid."
    )
    print(
        "  So the quantity to measure on the real benchmark is how often"
        " Astra-Global must re-localise under real occlusion and clutter."
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline")
    p.add_argument("hybrid")
    p.add_argument("--grid", action="store_true", help="also re-run the parameter grid")
    p.add_argument("--world", default="fake", choices=["fake", "sim"])
    p.add_argument("-n", "--num-traj", type=int, default=20)
    args = p.parse_args(argv)

    a, b = load(args.baseline), load(args.hybrid)
    print(f"baseline: {a['pipeline']}   hybrid: {b['pipeline']}"
          f"   ({a['episodes']} episodes, {a.get('world')}/{a.get('backend')})")
    print(f"nominal USD/success: {a['usd_per_success']:.4e} -> {b['usd_per_success']:.4e}")

    sweep_price(a, b, "usd_per_mtok_in", "input price", "USD/Mtok")
    sweep_price(a, b, "params_b", "parameter count", "B params")

    if args.grid:
        run_grid(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
