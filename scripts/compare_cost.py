#!/usr/bin/env python3
"""Compare two results JSONs and print the table a paper would quote.

    python scripts/compare_cost.py results/astra_fake_mock.json \
                                   results/astra_jev_fake_mock.json

Leads with cost **per successful episode**, not per episode. A pipeline can
always be made cheaper by failing sooner, and per-episode cost hides that;
per-success cost does not. If the two pipelines' success rates differ, the
report says so explicitly rather than letting the cost number stand alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pct(new: float, old: float) -> str:
    if not math.isfinite(old) or old == 0:
        return "n/a"
    if not math.isfinite(new):
        return "worse (never succeeds)"
    d = (new - old) / old * 100.0
    return f"{d:+.1f}%"


def fmt(x: float, unit: str) -> str:
    if not math.isfinite(x):
        return "inf"
    if unit == "usd":
        return f"${x:.4e}"
    if unit == "frac":
        return f"{x:.3f}"
    return f"{x:,.0f}"


def row(label: str, a: float, b: float, unit: str, lower_is_better=True) -> str:
    return f"{label:<26} {fmt(a, unit):>14} {fmt(b, unit):>14} {pct(b, a):>14}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", help="results JSON for the Astra-only pipeline")
    p.add_argument("hybrid", help="results JSON for the Astra+Jev pipeline")
    p.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = p.parse_args(argv)

    a, b = load(args.baseline), load(args.hybrid)

    print()
    print(f"world/backend   {a.get('world')}/{a.get('backend')}"
          f"   vs   {b.get('world')}/{b.get('backend')}")
    print(f"episodes        {a['episodes']} vs {b['episodes']}")
    print()
    print(f"{'metric':<26} {a['pipeline']:>14} {b['pipeline']:>14} {'delta':>14}")
    print("-" * 72)
    print(row("success rate", a["success_rate"], b["success_rate"], "frac"))
    print(row("USD / episode", a["usd_per_episode"], b["usd_per_episode"], "usd"))
    print(row("USD / success", a["usd_per_success"], b["usd_per_success"], "usd"))
    print(row("GFLOP / episode", a["gflops_per_episode"], b["gflops_per_episode"], "num"))
    print(row("GFLOP / success", a["gflops_per_success"], b["gflops_per_success"], "num"))
    print(row("env steps / episode", a["mean_env_steps"], b["mean_env_steps"], "num"))
    print()

    print("model calls per episode")
    print(f"{'backend':<16} {a['pipeline']:>12} {b['pipeline']:>12}")
    print("-" * 44)
    backends = sorted(set(a["by_backend"]) | set(b["by_backend"]))
    for name in backends:
        ca = a["by_backend"].get(name, {}).get("calls", 0) / max(1, a["episodes"])
        cb = b["by_backend"].get(name, {}).get("calls", 0) / max(1, b["episodes"])
        print(f"{name:<16} {ca:12.2f} {cb:12.2f}")
    print()

    print("USD per episode, by backend")
    print(f"{'backend':<16} {a['pipeline']:>14} {b['pipeline']:>14}")
    print("-" * 48)
    for name in backends:
        ua = a["by_backend"].get(name, {}).get("usd", 0.0) / max(1, a["episodes"])
        ub = b["by_backend"].get(name, {}).get("usd", 0.0) / max(1, b["episodes"])
        print(f"{name:<16} {ua:14.4e} {ub:14.4e}")
    print()

    # The honest caveats, printed every time so they cannot be dropped by
    # accident when the table is copied into a paper.
    sr_gap = b["success_rate"] - a["success_rate"]
    print("read this before quoting the numbers above")
    print("-" * 72)
    if abs(sr_gap) > 0.02:
        worse = "lower" if sr_gap < 0 else "higher"
        print(
            f"* Success rates differ by {sr_gap:+.1%} ({b['pipeline']} is {worse})."
            " Compare per-success cost, not per-episode cost."
        )
    else:
        print(f"* Success rates match to within {abs(sr_gap):.1%}.")
    if a.get("world") == "fake":
        print(
            "* world=fake: the success rates are properties of tests/fake_world.py,"
            " not benchmark results. Re-run with --world sim for those."
        )
    print(
        "* Dollar figures depend on config/pricing.yaml, whose Jev entry is an"
        " assumption. Run scripts/sweep_pricing.py for the break-even."
    )
    print(
        "* GFLOP figures are vendor independent but depend on the assumed"
        " parameter counts, which are also in config/pricing.yaml."
    )

    if args.markdown:
        print()
        print(f"| metric | {a['pipeline']} | {b['pipeline']} | delta |")
        print("|---|---|---|---|")
        print(f"| success rate | {a['success_rate']:.3f} | {b['success_rate']:.3f} | "
              f"{pct(b['success_rate'], a['success_rate'])} |")
        print(f"| USD / episode | {a['usd_per_episode']:.3e} | {b['usd_per_episode']:.3e} | "
              f"{pct(b['usd_per_episode'], a['usd_per_episode'])} |")
        print(f"| USD / success | {a['usd_per_success']:.3e} | {b['usd_per_success']:.3e} | "
              f"{pct(b['usd_per_success'], a['usd_per_success'])} |")
        print(f"| GFLOP / episode | {a['gflops_per_episode']:,.0f} | "
              f"{b['gflops_per_episode']:,.0f} | "
              f"{pct(b['gflops_per_episode'], a['gflops_per_episode'])} |")
        for name in backends:
            ca = a["by_backend"].get(name, {}).get("calls", 0) / max(1, a["episodes"])
            cb = b["by_backend"].get(name, {}).get("calls", 0) / max(1, b["episodes"])
            print(f"| {name} calls / episode | {ca:.2f} | {cb:.2f} | {pct(cb, ca)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
