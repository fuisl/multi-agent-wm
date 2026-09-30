"""Summarize eval_multi.py runs: one row per run, paired against a reference run.

    python scripts/summarize_multi.py                      # all runs under $STABLEWM_HOME/multipusht
    python scripts/summarize_multi.py --ref 1a_single

Paired columns (same seed -> same episodes in every run):
    helped  episodes solved by this run but not by the reference
    hurt    episodes solved by the reference but not by this run
non-trivial: success on the episodes whose T was not already at the goal at t=0
T must move: success on the episodes whose T starts outside the success tolerance (0% for one agent with agent_force)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parents[1] / "data"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(os.environ["STABLEWM_HOME"]) / "multipusht", type=Path)
    parser.add_argument("--ref", default="1a_single")
    args = parser.parse_args()

    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(args.root.glob("*/results.json"))}
    runs = {k: r for k, r in runs.items() if isinstance(r, dict) and "summary" in r}  # eval_multi.py runs only
    ref = runs.get(args.ref)
    ref_success = np.array([e["success"] for e in ref["episodes"]]) if ref else None

    print("| run | agents | success | non-trivial | T must move | mean steps | block contact / agent | co-contact | agent-agent contact | helped | hurt |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in runs.items():
        s, cfg = r["summary"], r["config"]
        success = np.array([e["success"] for e in r["episodes"]])
        policies = cfg["policies"] or [cfg["policy"]] * cfg["env"]["n_agents"]
        helped = hurt = "-"
        if ref is not None and len(success) == len(ref_success) and name != args.ref:
            helped, hurt = int((success & ~ref_success).sum()), int((~success & ref_success).sum())
        steps = f"{s['mean_success_step']:.1f}" if s["mean_success_step"] else "-"
        trivial = np.array([e.get("solved_at_start", False) for e in r["episodes"]])
        nontrivial = f"{100 * success[~trivial].mean():.0f}% ({int((~trivial).sum())})" if "solved_at_start" in r["episodes"][0] else "-"
        placed = np.array([e.get("block_placed_at_start", False) for e in r["episodes"]])
        must_move = f"{100 * success[~placed].mean():.0f}% ({int((~placed).sum())})" if "block_placed_at_start" in r["episodes"][0] else "-"
        contact = " / ".join(f"{c:.2f}" for c in s["block_contact_frac"])
        print(f"| {name} | {', '.join(policies)} ({cfg['env']['others']}) | {s['success_rate']:.0f}% | {nontrivial} | {must_move} | {steps} | "
              f"{contact} | {s['co_contact_frac']:.2f} | {s['agent_contact_frac']:.2f} | {helped} | {hurt} |")


if __name__ == "__main__":
    main()
