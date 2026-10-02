"""Centralized CEM with the true simulator as its model: the planner ceiling for cooperative Push-T.

One CEM plans both agents' actions jointly. Candidates are rolled out in a separate copy of the
simulator (the real episode is never touched), and the cost is the success test's own quantity at
the end of the horizon: ||[agent 0 xy, T xy] - goal|| + T angle error (20 deg weighted like 20 px).
This asks whether CEM, with a given budget, horizon and sampling noise, finds a joint push at all,
without model error (the iCEM ablations use ground-truth dynamics for the same reason). The search
mirrors the latent planner: z-scored actions (the Push-T expert's scaler), mean 0 / std 1 init,
warm start from the shifted previous plan, the mean executed.

    python scripts/oracle_cem.py                    # the sweep below, on 12 'T must move' episodes
    python scripts/oracle_cem.py --quick            # one config, 2 episodes (smoke test)
-> $STABLEWM_HOME/multipusht/coop/oracle_cem/<config>/ep<e>.json, summary.json
"""

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("STABLEWM_HOME", str(REPO / "data"))

import numpy as np

OUT = Path(os.environ["STABLEWM_HOME"]) / "multipusht/coop/oracle_cem"
ENV = dict(n_agents=2, agent_force=1e5, solver_iterations=50)
SEED, BUDGET, FRAMESKIP = 42, 150, 5
ANGLE_W = 20.0 / np.radians(20)  # 20 deg costs like 20 px, the two success tolerances


def colored_noise(beta, shape, rng):
    """Gaussian noise with power spectrum 1/f^beta along axis 1 (time), unit std per series (iCEM)."""
    n = shape[1]
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if n > 1 else 1.0
    amp = f ** (-beta / 2.0)
    spec = amp[None, :, None] * (rng.standard_normal((shape[0], len(f), shape[2])) + 1j * rng.standard_normal((shape[0], len(f), shape[2])))
    x = np.fft.irfft(spec, n=n, axis=1)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)


class SimModel:
    """Snapshot / restore of the bodies of a PushTN core, for rolling out candidates."""

    def __init__(self, core):
        self.core = core
        self.bodies = [*core.agents, *core.drives, core.block]

    def load(self, real):
        for b, r in zip(self.bodies, [*real.agents, *real.drives, real.block]):
            b.position, b.velocity, b.angle, b.angular_velocity = r.position, r.velocity, r.angle, r.angular_velocity

    def snapshot(self):
        return [(b.position, b.velocity, b.angle, b.angular_velocity) for b in self.bodies]

    def restore(self, snap):
        for b, (p, v, a, w) in zip(self.bodies, snap):
            b.position, b.velocity, b.angle, b.angular_velocity = p, v, a, w


def cost(core, goal):
    s, N = core._get_obs(), core.n_agents
    pos = np.linalg.norm(np.concatenate([goal[:2] - s[:2], goal[2 * N : 2 * N + 2] - s[2 * N : 2 * N + 2]]))
    ang = abs(goal[2 * N + 2] - s[2 * N + 2])
    return pos + ANGLE_W * min(ang, 2 * np.pi - ang)


def plan(model, real, goal, mean, cfg, rng, mu, sd):
    """One CEM solve. mean: (T, 2 agents, 2) warm start in z-space; returns the new mean."""
    T = mean.shape[0]
    model.load(real)
    snap = model.snapshot()
    std = np.ones_like(mean)
    elites = None
    lo, hi = (-1 - mu) / sd, (1 - mu) / sd  # env action bounds in z-space
    for it in range(cfg["iters"]):
        eps = colored_noise(cfg["beta"], (cfg["samples"], T, 4), rng).reshape(cfg["samples"], T, 2, 2) if cfg["beta"] else rng.standard_normal((cfg["samples"], T, 2, 2))
        cand = np.clip(mean[None] + std[None] * eps, lo, hi)
        cand[0] = mean
        if elites is not None:  # iCEM memory: a fraction of the previous elites stays in the population
            cand[1 : 1 + len(elites)] = elites
        costs = np.empty(len(cand))
        for k, z in enumerate(cand):
            model.restore(snap)
            for a in z:
                model.core.simulate((a * sd + mu).astype(np.float32))
            costs[k] = cost(model.core, goal)
        top = np.argsort(costs)[: cfg["elites"]]
        mean, std = cand[top].mean(0), np.maximum(cand[top].std(0), 0.05)
        elites = cand[top[: int(0.3 * cfg["elites"])]] if cfg["keep_elites"] else None
    return mean, float(costs[top].mean())


def run_episode(args):
    cfg, e, ep_data = args
    from world import MultiPushT
    name = cfg_name(cfg)
    out = OUT / name / f"ep{e:02d}.json"
    if out.exists():
        return json.loads(out.read_text())
    mu, sd, init_state, goal_state = ep_data
    rng = np.random.default_rng([SEED, e])
    real = MultiPushT(max_episode_steps=10 * BUDGET, **ENV)
    real.reset(seed=SEED + e, options={"state": init_state, "goal_state": goal_state})
    sim = MultiPushT(max_episode_steps=10 * BUDGET, **ENV)
    sim.reset(seed=SEED + e, options={"state": init_state, "goal_state": goal_state})
    model, goal = SimModel(sim.core), real.core.goal_state
    T = cfg["H"] * FRAMESKIP
    mean = np.zeros((T, 2, 2))
    trace, t0, success_step, plan_costs = [], time.time(), -1, []
    t = 0
    while t < BUDGET:
        mean, pc = plan(model, real.core, goal, mean, cfg, rng, mu, sd)
        plan_costs.append(pc)
        n_exec = cfg["replan"] * FRAMESKIP
        for a in mean[:n_exec]:
            _, _, term, _, _ = real.step({ag: (a[i] * sd + mu).astype(np.float32) for i, ag in enumerate(real.possible_agents)})
            t += 1
            trace.append(round(cost(real.core, goal), 2))
            if term["agent_0"]:
                success_step = t
                break
            if t >= BUDGET:
                break
        if success_step > 0:
            break
        mean = np.concatenate([mean[n_exec:], np.zeros((n_exec, 2, 2))])  # shift warm start
    block, angle, agents = real.core.errors(goal, real.state())
    res = dict(config=name, episode=e, success=success_step > 0, success_step=success_step, steps=t,
               cost_start=trace[0] if trace else None, cost_end=trace[-1] if trace else None,
               block_pos_err=block, block_angle_err=angle, agent_dist=agents.tolist(),
               plan_costs=plan_costs, trace=trace, wall_s=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def cfg_name(c):
    return f"H{c['H']}_n{c['samples']}_b{c['beta']}_r{c['replan']}_i{c['iters']}" + ("_keep" if c["keep_elites"] else "")


def eval_episodes():
    """The 50 official eval episodes (eval_multi.py protocol) -> scaler, starts, goals, must-move mask."""
    from omegaconf import OmegaConf
    from stable_worldmodel.world.world import _extract_init_goal
    from eval import get_dataset
    from eval_multi import sample_eval_starts
    from world import MultiPushT
    cfg = OmegaConf.load(REPO / "config/eval/multipusht.yaml")
    cfg.cache_dir, cfg.seed = None, SEED
    ds = get_dataset(cfg, cfg.eval.dataset_name)
    eps, starts = sample_eval_starts(cfg, ds)
    init, goal, _ = _extract_init_goal(ds, eps, starts, cfg.eval.goal_offset_steps)
    act = ds.get_col_data("action")
    act = act[~np.isnan(act).any(1)]
    mu, sd = act.mean(0), act.std(0)
    must_move = []
    for e in range(len(eps)):
        env = MultiPushT(max_episode_steps=10, **ENV)
        env.reset(seed=SEED + e, options={"state": init["state"][e], "goal_state": goal["goal_state"][e]})
        b, a, _ = env.core.errors(env.core.goal_state, env.state())
        must_move.append(not (b < 20 and a < np.pi / 9))
    return mu, sd, init["state"], goal["goal_state"], np.array(must_move)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--episodes", type=int, default=12, help="first n 'T must move' episodes")
    p.add_argument("--procs", type=int, default=64)
    args = p.parse_args()
    mu, sd, init, goal, must_move = eval_episodes()
    episodes = np.nonzero(must_move)[0][: 2 if args.quick else args.episodes].tolist()
    print(f"{must_move.sum()} must-move episodes; running {episodes}", flush=True)
    base = dict(iters=10, elites=30, keep_elites=False)
    if args.quick:
        grid = [dict(base, H=5, samples=60, beta=0.0, replan=5, iters=3, elites=6)]
    else:
        grid = [dict(base, H=H, samples=n, beta=b, replan=r, elites=n // 10, keep_elites=b > 0)
                for H, n, b, r in itertools.product((5, 10), (300, 1000), (0.0, 2.0), (1, 5))]
    jobs = [(c, e, (mu, sd, init[e], goal[e])) for c in grid for e in episodes]
    jobs.sort(key=lambda j: -j[0]["samples"] * j[0]["H"] / j[0]["replan"])  # longest first
    with Pool(min(args.procs, len(jobs))) as pool:
        results = list(pool.imap_unordered(run_episode, jobs))
    summary = {}
    for c in grid:
        rs = [r for r in results if r["config"] == cfg_name(c)]
        summary[cfg_name(c)] = dict(
            n=len(rs), success=sum(r["success"] for r in rs),
            median_cost_drop=float(np.median([r["cost_start"] - r["cost_end"] for r in rs])),
            median_wall_s=float(np.median([r["wall_s"] for r in rs])))
        print(cfg_name(c), summary[cfg_name(c)], flush=True)
    (OUT / ("summary_quick.json" if args.quick else "summary.json")).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
