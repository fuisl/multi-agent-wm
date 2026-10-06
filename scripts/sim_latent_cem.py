"""Centralized CEM on the true simulator, scored by the true task cost or by a latent cost: the dynamics x cost 2x2.

                 true-state cost            latent cost ||z - z_goal||^2
  true simulator cost=true (search ceiling)  cost=<model>[_visible|_sum]   <- this script
  world model    (probe-cost planning, G5)   the coop2 eval (jobs/coop2_eval_b150.txt, seed 0)

With the simulator as the model there is no model error, so a latent cost that fails here fails on its own: it does
not reward the walk to the contact spots or the joint push. The search mirrors the latent planner of the eval
(plan_config horizon 5 x 5 env steps, receding horizon 5, CEM 300 samples x 30 iterations, top 30, z-scored
actions, warm start from the shifted plan), and plans both agents' actions (centralized). Candidates are rolled out
in a copy of the simulator; for a latent cost, agent 0's own view at the end of each candidate is rendered (JPEG q95,
as the eval) and encoded.

Cost variants: true | <model> (agent 0 view vs its eval goal, partner hidden) | <model>_visible (goal with the
partner at its goal spot) | <model>_sum (agent 0 + agent 1 views, each vs its own eval goal).

    python scripts/sim_latent_cem.py --costs true coop2_ftfull coop2_ftfull_visible --procs 8
-> data/multipusht/coop/sim_latent_cem/<cost>/ep<e>.json, summary.json   (eval seed 0, episodes where the T must move)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("STABLEWM_HOME", str(REPO / "data"))

import numpy as np

from oracle_cem import SimModel, cost as true_cost

OUT = Path(os.environ["STABLEWM_HOME"]) / "multipusht/coop/sim_latent_cem"
MODELS = {
    "lewm": "lewm/pusht/weights.pt",
    "coop2_ftfull": "lewm_coop2_ftfull/weights_epoch_30.pt",
    "coop2_joint": "lewm_coop2_joint/weights_epoch_30.pt",
}
ENV = dict(n_agents=2, others="distinct", agent_collisions=True, agent_force=1e5, solver_iterations=50,
           success="pusht", exact_reset_render=True, jpeg_quality=95)
BUDGET, FRAMESKIP = 150, 5
CEM = dict(H=5, samples=300, iters=30, elites=30, replan=5)

_models = {}


def latent_scorer(variant, real_core):
    """variant -> f(list of per-candidate (agent 0 view, agent 1 view or None)) -> costs, plus which views it needs."""
    import torch
    from probe import load_model, MEAN, STD
    name = variant.removesuffix("_visible").removesuffix("_sum")
    if name not in _models:
        _models[name] = load_model(MODELS[name], "cuda")
    model = _models[name]

    @torch.no_grad()
    def enc(px):
        x = torch.from_numpy(np.stack(px)).cuda().permute(0, 3, 1, 2).float().div(255)
        x = (x - MEAN.cuda()) / STD.cuda()
        Z = []
        for lo in range(0, len(x), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Z.append(model.projector(model.encoder(x[lo:lo + 1024], interpolate_pos_encoding=True).last_hidden_state[:, 0]).float())
        return torch.cat(Z)

    if variant.endswith("_visible"):
        with real_core._placed(real_core.goal_state):
            goals = [real_core.render_view(i, others="distinct") for i in range(2)]
    else:
        goals = list(real_core.goals)  # the eval goals: partner hidden
    zg = enc(goals)
    agents = (0, 1) if variant.endswith("_sum") else (0,)

    def score(views):  # views: (n_cand, n_agents_used, H, W, 3)
        c = 0
        for j, i in enumerate(agents):
            c = c + (enc([v[j] for v in views]) - zg[i]).pow(2).sum(-1)
        return c.cpu().numpy()

    return agents, score


def plan(model, real, goal, mean, rng, mu, sd, variant, scorer):
    T = mean.shape[0]
    model.load(real)
    snap = model.snapshot()
    std = np.ones_like(mean)
    lo, hi = (-1 - mu) / sd, (1 - mu) / sd
    for _ in range(CEM["iters"]):
        cand = np.clip(mean[None] + std[None] * rng.standard_normal((CEM["samples"], T, 2, 2)), lo, hi)
        cand[0] = mean
        costs, views = np.empty(len(cand)), []
        for k, z in enumerate(cand):
            model.restore(snap)
            for a in z:
                model.core.simulate((a * sd + mu).astype(np.float32))
            if variant == "true":
                costs[k] = true_cost(model.core, goal)
            else:
                views.append([model.core.render_view(i) for i in scorer[0]])
        if variant != "true":
            costs = scorer[1](views)
        top = np.argsort(costs)[: CEM["elites"]]
        mean, std = cand[top].mean(0), np.maximum(cand[top].std(0), 0.05)
    return mean, float(costs[top].mean())


def run_episode(args):
    variant, seed, e, (mu, sd, init_state, goal_state) = args
    from world import MultiPushT
    out = OUT / variant / f"s{seed}_ep{e:02d}.json"
    if out.exists():
        return json.loads(out.read_text())
    rng = np.random.default_rng([seed, e])
    real = MultiPushT(max_episode_steps=10 * BUDGET, **ENV)
    real.reset(seed=seed + e, options={"state": init_state, "goal_state": goal_state})  # as eval_multi.py
    sim = MultiPushT(max_episode_steps=10 * BUDGET, **ENV)
    sim.reset(seed=seed + e, options={"state": init_state, "goal_state": goal_state})
    model, goal = SimModel(sim.core), real.core.goal_state
    scorer = None if variant == "true" else latent_scorer(variant, real.core)
    mean = np.zeros((CEM["H"] * FRAMESKIP, 2, 2))
    trace, t0, success_step, plan_costs, t, co = [], time.time(), -1, [], 0, 0
    while t < BUDGET:
        mean, pc = plan(model, real.core, goal, mean, rng, mu, sd, variant, scorer)
        plan_costs.append(pc)
        n_exec = CEM["replan"] * FRAMESKIP
        for a in mean[:n_exec]:
            _, _, term, _, info = real.step({ag: (a[i] * sd + mu).astype(np.float32) for i, ag in enumerate(real.possible_agents)})
            t += 1
            co += info["agent_0"]["block_contact"] and info["agent_1"]["block_contact"]
            trace.append(round(true_cost(real.core, goal), 2))
            if term["agent_0"]:
                success_step = t
                break
            if t >= BUDGET:
                break
        if success_step > 0:
            break
        mean = np.concatenate([mean[n_exec:], np.zeros((n_exec, 2, 2))])
    block, angle, agents = real.core.errors(goal, real.state())
    res = dict(variant=variant, seed=seed, episode=e, success=success_step > 0, success_step=success_step, steps=t,
               cost_start=trace[0], cost_end=trace[-1], cost_min=min(trace), block_pos_err=block, block_angle_err=angle,
               agent_dist=agents.tolist(), co_contact_frac=co / t, plan_costs=plan_costs, trace=trace,
               wall_s=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"{variant:24s} s{seed} ep{e:02d} success={res['success']} cost {res['cost_start']:.0f} -> {res['cost_end']:.0f} "
          f"co-contact {res['co_contact_frac']:.2f} ({res['wall_s']:.0f}s)", flush=True)
    return res


def eval_episodes(seed):
    """eval_multi.py's episodes for this seed -> scaler, starts, goals, must-move mask (as oracle_cem.py)."""
    from omegaconf import OmegaConf
    from stable_worldmodel.world.world import _extract_init_goal
    from eval import get_dataset
    from eval_multi import sample_eval_starts
    from world import MultiPushT
    cfg = OmegaConf.load(REPO / "config/eval/multipusht.yaml")
    cfg.cache_dir, cfg.seed = None, seed
    ds = get_dataset(cfg, cfg.eval.dataset_name)
    eps, starts = sample_eval_starts(cfg, ds)
    init, goal, _ = _extract_init_goal(ds, eps, starts, cfg.eval.goal_offset_steps)
    act = ds.get_col_data("action")
    act = act[~np.isnan(act).any(1)]
    mu, sd = act.mean(0), act.std(0)
    must_move = []
    for e in range(len(eps)):
        env = MultiPushT(max_episode_steps=10, **ENV)
        env.reset(seed=seed + e, options={"state": init["state"][e], "goal_state": goal["goal_state"][e]})
        b, a, _ = env.core.errors(env.core.goal_state, env.state())
        must_move.append(not (b < 20 and a < np.pi / 9))
    return mu, sd, init["state"], goal["goal_state"], np.array(must_move)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--costs", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=0, help="first n must-move episodes (0 = all)")
    p.add_argument("--procs", type=int, default=8)
    args = p.parse_args()
    mu, sd, init, goal, must_move = eval_episodes(args.seed)
    eps = np.nonzero(must_move)[0]
    eps = eps[: args.episodes] if args.episodes else eps
    print(f"seed {args.seed}: {must_move.sum()} must-move episodes, running {len(eps)} x {args.costs}", flush=True)
    jobs = [(v, args.seed, int(e), (mu, sd, init[e], goal[e])) for e in eps for v in args.costs]
    jobs.sort(key=lambda j: j[0] == "true")  # latent variants (slow) first
    with get_context("spawn").Pool(args.procs) as pool:
        results = list(pool.imap_unordered(run_episode, jobs))
    summary = {}
    for v in args.costs:
        rs = [r for r in results if r["variant"] == v]
        summary[v] = dict(n=len(rs), success=sum(r["success"] for r in rs),
                          median_cost_drop=float(np.median([r["cost_start"] - r["cost_end"] for r in rs])),
                          median_T_err_end_px=float(np.median([r["block_pos_err"] for r in rs])),
                          median_co_contact=float(np.median([r["co_contact_frac"] for r in rs])),
                          median_wall_s=float(np.median([r["wall_s"] for r in rs])))
        print(v, summary[v], flush=True)
    (OUT / f"summary_s{args.seed}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
