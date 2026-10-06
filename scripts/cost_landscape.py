"""Does the latent planning cost see progress before the T moves? (F6 cost geometry, F7 flat landscape, F11 goals)

The scripted CoopPolicy (collect.py, noise-free) plays the coop2 eval episodes (eval_multi.py protocol, seeds 0-2,
budget 150, the coop env of jobs/coop2_eval_*.txt). Every step keeps both agents' views and the true state. Each
view is encoded by each model, and the planning cost ||z_t - z_goal||^2 is taken against the agent's goal image,
rendered two ways: partner hidden (what the eval uses) and partner visible at its goal spot (its start position).

Per model x goal variant x agent view, on episodes where the T must move:
  spearman    rank correlation of the latent cost with the true task cost (oracle_cem.cost), per episode
  approach    the phase before the first joint push (both agents touch the T): relative change of the latent cost
              from t=0 to that push. < 0 means the cost rewards walking to the contact spots
  window25    pairs (t, t+25) inside the approach phase, the planner's horizon: share where the latent cost drops
  push        relative change from the first joint push to the episode's end
The true task cost is reported on the same windows, for reference.

    python scripts/cost_landscape.py    # -> data/multipusht/coop/cost_landscape/{summary.json, traces.npz}
"""
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("STABLEWM_HOME", str(REPO / "data"))

import numpy as np
import torch

OUT = REPO / "data/multipusht/coop/cost_landscape"
MODELS = {
    "lewm": "lewm/pusht/weights.pt",
    "coop2_ftfull": "lewm_coop2_ftfull/weights_epoch_30.pt",
    "coop2_joint": "lewm_coop2_joint/weights_epoch_30.pt",
}
ENV = dict(n_agents=2, others="distinct", agent_collisions=True, agent_force=1e5, solver_iterations=50,
           success="pusht", exact_reset_render=True, jpeg_quality=95)
SEEDS, BUDGET = (0, 1, 2), 150
ANGLE_W = 20.0 / np.radians(20)  # 20 deg costs like 20 px, as oracle_cem.py


def true_cost(state, goal):
    pos = np.linalg.norm(np.concatenate([goal[:2] - state[:2], goal[4:6] - state[4:6]]))
    ang = abs(goal[6] - state[6]) % (2 * np.pi)
    return pos + ANGLE_W * min(ang, 2 * np.pi - ang)


def episodes(seed):
    from omegaconf import OmegaConf
    from stable_worldmodel.world.world import _extract_init_goal
    from eval import get_dataset
    from eval_multi import sample_eval_starts
    cfg = OmegaConf.load(REPO / "config/eval/multipusht.yaml")
    cfg.cache_dir, cfg.seed = None, seed
    ds = get_dataset(cfg, cfg.eval.dataset_name)
    eps, starts = sample_eval_starts(cfg, ds)
    init, goal, _ = _extract_init_goal(ds, eps, starts, cfg.eval.goal_offset_steps)
    return [(seed, e, init["state"][e], goal["goal_state"][e]) for e in range(len(eps))]


def rollout(args):
    """One scripted episode -> views (T+1, 2, 224, 224, 3), goals (2 variants, 2 agents, ...), states, contacts."""
    seed, e, init_state, goal_state = args
    from world import MultiPushT
    from collect import CoopPolicy
    env = MultiPushT(max_episode_steps=BUDGET, **ENV)
    obs, infos = env.reset(seed=seed + e, options={"state": init_state, "goal_state": goal_state})  # as eval_multi.py
    core, ags = env.core, env.possible_agents
    goals_hidden = np.stack([infos[ag]["goal"] for ag in ags])
    with core._placed(core.goal_state):
        goals_visible = np.stack([core.render_view(i, others="distinct") for i in range(2)])
    b, a, _ = core.errors(core.goal_state, env.state())
    must_move = not (b < 20 and a < np.pi / 9)
    pol = CoopPolicy(np.random.default_rng(e))
    pol.reset(core, 0)
    views, states, contact = [np.stack([obs[ag]["pixels"] for ag in ags])], [env.state()], [[False, False]]
    success = False
    while env.agents:
        obs, _, term, _, infos = env.step({ag: pol(core, i) for i, ag in enumerate(ags)})
        views.append(np.stack([obs[ag]["pixels"] for ag in ags]))
        states.append(env.state())
        contact.append([infos[ag]["block_contact"] for ag in ags])
        success = term[ags[0]]
    goal_global = core.goal_state.copy()
    return dict(seed=seed, ep=e, must_move=must_move, success=bool(success), views=np.stack(views),
                goals=np.stack([goals_hidden, goals_visible]), states=np.stack(states), contact=np.array(contact),
                true_cost=np.array([true_cost(s, goal_global) for s in states]))


@torch.no_grad()
def encode(model, px, dev="cuda"):
    from probe import MEAN, STD
    flat = px.reshape(-1, *px.shape[-3:])
    Z = []
    for lo in range(0, len(flat), 512):
        x = torch.from_numpy(flat[lo:lo + 512]).to(dev).permute(0, 3, 1, 2).float().div(255)
        x = (x - MEAN.to(dev)) / STD.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z.append(model.projector(model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float())
    return torch.cat(Z).reshape(*px.shape[:-3], -1)


def rel(c, t0, t1):
    return float((c[t1] - c[t0]) / max(c[t0], 1e-8))


def main():
    from probe import load_model
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [j for s in SEEDS for j in episodes(s)]
    print(f"{len(jobs)} episodes", flush=True)
    models = {n: load_model(ck, "cuda") for n, ck in MODELS.items()}
    traces, meta = {}, []
    with get_context("spawn").Pool(int(os.environ.get("PROCS", 4))) as pool:
        for r in pool.imap(rollout, jobs):
            key = f"s{r['seed']}_e{r['ep']:02d}"
            both = r["contact"].all(1)
            t_push = int(np.argmax(both)) if both.any() else -1
            meta.append(dict(key=key, seed=r["seed"], ep=r["ep"], must_move=r["must_move"], success=r["success"],
                             steps=len(r["true_cost"]) - 1, first_joint_push=t_push))
            traces[f"{key}/true_cost"] = r["true_cost"]
            traces[f"{key}/contact"] = r["contact"]
            for n, m in models.items():
                zv = encode(m, r["views"])                     # (T+1, 2 agents, 192)
                zg = encode(m, r["goals"])                     # (2 variants, 2 agents, 192)
                c = (zv[None] - zg[:, None]).pow(2).sum(-1)    # (2 variants, T+1, 2 agents)
                traces[f"{key}/{n}"] = c.cpu().numpy()
            print(f"{key} must_move={r['must_move']} success={r['success']} steps={meta[-1]['steps']} first joint push {t_push}", flush=True)
    np.savez_compressed(OUT / "traces.npz", **traces)

    summary = {"episodes": meta, "n_must_move": sum(m["must_move"] for m in meta),
               "scripted_success_must_move": sum(m["success"] for m in meta if m["must_move"])}
    mm = [m for m in meta if m["must_move"] and m["first_joint_push"] > 0]
    summary["n_with_joint_push"] = len(mm)
    for n in list(MODELS) + ["true"]:
        for vi, variant in enumerate(("goal partner hidden", "goal partner visible")):
            if n == "true" and vi:
                continue
            for agent in (0, 1):
                if n == "true" and agent:
                    continue
                rows = dict(spearman=[], approach=[], window25=[], push=[])
                for m in mm:
                    tc = traces[f"{m['key']}/true_cost"]
                    c = tc if n == "true" else traces[f"{m['key']}/{n}"][vi, :, agent]
                    rk = lambda x: np.argsort(np.argsort(x))
                    rows["spearman"].append(float(np.corrcoef(rk(c), rk(tc))[0, 1]))
                    tp = m["first_joint_push"]
                    rows["approach"].append(rel(c, 0, tp))
                    rows["window25"] += [float(c[t + 25] < c[t]) for t in range(0, tp - 25 + 1)]
                    rows["push"].append(rel(c, tp, len(c) - 1))
                name = "true task cost" if n == "true" else f"{n} | {variant} | agent {agent} view"
                summary[name] = {k: (float(np.nanmedian(v)) if k != "window25" else float(np.mean(v))) if v else None
                                 for k, v in rows.items()}
                summary[name]["n_windows25"] = len(rows["window25"])
                print(name, summary[name], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
