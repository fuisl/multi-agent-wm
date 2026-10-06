"""N independent LeWM planners acting at once in multi-agent Push-T.

Same protocol as eval.py (same dataset episodes, start states, goal frames, budget and
solver), except that every agent runs its own model + CEM solver on its own view (joint-action
checkpoints: `joint=decentralized` or `joint=centralized`, see config/eval/multipusht.yaml).
Agent 0 and the block start from the dataset state; agents 1..N-1 are placed at random
(seeded) positions away from the block. Writes results.json and one audit video per episode.
"""

import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))  # ./data -> big disk, see scripts/setup_storage.sh

import copy
import json
import time
from types import SimpleNamespace

import cv2
import hydra
import imageio
import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.world.world import _extract_init_goal

from eval import get_dataset, img_transform
from world import MultiPushT


class LazyWorldModelPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that skips image preprocessing on steps where no env replans.

    swm preprocesses every env's pixels + goal on CPU at every step, although it only
    replans every receding_horizon * action_block steps; the actions are unchanged.
    """

    def _prepare_info(self, info_dict):
        dead = np.asarray(info_dict.get("terminated", np.zeros(self.env.num_envs)), dtype=bool)
        if all(len(buf) > 0 or d for buf, d in zip(self._action_buffer, dead)):
            return {"terminated": info_dict.get("terminated")}
        return super()._prepare_info(info_dict)


def sample_eval_starts(cfg, dataset):
    """Identical to eval.py: same seed -> same (episode, start step) pairs."""
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx, step_idx = dataset.get_col_data(col_name), dataset.get_col_data("step_idx")
    ep_indices = np.unique(episode_idx)
    max_step = np.full(episode_idx.max() + 1, -1)
    np.maximum.at(max_step, episode_idx, step_idx)  # = eval.get_episodes_length, vectorized
    episode_len = max_step[ep_indices] + 1
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in episode_idx])
    valid_indices = np.nonzero(step_idx <= max_start_per_row)[0]

    g = np.random.default_rng(cfg.seed)
    rows = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    rows = np.sort(valid_indices[rows])
    data = dataset.get_row_data(rows)
    return data[col_name].tolist(), data["step_idx"].tolist()


def fit_process(cfg, dataset):
    """Identical to eval.py: z-score scalers fitted on the dataset columns."""
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        processor.fit(col_data[~np.isnan(col_data).any(axis=1)])
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor
    return process


def train_dataset_name(name):
    """Dataset a checkpoint was trained on, from the config.yaml train.py saves next to it (None if absent).

    The action scaler must match training: the model was fitted on actions z-scored with that dataset's stats.
    """
    ckpt = Path(swm.data.utils.get_cache_dir(), "checkpoints", name)
    cfg_path = (ckpt.parent if ckpt.suffix == ".pt" else ckpt) / "config.yaml"
    if not cfg_path.exists():
        return None
    train_cfg = OmegaConf.load(cfg_path)
    name = OmegaConf.select(train_cfg, "scalers_from") or OmegaConf.select(train_cfg, "data.dataset.name")
    return name.removesuffix(".h5")


def tile_process(process, n):
    """Action scaler repeated n times: joint actions [self, partner] are z-scored per agent alike, as in train.py."""
    act = copy.deepcopy(process["action"])
    act.mean_, act.var_, act.scale_ = np.tile(act.mean_, n), np.tile(act.var_, n), np.tile(act.scale_, n)
    act.n_features_in_ *= n
    return {**process, "action": act}


ANGLE_W = 20.0 / np.radians(20)  # probe cost: 20 deg weighs like 20 px, the two success tolerances


def use_probe_cost(model, path):
    """Replace the planning cost ||z_hat - z_goal||^2 by the success test's quantities read off the latents with a
    probe (scripts/fit_cost_probe.py): |self xy|^2 + |T xy|^2 + (ANGLE_W * T angle)^2 between prediction and goal, px^2."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    from probe import make_probe
    ck = torch.load(Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / path, map_location="cpu", weights_only=False)
    probe = make_probe(ck["kind"], ck["in_dim"], ck["out_dim"])
    probe.load_state_dict(ck["state_dict"])
    probe = probe.to("cuda").eval().requires_grad_(False)
    m, s = (torch.as_tensor(ck[k], dtype=torch.float32, device="cuda") for k in ("mean", "std"))

    def criterion(info_dict):
        pred = probe(info_dict["predicted_emb"][..., -1, :].float()) * s + m   # (B, S, 6)
        goal = probe(info_dict["goal_emb"][..., -1, :].float()) * s + m
        goal = goal.expand_as(pred)
        ang = torch.atan2(pred[..., 4], pred[..., 5]) - torch.atan2(goal[..., 4], goal[..., 5])
        ang = torch.atan2(torch.sin(ang), torch.cos(ang))
        return (pred[..., :4] - goal[..., :4]).pow(2).sum(-1) + (ANGLE_W * ang).pow(2)

    model.criterion = criterion


def use_action_penalty(model, weight):
    """Add weight * mean(a_z^2) to each candidate's cost. Actions are z-scored by the data's scaler, so this keeps
    plans near the action distribution the model was trained on, where its predictions hold (open-loop audit)."""
    get_cost = model.get_cost

    def penalized(info_dict, action_candidates):  # candidates (B, S, H, D)
        return get_cost(info_dict, action_candidates) + weight * action_candidates.pow(2).mean(dim=(2, 3))

    model.get_cost = penalized


def load_policy(name, cfg, n_envs, process, transform, seed, action_dim=2):
    """One independent LeWM planner (own model instance + own CEM solver), or None for random."""
    if name == "random":
        return None
    model = swm.wm.utils.load_pretrained(name).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if cfg.get("cost", "latent") == "probe":
        use_probe_cost(model, cfg.cost_probe)
    if cfg.get("action_penalty"):
        use_action_penalty(model, float(cfg.action_penalty))
    solver = hydra.utils.instantiate(cfg.solver, model=model, seed=seed)
    policy = LazyWorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process, transform=transform
    )
    # the planner only needs the batched action space of "its" env
    policy.set_env(SimpleNamespace(
        num_envs=n_envs,
        action_space=spaces.Box(-1.0, 1.0, (n_envs, action_dim), np.float32),
        single_action_space=spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
    ))
    return policy


def agent_info(obs, agent, goal, last_action, done):
    """Per-agent info dict in the exact format WorldModelPolicy gets from swm.World."""
    stack = lambda k: np.stack([o[agent][k] for o in obs])[:, None]
    return {
        "pixels": stack("pixels"),
        "proprio": stack("proprio"),
        "state": stack("state"),
        "action": last_action[:, None],
        "goal": goal["goal"][:, None],
        "goal_proprio": goal["goal_proprio"][:, None],
        "goal_state": goal["goal_state"][:, None],
        "terminated": done.copy(),
    }


def panel(env, obs, goal_img, text, size=448):
    """Audit frame | each agent's own view | goal image."""
    tiles = [env.core.render_audit(text)]
    for i, ag in enumerate(env.possible_agents):
        tiles.append(obs[ag]["pixels"].copy())
    tiles.append(goal_img.copy())
    labels = ["audit"] + [f"agent {i} view" for i in range(env.core.n_agents)] + ["goal"]
    out = []
    for tile, label in zip(tiles, labels):
        tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, label, (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        out.append(tile)
    return np.concatenate(out, axis=1)


@hydra.main(version_base=None, config_path="./config/eval", config_name="multipusht")
def run(cfg: DictConfig):
    n_agents = cfg.env.n_agents
    names = list(cfg.policies) if cfg.get("policies") else [cfg.policy] * n_agents
    assert len(names) == n_agents, f"{len(names)} policies for {n_agents} agents"
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget

    #########################
    ##   data / episodes   ##
    #########################

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    episodes, start_steps = sample_eval_starts(cfg, dataset)
    init, goal, _ = _extract_init_goal(dataset, episodes, start_steps, cfg.eval.goal_offset_steps)
    n = len(episodes)

    # scalers per policy, fitted on the dataset that policy was trained on (default: the eval dataset,
    # which is the official checkpoint's training data)
    norm_names = [train_dataset_name(name) if name != "random" and cfg.eval.scalers == "train" else None for name in names]
    norm_names = [nm or cfg.eval.dataset_name for nm in norm_names]
    processes = {nm: fit_process(cfg, dataset if nm == cfg.eval.dataset_name else get_dataset(cfg, nm)) for nm in set(norm_names)}
    print("action/proprio scalers fitted on:", dict(zip(names, norm_names)))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    #########################
    ##   envs / policies   ##
    #########################

    envs = [MultiPushT(max_episode_steps=10 * cfg.eval.eval_budget, **cfg.env) for _ in range(n)]
    obs, infos, solved_at_start = [], [], np.zeros(n, dtype=bool)
    block_placed = np.zeros(n, dtype=bool)  # T already within the success tolerance, so it need not move
    for e, env in enumerate(envs):
        if cfg.eval.swm_reset:  # swm.World order: random reset, then set state / goal (1 agent only)
            assert n_agents == 1, "eval.swm_reset only reproduces the single-agent protocol"
            assert cfg.eval.goal == "dataset", "eval.swm_reset sets the goal after reset, so goal renders would be stale"
            env.reset()
            env.core._set_state(init["state"][e])
            env.core._set_goal_state(goal["goal_state"][e])
            o, i = env._obs(), env._infos()
        else:
            o, i = env.reset(seed=cfg.seed + e, options={"state": init["state"][e], "goal_state": goal["goal_state"][e]})
        obs.append(o)
        infos.append(i)
        solved_at_start[e] = env.core.eval_state(env.core.goal_state, env.state())[0]  # T already at the goal
        block_err, angle_err, _ = env.core.errors(env.core.goal_state, env.state())
        block_placed[e] = block_err < 20 and angle_err < np.pi / 9
    agent_goals = []
    for i, ag in enumerate(envs[0].possible_agents):
        g = {k: goal[k] for k in ("goal", "goal_proprio", "goal_state")}
        if cfg.eval.goal == "render":  # env-rendered goal for this agent's view instead of the dataset goal frame
            g["goal"] = np.stack([inf[ag]["goal"] for inf in infos])
        agent_goals.append(g)

    joint = cfg.get("joint")
    assert joint in (None, "decentralized", "centralized"), f"joint={joint}"
    if joint:
        assert n_agents == 2 and "random" not in names, "joint-action planning: 2 agents, both LeWM"
        processes = {nm: tile_process(p, 2) for nm, p in processes.items()}
        # centralized: only agent 0's planner exists; agent 1 executes the partner half of its plan
        n_planners = 1 if joint == "centralized" else n_agents
        policies = [load_policy(names[i], cfg, n, processes[norm_names[i]], transform, cfg.seed + i, action_dim=4) for i in range(n_planners)]
    else:
        policies = [load_policy(name, cfg, n, processes[nm], transform, cfg.seed + i) for i, (name, nm) in enumerate(zip(names, norm_names))]
    rng = np.random.default_rng(cfg.seed)

    #########################
    ##        rollout      ##
    #########################

    done = np.zeros(n, dtype=bool)
    success_step = np.full(n, -1)
    last_action = [np.zeros((n, 2), dtype=np.float32) for _ in range(n_agents)]
    block_contact = np.zeros((n, cfg.eval.eval_budget, n_agents), dtype=bool)
    agent_contact = np.zeros((n, cfg.eval.eval_budget, n_agents), dtype=bool)
    steps = np.zeros(n, dtype=int)
    trace = [[] for _ in range(n)]  # per step: T pos err, T angle err, each agent's distance to its goal

    def log(e):
        block, angle, agents = envs[e].core.errors(envs[e].core.goal_state, envs[e].state())
        trace[e].append([round(block, 2), round(angle, 3), *np.round(agents, 2).tolist()])

    for e in range(n):
        log(e)
    frames = [[panel(env, obs[e], agent_goals[0]["goal"][e], "t=0")] for e, env in enumerate(envs)] if cfg.eval.video else None

    start_time = time.time()
    for t in range(cfg.eval.eval_budget):
        actions = []
        if joint == "centralized":  # agent 0's plan is [agent 0, agent 1]
            a = policies[0].get_action(agent_info(obs, envs[0].possible_agents[0], agent_goals[0], np.concatenate(last_action, 1), done))
            actions = [a[:, :2], a[:, 2:]]
        else:
            for i, (ag, policy) in enumerate(zip(envs[0].possible_agents, policies)):
                if policy is None:
                    a = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
                elif joint == "decentralized":  # plan [self, partner] on own view, execute own half
                    joint_last = np.concatenate([last_action[i], last_action[1 - i]], 1)
                    a = policy.get_action(agent_info(obs, ag, agent_goals[i], joint_last, done))[:, :2]
                else:
                    info = agent_info(obs, ag, agent_goals[i], last_action[i], done)
                    if t == 0 and cfg.eval.dataset_first_frame:  # what swm.World does (single agent only)
                        info["pixels"] = init["pixels"][:, None]
                    a = policy.get_action(info)
                actions.append(a)
        for i, a in enumerate(actions):
            last_action[i] = np.nan_to_num(a)

        for e, env in enumerate(envs):
            if done[e]:
                continue
            obs[e], _, term, _, info = env.step({ag: actions[i][e] for i, ag in enumerate(env.possible_agents)})
            steps[e] = t + 1
            log(e)
            for i, ag in enumerate(env.possible_agents):
                block_contact[e, t, i] = info[ag]["block_contact"]
                agent_contact[e, t, i] = info[ag]["agent_contact"]
            if term[env.possible_agents[0]]:
                done[e], success_step[e] = True, t + 1
            if frames is not None:
                frames[e].append(panel(env, obs[e], agent_goals[0]["goal"][e], f"t={t + 1}" + (" SUCCESS" if done[e] else "")))
        if done.all():
            break
    eval_time = time.time() - start_time

    #########################
    ##       results       ##
    #########################

    out_dir = Path(swm.data.utils.get_cache_dir(), cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_out = []
    for e, env in enumerate(envs):
        s, g = env.state(), env.core.goal_state
        N = n_agents
        angle = abs(s[2 * N + 2] - g[2 * N + 2])
        bc = block_contact[e, : steps[e]]
        episodes_out.append({
            "episode": int(episodes[e]), "start_step": int(start_steps[e]),
            "success": bool(done[e]), "success_step": int(success_step[e]), "solved_at_start": bool(solved_at_start[e]),
            "block_placed_at_start": bool(block_placed[e]),
            "block_pos_err": float(np.linalg.norm(s[2 * N : 2 * N + 2] - g[2 * N : 2 * N + 2])),
            "block_angle_err": float(min(angle, 2 * np.pi - angle)),
            "block_contact_frac": bc.mean(0).tolist(),            # per agent
            "co_contact_frac": float((bc.sum(1) >= 2).mean()),    # >= 2 agents touching the block
            "agent_contact_frac": float(agent_contact[e, : steps[e]].any(1).mean()),
            "trace": trace[e],
        })
        if frames is not None:
            tag = "success" if done[e] else "fail"
            imageio.mimsave(out_dir / f"ep{e:02d}_{tag}.mp4", frames[e], fps=10, macro_block_size=1)

    summary = {
        "success_rate": 100.0 * float(done.mean()),
        "n_solved_at_start": int(solved_at_start.sum()),
        "success_rate_nontrivial": 100.0 * float(done[~solved_at_start].mean()) if (~solved_at_start).any() else None,
        "n_block_placed_at_start": int(block_placed.sum()),
        "success_rate_block_moves": 100.0 * float(done[~block_placed].mean()) if (~block_placed).any() else None,  # episodes where the T must move
        "mean_success_step": float(success_step[done].mean()) if done.any() else None,
        "block_contact_frac": np.mean([ep["block_contact_frac"] for ep in episodes_out], 0).tolist(),
        "co_contact_frac": float(np.mean([ep["co_contact_frac"] for ep in episodes_out])),
        "agent_contact_frac": float(np.mean([ep["agent_contact_frac"] for ep in episodes_out])),
        "evaluation_time": eval_time,
    }
    print(json.dumps(summary, indent=2))
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": OmegaConf.to_container(cfg, resolve=True), "summary": summary, "episodes": episodes_out}, f, indent=2)
    print(f"results + videos -> {out_dir}")


if __name__ == "__main__":
    run()
