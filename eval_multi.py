"""N independent LeWM planners acting at once in multi-agent Push-T.

Same protocol as eval.py (same dataset episodes, start states, goal frames, budget and
solver), except that every agent runs its own model + CEM solver on its own view.
Agent 0 and the block start from the dataset state; agents 1..N-1 are placed at random
(seeded) positions away from the block. Writes results.json and one audit video per episode.
"""

import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))  # ./data -> big disk, see scripts/setup_storage.sh

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


def load_policy(name, cfg, n_envs, process, transform, seed):
    """One independent LeWM planner (own model instance + own CEM solver), or None for random."""
    if name == "random":
        return None
    model = swm.wm.utils.load_pretrained(name).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver = hydra.utils.instantiate(cfg.solver, model=model, seed=seed)
    policy = LazyWorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process, transform=transform
    )
    # the planner only needs the batched action space of "its" env
    policy.set_env(SimpleNamespace(
        num_envs=n_envs,
        action_space=spaces.Box(-1.0, 1.0, (n_envs, 2), np.float32),
        single_action_space=spaces.Box(-1.0, 1.0, (2,), np.float32),
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

    process = fit_process(cfg, dataset)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    #########################
    ##   envs / policies   ##
    #########################

    envs = [MultiPushT(max_episode_steps=10 * cfg.eval.eval_budget, **cfg.env) for _ in range(n)]
    obs, infos = [], []
    for e, env in enumerate(envs):
        o, i = env.reset(seed=cfg.seed + e, options={"state": init["state"][e], "goal_state": goal["goal_state"][e]})
        obs.append(o)
        infos.append(i)
    agent_goals = []
    for i, ag in enumerate(envs[0].possible_agents):
        g = {k: goal[k] for k in ("goal", "goal_proprio", "goal_state")}
        if cfg.eval.goal == "render":  # env-rendered goal for this agent's view instead of the dataset goal frame
            g["goal"] = np.stack([inf[ag]["goal"] for inf in infos])
        agent_goals.append(g)

    policies = [load_policy(name, cfg, n, process, transform, cfg.seed + i) for i, name in enumerate(names)]
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
    frames = [[panel(env, obs[e], agent_goals[0]["goal"][e], "t=0")] for e, env in enumerate(envs)] if cfg.eval.video else None

    start_time = time.time()
    for t in range(cfg.eval.eval_budget):
        actions = []
        for i, (ag, policy) in enumerate(zip(envs[0].possible_agents, policies)):
            if policy is None:
                a = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
            else:
                info = agent_info(obs, ag, agent_goals[i], last_action[i], done)
                if t == 0 and cfg.eval.dataset_first_frame:  # what swm.World does (single agent only)
                    info["pixels"] = init["pixels"][:, None]
                a = policy.get_action(info)
            last_action[i] = np.nan_to_num(a)
            actions.append(a)

        for e, env in enumerate(envs):
            if done[e]:
                continue
            obs[e], _, term, _, info = env.step({ag: actions[i][e] for i, ag in enumerate(env.possible_agents)})
            steps[e] = t + 1
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
            "success": bool(done[e]), "success_step": int(success_step[e]),
            "block_pos_err": float(np.linalg.norm(s[2 * N : 2 * N + 2] - g[2 * N : 2 * N + 2])),
            "block_angle_err": float(min(angle, 2 * np.pi - angle)),
            "block_contact_frac": bc.mean(0).tolist(),            # per agent
            "co_contact_frac": float((bc.sum(1) >= 2).mean()),    # >= 2 agents touching the block
            "agent_contact_frac": float(agent_contact[e, : steps[e]].any(1).mean()),
        })
        if frames is not None:
            tag = "success" if done[e] else "fail"
            imageio.mimsave(out_dir / f"ep{e:02d}_{tag}.mp4", frames[e], fps=10, macro_block_size=1)

    summary = {
        "success_rate": 100.0 * float(done.mean()),
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
