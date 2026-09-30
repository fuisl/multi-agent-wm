"""Scripted CoopPolicy (collect.py) on the eval_multi.py episodes (same start / goal states) in the
force-threshold env: shows the task is solvable by two agents.

    python scripts/coop_oracle.py [budget=300] [n_videos=6]   # -> $STABLEWM_HOME/multipusht/coop/oracle
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))
import numpy as np, imageio
from hydra import compose, initialize_config_dir
from stable_worldmodel.world.world import _extract_init_goal
from eval import get_dataset
from eval_multi import sample_eval_starts
from world import MultiPushT
from collect import CoopPolicy

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 300
n_video = int(sys.argv[2]) if len(sys.argv) > 2 else 6
out = f"{os.environ['STABLEWM_HOME']}/multipusht/coop/oracle"
os.makedirs(out, exist_ok=True)
with initialize_config_dir(config_dir=str(ROOT / "config" / "eval"), version_base=None):
    cfg = compose(config_name="multipusht")
dataset = get_dataset(cfg, cfg.eval.dataset_name)
episodes, start_steps = sample_eval_starts(cfg, dataset)
init, goal, _ = _extract_init_goal(dataset, episodes, start_steps, cfg.eval.goal_offset_steps)

res = []
for e in range(len(episodes)):
    env = MultiPushT(n_agents=2, agent_force=1e5, solver_iterations=50, max_episode_steps=budget)
    env.reset(seed=cfg.seed + e, options={"state": init["state"][e], "goal_state": goal["goal_state"][e]})
    b, a, _ = env.core.errors(env.core.goal_state, env.state())
    placed0 = b < 20 and a < np.pi / 9
    pol = CoopPolicy(np.random.default_rng(e))
    pol.reset(env.core, 0)
    frames = [env.render()] if e < n_video else None
    t, success, co = 0, False, 0
    while env.agents:
        acts = {ag: pol(env.core, i) for i, ag in enumerate(env.possible_agents)}
        _, _, term, _, info = env.step(acts)
        t += 1
        co += info["agent_0"]["block_contact"] and info["agent_1"]["block_contact"]
        success = term["agent_0"]
        if frames is not None:
            frames.append(env.render())
    b, a, ag = env.core.errors(env.core.goal_state, env.state())
    res.append(dict(ep=e, success=bool(success), steps=t, placed_at_start=bool(placed0), block_err=b, angle_err=float(np.degrees(a)),
                    agent0_err=float(ag[0]), co_contact=co / t))
    if frames is not None:
        imageio.mimsave(f"{out}/ep{e:02d}_{'success' if success else 'fail'}.mp4", frames, fps=10, macro_block_size=1)
    print(f"ep{e:02d} {'OK ' if success else 'FAIL'} steps={t:3d} placed0={placed0} T err {b:5.1f}px {np.degrees(a):5.1f}deg agent0 {ag[0]:5.1f}px co-contact {co / t:.2f}", flush=True)

s = np.array([r["success"] for r in res]); st = np.array([r["steps"] for r in res]); p0 = np.array([r["placed_at_start"] for r in res])
print(f"success {s.mean():.0%} (T must move: {s[~p0].mean():.0%} of {(~p0).sum()}), within 50 steps {(s & (st <= 50)).mean():.0%}, "
      f"median steps {np.median(st[s]) if s.any() else None}")
json.dump(res, open(f"{out}/results.json", "w"), indent=1)
