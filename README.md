# Multi-Agent World Model (MultiPush-T)

A minimal, [LeWM](https://github.com/lucas-maes/le-wm)-style codebase for studying whether an object-centric world model learned from single-agent physical interaction (Push-T) transfers to **N agents cooperatively pushing a T-block**, and whether planning through it produces coordination.

Like LeWM, this repo contains only the core contribution. [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) handles environments, data, planning and evaluation, [stable-pretraining](https://github.com/galilai-group/stable-pretraining) handles training, and [PettingZoo](https://pettingzoo.farama.org) is the multi-agent API.

> **Status:** `jepa.py`, `module.py`, `train.py`, `eval.py` and `utils.py` are the official LeWM code, and they reproduce the LeWM Push-T checkpoint. `world.py` is the multi-agent Push-T env and `eval_multi.py` runs N independent LeWM planners in it. `collect.py` is still a scaffold.

## Layout

| File | Role |
|---|---|
| `world.py` | `MultiPushT`: N-agent Push-T as a PettingZoo `ParallelEnv` (built on swm's PushT) |
| `eval_multi.py` | N independent LeWM planners acting at once, with audit videos |
| `collect.py` | (scaffold) roll out a policy in the world, write a dataset to `$STABLEWM_HOME/datasets` |
| `jepa.py` | World model: `encode`, `predict`, `rollout`, `criterion`, `get_cost` |
| `module.py` | Building blocks: `SIGReg`, `ARPredictor`, `Embedder`, `MLP`, Transformer blocks |
| `train.py` | Training loop (`lejepa_forward`, Hydra `run`) |
| `eval.py` | MPC evaluation (CEM / Adam) from dataset start/goal pairs |
| `utils.py` | Preprocessing, normalizers, checkpoint callback |
| `config/` | Hydra configs: `collect/`, `train/`, `eval/` |
| `scripts/` | `setup_storage.sh` (storage symlink), `download_lewm.py` (original LeWM data + checkpoints), `summarize_multi.py` (results table) |

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10 (pinned in `.python-version`).

```bash
uv sync                      # creates .venv from uv.lock
source .venv/bin/activate    # or prefix commands with `uv run`
```

Notes:
- `box2d-py` (pulled in by `gymnasium[all]`) needs `swig` to build. `pyproject.toml` supplies it as a build dependency, so no system package is needed.
- `transformers` is pinned `<5`. The LeWM checkpoints use the 4.x ViT parameter names, and 5.x renamed them.

## Storage

All heavy files (datasets, checkpoints, Hydra/Lightning/W&B outputs, eval videos) go under `$STABLEWM_HOME`. Every script defaults it to `<repo>/data`, so the only per-machine step is pointing `./data` at a large disk:

```bash
scripts/setup_storage.sh /data/$USER/multi-agent-wm   # ./data -> /data/$USER/multi-agent-wm
scripts/setup_storage.sh                              # or: a plain local ./data directory
```

Exporting `STABLEWM_HOME` explicitly still overrides the default.

```
data/
├── datasets/       # *.h5 / *.lance
├── checkpoints/    # <run>/{weights*.pt, config.json}
├── outputs/        # hydra run dirs (logs, lightning, wandb)
└── archives/       # downloaded .zst (deleted after extraction unless --keep-archive)
```

## Reproducing LeWM on Push-T

```bash
# checkpoint (~70MB) + dataset (13GB download, 44GB extracted)
python scripts/download_lewm.py pusht

# plan with the official checkpoint and eval config (50 episodes, CEM, horizon 5)
python eval.py --config-name=pusht policy=lewm/pusht

# retrain from scratch with the official hyperparameters
python train.py data=pusht
```

`download_lewm.py` rewrites the HF `config.json` so that the checkpoint loads into this repo's `jepa.JEPA` / `module.*` (strict `load_state_dict`), not swm's bundled copy. `policy=` is a path relative to `$STABLEWM_HOME/checkpoints`. `python scripts/download_lewm.py all` also fetches TwoRoom, Cube and Reacher (~73GB compressed).

| Push-T (official eval config: 50 episodes, seed 42) | Success rate | Eval time (1× A100) |
|---|---|---|
| LeWM checkpoint (`policy=lewm/pusht`) | **98%** (49/50) | 95 s |
| Random policy (`policy=random`) | 0% | 27 s |

Results and per-episode videos are written to `$STABLEWM_HOME/<policy dir>/` (e.g. `data/lewm/pusht_results.txt`).

## Multi-agent Push-T

`world.MultiPushT` is swm's Push-T with N agents as a PettingZoo `ParallelEnv`: every agent acts at every step and they share the reward and termination.

```python
from world import MultiPushT

env = MultiPushT(n_agents=2, others="visible")
obs, infos = env.reset(seed=0)            # obs["agent_0"] = {"pixels", "proprio", "state"}
obs, rew, term, trunc, infos = env.step({a: env.action_space(a).sample() for a in env.agents})
frame = env.render()                      # audit frame: agents colored + numbered
```

- **Observation per agent:** the same keys and shapes as single-agent Push-T. `pixels` is the agent's own 224 px view, `proprio` is `[x, y, vx, vy]`, and `state` is `[x, y, block_x, block_y, block_angle, vx, vy]`. `infos[agent]["goal"]` holds the goal image.
- **`others`:** how other agents appear in a view. `visible` draws them like the agent itself, `distinct` uses another color, `hidden` leaves them out.
- **`agent_collisions`** (default on): the original agent is a kinematic body, and pymunk never collides two kinematic bodies, so agents would pass through each other. With this on, agents are heavy dynamic bodies driven by the same PD controller. The block dynamics differ from the original by under 0.01 px.
- **`success`:** `block` requires only the T pose (default). `pusht` is the original criterion, which also requires agent 0 at its goal position.

Checks: with `n_agents=1` and kinematic agents, the env is bit-identical to swm PushT (pixels and 200-step trajectories). `pettingzoo.test.parallel_api_test` passes.

### Independent LeWM planners

`eval_multi.py` gives every agent its own LeWM instance and its own CEM solver (seed + i), planning on its own view. Everything else follows the `eval.py` protocol: the same 50 dataset episodes, start states and goal frames, with a budget of 50 steps. Agent 0 and the T start from the dataset state. Other agents spawn at seeded random positions away from the T, so every run sees the same episodes and placements.

```bash
python eval_multi.py                                   # 2 LeWMs, others=visible
python eval_multi.py env.others=distinct               # or hidden
python eval_multi.py 'policies=[lewm/pusht,random]'    # control: LeWM + random agent
python eval_multi.py env.n_agents=1                    # single-agent reference
python eval_multi.py env.n_agents=3 eval.num_eval=10
python scripts/summarize_multi.py                      # table below
```

Each run writes `results.json` and one audit video per episode (audit view | each agent's view | goal) to `$STABLEWM_HOME/multipusht/<run>/`.

**Harness check:** `eval_multi.py env.n_agents=1 env.agent_collisions=false env.success=pusht eval.dataset_first_frame=true` gives 98%, and the one failure is the same episode (14) that fails in `eval.py`.

**Results** (50 paired episodes, seed 42, `success=block`). *helped* / *hurt* count the episodes solved or lost relative to the single agent:

| run | success | block contact (agent 0 / 1) | both touching | helped | hurt |
|---|---|---|---|---|---|
| 1 LeWM | 100% | 0.60 | - | - | - |
| 2 LeWM, `hidden` | 94% | 0.58 / 0.03 | 0.01 | 0 | 3 |
| 2 LeWM, `distinct` | 92% | 0.58 / 0.03 | 0.02 | 0 | 4 |
| 2 LeWM, `visible` | 58% | 0.44 / 0.05 | 0.01 | 0 | 21 |
| LeWM + random, `visible` | 60% | 0.44 / 0.02 | 0.01 | 0 | 20 |

What this shows so far:
- **No cooperation emerges.** The second planner never turns a failure into a success (helped = 0). It rarely touches the T, and two agents push together in only 1–2% of steps.
- **Seeing an identical second disk breaks planning.** With `visible`, success falls to 58%, the same as with a *random* second agent (60%). The damage comes from agent 0's confused perception, not from its partner's actions. The audit videos show agent 0 walking away from the T.
- **An identity cue restores most of the performance.** A different color (`distinct`) or hiding the other agent (`hidden`) brings success back to 92–94%.

This protocol leaves no room for help, because one agent already solves every episode (goals are 25 steps ahead). Measuring cooperation needs goals a single agent cannot reach, for example a larger `eval.goal_offset_steps` with a matching `eval.eval_budget`.

## Roadmap

| Level | Setting | Question |
|---|---|---|
| 0 | Push-T, 1 agent | Can the WM learn contact dynamics? |
| 1 | MultiPush-T, 2 agents, cooperative | Does N=1 physics transfer compositionally to N=2? |
| 2 | 2v2 | Can the WM support cooperative/competitive planning? |
| 3 | + movable tools | Does self-play produce emergent strategies? |
