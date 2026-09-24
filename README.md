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

env = MultiPushT(n_agents=3)              # egocentric views: self blue + ring, others orange
obs, infos = env.reset(seed=0)            # obs["agent_0"] = {"pixels", "proprio", "state"}, infos[...]["goal"]
obs, rew, term, trunc, infos = env.step({a: env.action_space(a).sample() for a in env.agents})
frame = env.render()                      # audit frame: global view, identity color + index per agent
```

**Observation.** Each agent gets the single-agent Push-T keys and shapes: a 224 px `pixels` view, `proprio` `[x, y, vx, vy]` and `state` `[x, y, block_x, block_y, block_angle, vx, vy]`. The view is egocentric: shape encodes the entity type, color encodes its role relative to the viewer.

| entity | shape | appearance in agent *i*'s view |
|---|---|---|
| self (agent *i*) | circle | blue, plus a dark ring (`self_marker=ring`, default; `none` = original look) |
| any other agent | circle | orange (`others=distinct`, default). The same orange for every other agent, so it means "another agent", not an identity. `visible` draws them blue, `hidden` leaves them out |
| T-block / goal | T | gray / green, as in the original |

An agent's goal image shows the T at its goal pose and the agent itself at its goal position (other agents are left out). Agent 0's goal position is the dataset's; agents 1..N-1 keep their start position. The audit view uses identity colors that are none of blue, orange, gray or green.

**Physics.** `agent_collisions=true` (default): the original agent is a kinematic body, and pymunk never collides two kinematic bodies. So agents become heavy dynamic bodies driven by the same PD controller and kept inside the walls. The T's motion differs from the original by under 0.01 px. `success=block` (default) requires only the T pose; `pusht` is the original criterion, which also requires agent 0 at its goal position.

Checks: with `n_agents=1`, kinematic agents and `self_marker=none`, the env is bit-identical to swm PushT (pixels and 200-step trajectories). `pettingzoo.test.parallel_api_test` passes.

### Independent LeWM planners

`eval_multi.py` gives every agent its own LeWM instance and its own CEM solver (seed + i), planning on its own view toward its own goal image. It follows the `eval.py` protocol: the same 50 dataset episodes and start states, with a budget of 50 steps. Agent 0 and the T start from the dataset state. Other agents spawn at seeded random positions away from the T, so every run sees the same episodes and placements.

```bash
python eval_multi.py env.n_agents=3                    # defaults: egocentric views + ring, per-agent goal renders
python eval_multi.py env.n_agents=3 env.self_marker=none
python eval_multi.py 'policies=[lewm/pusht,random]'    # control: LeWM + random agent
python scripts/summarize_multi.py --root data/multipusht/egocentric --ref 1a_plain
```

Each run writes `results.json` and one audit video per episode (audit | each agent's view | goal) to `$STABLEWM_HOME/multipusht/<run>/`.

**Harness check.** The following reproduces `eval.py` exactly: 98%, with the same failed episode (14) and the same per-step trajectories.

```bash
python eval_multi.py env.n_agents=1 env.agent_collisions=false env.success=pusht env.self_marker=none \
    eval.goal=dataset eval.dataset_first_frame=true eval.swm_reset=true
```

When comparing videos, note that swm records frames only after each step, while `eval_multi.py` also stores t=0. With `success=block`, an episode ends as soon as the T is in place, so the agent does not travel on to its goal position as it does in the official videos.

**Pre-solved episodes.** In 14 of the 50 official episodes the T is already within tolerance at t=0 (the goal is only 25 steps ahead). Under `success=block` these count as successes, so the tables also report success on the 36 non-trivial episodes.

### Results (50 paired episodes, seed 42, `success=block`)

Egocentric views (orange others), with and without the self ring. *helped* / *hurt* count episodes won or lost relative to one agent with the same look:

| agents | self ring | success | non-trivial (36) | block contact per agent | both touching | helped / hurt |
|---|---|---|---|---|---|---|
| 1 | no | 96% | 94% | 0.59 | - | - |
| 2 | no | 98% | 97% | 0.59 / 0.04 | 0.03 | 2 / 1 |
| 3 | no | 80% | 72% | 0.53 / 0.04 / 0.03 | 0.03 | 2 / 10 |
| 1 | yes | 86% | 81% | 0.55 | - | - |
| 2 | yes | 82% | 75% | 0.53 / 0.03 | 0.01 | 2 / 4 |
| 3 | yes | 74% | 64% | 0.50 / 0.03 / 0.05 | 0.04 | 3 / 9 |

Earlier view modes, all with the dataset goal frame for every agent:

| run | success | non-trivial (36) | helped / hurt vs 1 agent |
|---|---|---|---|
| 1 LeWM | 100% | 100% | - |
| 2 LeWM, others hidden | 94% | 92% | 0 / 3 |
| 2 LeWM, others orange | 92% | 89% | 0 / 4 |
| 2 LeWM, others blue (identical) | 58% | 44% | 0 / 21 |
| LeWM + random, identical | 62% | 47% | 0 / 19 |

What this shows:
- **No cooperation emerges at N = 1, 2 or 3.** Independent planners almost never push together (≤ 4% of steps). They solve at most 3 episodes the single agent missed, which is within noise.
- **Egocentric color coding fixes the identity problem.** With identical blue disks, 2 agents drop to 58%, the same as with a random partner, which fits agent 0 not knowing which disk it moves. With orange others, 2 agents match one agent (98% vs 96%).
- **The ring costs a pretrained LeWM 6–16 points,** already with one agent (96% → 86%; 2 agents 98% → 82%). The checkpoint never saw a ring on itself, so the ring is out of distribution until a model is trained on this rendering.
- **3 agents lose through physical interference.** Without the ring, another agent touched the T in all 10 episodes the 3-agent team loses relative to one agent, and when no other agent touches it, success is 100% (with the ring: 10 of 12 losses). The uncoordinated planners bump the T off course (`egocentric/overview_ep10.png`).

## Roadmap

| Level | Setting | Question |
|---|---|---|
| 0 | Push-T, 1 agent | Can the WM learn contact dynamics? |
| 1 | MultiPush-T, 2 agents, cooperative | Does N=1 physics transfer compositionally to N=2? |
| 2 | 2v2 | Can the WM support cooperative/competitive planning? |
| 3 | + movable tools | Does self-play produce emergent strategies? |
