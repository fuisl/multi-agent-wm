# Multi-Agent World Model (MultiPush-T)

A minimal, [LeWM](https://github.com/lucas-maes/le-wm)-style codebase for studying whether an object-centric world model learned from single-agent physical interaction (Push-T) transfers to **N agents cooperatively pushing a T-block**, and whether planning through it produces coordination.

Like LeWM, this repo contains only the core contribution. [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) handles environments, data, planning and evaluation, [stable-pretraining](https://github.com/galilai-group/stable-pretraining) handles training, and [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) provides the multi-agent 2D physics.

> **Status:** `jepa.py`, `module.py`, `train.py`, `eval.py` and `utils.py` are the official LeWM code, and they reproduce the LeWM Push-T checkpoint (see below). `world.py` and `collect.py` (MultiPush-T) are still scaffolds.

## Layout

| File | Role |
|---|---|
| `world.py` | `MultiPushTScenario` (VMAS scenario, per-agent obs/reward) + `MultiPushT` (centralized Gym env registered as `swm/MultiPushT-v0`) |
| `collect.py` | Roll out a policy in the world, write a dataset to `$STABLEWM_HOME/datasets` |
| `jepa.py` | World model: `encode`, `predict`, `rollout`, `criterion`, `get_cost` |
| `module.py` | Building blocks: `SIGReg`, `ARPredictor`, `Embedder`, `MLP`, Transformer blocks |
| `train.py` | Training loop (`lejepa_forward`, Hydra `run`) |
| `eval.py` | MPC evaluation (CEM / Adam) from dataset start/goal pairs |
| `utils.py` | Preprocessing, normalizers, checkpoint callback |
| `config/` | Hydra configs: `collect/`, `train/`, `eval/` |
| `scripts/` | `setup_storage.sh` (storage symlink), `download_lewm.py` (original LeWM data + checkpoints) |

**Centralized interface.** stable-worldmodel expects single-agent Gym envs, so `MultiPushT` exposes the joint action `a = [a¹, …, aᴺ] ∈ [-1,1]^{2N}` and a dynamics-sufficient state:

```
state   = [agent_1 .. agent_N, block]    agent_i = [x, y, vx, vy]
                                         block   = [x, y, vx, vy, sinθ, cosθ, ω]
proprio = [agent_1 .. agent_N]
```

The action encoder's `input_dim` is set automatically to `frameskip * 2N`. Decentralized, partial per-agent observations live in `MultiPushTScenario.observation`, for later MARL / PettingZoo work.

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

## MultiPush-T

```bash
python collect.py world.n_agents=2 policy=random num_episodes=1000   # -> datasets/multipusht_2a_random
python train.py data=multipusht
python eval.py --config-name=multipusht policy=<run_name>
```

Add `--cfg job` to any command to print the resolved config.

## Roadmap

| Level | Setting | Question |
|---|---|---|
| 0 | Push-T, 1 agent | Can the WM learn contact dynamics? |
| 1 | MultiPush-T, 2 agents, cooperative | Does N=1 physics transfer compositionally to N=2? |
| 2 | 2v2 | Can the WM support cooperative/competitive planning? |
| 3 | + movable tools | Does self-play produce emergent strategies? |
