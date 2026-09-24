# Multi-Agent World Model (MultiPush-T)

A minimal, [LeWM](https://github.com/lucas-maes/le-wm)-style codebase for studying whether an object-centric world model learned from single-agent physical interaction (Push-T) transfers to **N agents cooperatively pushing a T-block**, and whether planning through it produces coordination.

Like LeWM, this repo contains only the core contribution. [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) handles environments, data, planning and evaluation, [stable-pretraining](https://github.com/galilai-group/stable-pretraining) handles training, and [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) provides the multi-agent 2D physics.

> **Status:** scaffold. Files hold function signatures and docstrings only (`NotImplementedError`).

## Layout

| File | Role |
|---|---|
| `world.py` | `MultiPushTScenario` (VMAS scenario, per-agent obs/reward) + `MultiPushT` (centralized Gym env registered as `swm/MultiPushT-v0`) |
| `collect.py` | Roll out a policy in the world, write a dataset to `$STABLEWM_HOME` |
| `jepa.py` | World model: `encode`, `predict`, `rollout`, `criterion`, `get_cost` |
| `module.py` | Building blocks: `SIGReg`, `ARPredictor`, `Embedder`, `MLP`, Transformer blocks |
| `train.py` | Training loop (`lejepa_forward`, Hydra `run`) |
| `eval.py` | MPC evaluation (CEM / Adam) from dataset start/goal pairs |
| `utils.py` | Preprocessing, normalizers, checkpoint callback |
| `config/` | Hydra configs: `collect/`, `train/`, `eval/` |

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

`box2d-py` (pulled in by `gymnasium[all]`) needs `swig` to build. `pyproject.toml` supplies it as a build dependency, so no system package is needed.

Data and checkpoints go under `$STABLEWM_HOME` (default `~/.stable_worldmodel`):

```bash
export STABLEWM_HOME=/path/to/storage
```

## Usage

```bash
# 1. collect MultiPush-T data  -> $STABLEWM_HOME/multipusht_2a_random
python collect.py world.n_agents=2 policy=random num_episodes=1000

# 2. train the world model
python train.py data=multipusht          # or data=pusht for the single-agent baseline

# 3. plan with it (policy = checkpoint path relative to $STABLEWM_HOME, no _object.ckpt suffix)
python eval.py --config-name=multipusht policy=multipusht/lewm
python eval.py --config-name=pusht      policy=pusht/lewm
```

Single-agent Push-T data (`pusht_expert_train`) and LeWM checkpoints are available from the [LeWM HuggingFace collection](https://huggingface.co/collections/quentinll/lewm). Add `--cfg job` to any command to print the resolved config.

## Roadmap

| Level | Setting | Question |
|---|---|---|
| 0 | Push-T, 1 agent | Can the WM learn contact dynamics? |
| 1 | MultiPush-T, 2 agents, cooperative | Does N=1 physics transfer compositionally to N=2? |
| 2 | 2v2 | Can the WM support cooperative/competitive planning? |
| 3 | + movable tools | Does self-play produce emergent strategies? |
