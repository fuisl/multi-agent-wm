# Multi-Agent World Model (MultiPush-T)

A minimal, [LeWM](https://github.com/lucas-maes/le-wm)-style codebase for studying whether an object-centric world model learned from single-agent physical interaction (Push-T) transfers to **N agents cooperatively pushing a T-block**, and whether planning through it produces coordination.

Like LeWM, this repo contains only the core contribution. [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) handles environments, data, planning and evaluation, [stable-pretraining](https://github.com/galilai-group/stable-pretraining) handles training, and [PettingZoo](https://pettingzoo.farama.org) is the multi-agent API.

> **Status:** `jepa.py`, `module.py`, `train.py`, `eval.py` and `utils.py` are the official LeWM code, and they reproduce the LeWM Push-T checkpoint. `world.py` is the multi-agent Push-T env, `eval_multi.py` runs N independent LeWM planners in it, and `collect.py` records multi-agent data in the Push-T training format.

## Layout

| File | Role |
|---|---|
| `world.py` | `MultiPushT`: N-agent Push-T as a PettingZoo `ParallelEnv` (built on swm's PushT) |
| `eval_multi.py` | N independent LeWM planners acting at once, with audit videos |
| `collect.py` | Roll out a scripted or random policy in `MultiPushT`, write one egocentric episode per agent to `$STABLEWM_HOME/datasets` |
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
python train.py data=pusht split_by=window
```

`split_by=window` is the official validation split: random windows, so neighbouring windows that share 3 of their 4 frames land on both sides. Our default, `split_by=episode`, holds out whole episodes, or whole scenes for multi-agent data. With the window split, val loss tracks memorization: `lewm_coop_ftfull` reaches 0.003 there but 0.017 on unseen scenes. Like the paper, we judge a model by planning success and latent probes. The episode-level val loss is a monitor of generalization, not the target.

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

env = MultiPushT(n_agents=3)              # egocentric views: self blue, others orange
obs, infos = env.reset(seed=0)            # obs["agent_0"] = {"pixels", "proprio", "state"}, infos[...]["goal"]
obs, rew, term, trunc, infos = env.step({a: env.action_space(a).sample() for a in env.agents})
frame = env.render()                      # audit frame: global view, identity color + index per agent
```

**Observation.** Each agent gets the single-agent Push-T keys and shapes: a 224 px `pixels` view, `proprio` `[x, y, vx, vy]` and `state` `[x, y, block_x, block_y, block_angle, vx, vy]`. The view is egocentric: shape encodes the entity type, color encodes its role relative to the viewer.

| entity | shape | appearance in agent *i*'s view |
|---|---|---|
| self (agent *i*) | circle | blue, as in the original |
| any other agent | circle | orange (`others=distinct`, default). The same orange for every other agent, so it means "another agent", not an identity. `visible` draws them blue, `hidden` leaves them out |
| T-block / goal | T | gray / green, as in the original |

An agent's goal image shows the T at its goal pose and the agent itself at its goal position (other agents are left out). Agent 0's goal position is the dataset's; agents 1..N-1 keep their start position. The audit view uses identity colors that are none of blue, orange, gray or green.

**Physics.** `agent_collisions=true` (default): the original agent is a kinematic body, and pymunk never collides two kinematic bodies. So agents become heavy dynamic bodies driven by the same PD controller and kept inside the walls. The T's motion differs from the original by under 0.01 px.

**Success** (`env.success`), checked after every step; an episode ends at the first success:

| criterion | test | notes |
|---|---|---|
| `pusht` (default) | ‖[agent 0 xy, T xy] − goal‖ < 20 px **and** T angle error < 20° | The original `PushT.eval_state`: the T *and* agent 0 must be at the goal configuration shown in the goal image |
| `block` | ‖T xy − goal‖ < 20 px and T angle error < 20° | T only. Ends before the agent reaches its goal spot, and 14 of the 50 official episodes already pass at t=0 |

Checks: with `n_agents=1`, kinematic agents and `others=hidden`, the env is bit-identical to swm PushT (pixels and 200-step trajectories). `pettingzoo.test.parallel_api_test` passes.

### Independent LeWM planners

`eval_multi.py` gives every agent its own LeWM instance and its own CEM solver (seed + i), planning on its own view toward its own goal image. It follows the `eval.py` protocol: the same 50 dataset episodes and start states, with a budget of 50 steps. Agent 0 and the T start from the dataset state. Other agents spawn at seeded random positions away from the T, so every run sees the same episodes and placements.

```bash
python eval_multi.py env.n_agents=3                    # defaults: orange others, per-agent goal renders, original success test
python eval_multi.py 'policies=[lewm/pusht,random]'    # control: LeWM + random agent
python scripts/summarize_multi.py --root data/multipusht/egocentric --ref 1a
```

Each run writes `results.json` to `$STABLEWM_HOME/multipusht/<run>/`, with per-episode metrics and a per-step `trace` of [T position error, T angle error, each agent's distance to its goal spot], plus one audit video per episode (audit | each agent's view | goal).

### Single-agent audit against the official eval

The following reproduces `eval.py` exactly: 98%, with the same failed episode (14) and the same per-step trajectories.

```bash
python eval_multi.py env.n_agents=1 env.agent_collisions=false env.others=hidden \
    eval.goal=dataset eval.dataset_first_frame=true eval.swm_reset=true
```

When comparing videos, note that swm records frames only after each step, while `eval_multi.py` also stores t=0.

With the T-only test, single-agent episodes ended long before the agent reached its goal spot. Over the 49 official successes, the T-only test would fire a median of 12 steps earlier (up to 47), when the agent is still a median 52 px from its spot. For example, in episode 03 the T is placed at step 3 but the agent arrives at step 24; in episode 08 the T starts on target, but the agent needs 22 steps to cover 191 px. The official videos stop moving exactly at those steps. The original test is therefore the default.

Our default single-agent setup scores 90% rather than 98% under the same test. Changing one factor at a time from the exact official setup:

| change from the official setup | success |
|---|---|
| none (exact official protocol) | 98% |
| dynamic agents (collisions on) | 98% |
| our reset order | 96% |
| our reset order + rendered goal image instead of the dataset frame | 90% |
| first plan from our render instead of the dataset frame | 90% |

The physics change costs nothing. What costs episodes is substituting our renders for dataset images. Those renders differ only at sub-pixel edges, because the dataset stores states as float32 and swm's bilinear resize is already the closest match. Such tiny differences flip 3–4 episodes, so on this protocol a single-seed success rate carries roughly ±6–8 points of noise.

### Results (50 paired episodes, seed 42, original success test)

Orange others, per-agent goal renders. *helped* / *hurt* count episodes won or lost relative to one agent:

| agents | success | block contact per agent | both touching | agents bumping | helped / hurt |
|---|---|---|---|---|---|
| 1 | 90% | 0.39 | - | - | - |
| 2 | 76% | 0.37 / 0.06 | 0.03 | 0.02 | 3 / 10 |
| 3 | 60% | 0.33 / 0.05 / 0.05 | 0.04 | 0.05 | 2 / 17 |

Every agent given the same dataset goal frame (which shows a single disk at agent 0's goal spot):

| run | success | helped / hurt vs 1 agent (90%) |
|---|---|---|
| 2 LeWM, others orange | 56% | 2 / 19 |
| 2 LeWM, others hidden | 54% | 0 / 18 |
| 2 LeWM, others blue (identical) | 22% | 1 / 35 |
| LeWM + random, identical | 20% | 1 / 36 |

What this shows:
- **No cooperation emerges.** Independent planners push together in at most 4% of steps. They win at most 3 episodes the single agent lost, which is within the noise above.
- **Extra agents lose by physical interference.** With per-agent goals, every episode lost relative to one agent involved another agent touching the T or bumping agent 0. When the others stay clear, success is 100% (31/31 episodes with 2 agents, 14/14 with 3).
- **A shared goal image makes agents compete.** When every agent's goal shows one disk at agent 0's spot, all of them head there: agents bump each other in 18 of the 19 lost episodes with orange others (13 of 18 with others hidden).
- **Identical-looking agents break planning.** With both agents drawn blue, 2 LeWMs do no better than LeWM with a random partner (22% vs 20%).

### Collecting multi-agent data

`collect.py` writes `MultiPushT` rollouts in the layout of `pusht_expert_train.h5`, so `python train.py data=multipusht` runs unchanged. Each env episode (a *scene*) becomes N episodes, one per agent, each from that agent's egocentric view. `action[t]` is applied after observation `t`, and the last row is NaN. The extra columns `scene_idx`, `agent_idx`, `global_state` (Markov-game state), `joint_action`, `block_contact` and `agent_contact` are not loaded by `train.py`.

```bash
python collect.py                                # 1000 scenes, 2 agents, heuristic -> datasets/multipusht_2a_heuristic.h5
python collect.py env.n_agents=3 policy=random   # -> datasets/multipusht_3a_random.h5
```

`policy=heuristic` gives every agent its own noisy scripted pusher. The pusher picks a point on the T's outer boundary, moves just outside it, and pushes into that face for 5–25 steps. One segment in five is instead a walk to a random spot. The agents act independently and do not aim for the goal, and their step size matches the expert's (mean |a| 0.14 vs 0.15). `multipusht_2a_heuristic.h5` (default config, seed 0) holds 1000 scenes, 2000 per-agent episodes and 401,812 frames, about 17% of the frames in the official Push-T data. It takes about 1.7 GB and 20 minutes to collect. Each agent touches the T in 56% of steps (the expert: 39%), both touch it at once in 28%, and the agents bump each other in 12%. The T moves a median of 233 px per scene, and one scene reached the goal by chance. `train.py`'s loader yields 363,812 windows (history 3 + 1 prediction, frameskip 5).

The output is HDF5 in the layout of the official LeWM files. Pixels are Blosc-compressed (lz4, level 5, byte shuffle) in 100-frame chunks, which is lossless and takes about 4 KB per frame instead of 150 KB raw. Other columns are uncompressed, in 1000-row chunks. The small file matters for training speed: the raw 60 GB file does not fit in the page cache, and training became disk-bound at 0.5 it/s on a 240 MB/s disk. Lance is not used because it JPEG-encodes pixels, which adds the kind of render mismatch measured above.

### Training on multi-agent data

```bash
sbatch scripts/train_slurm.sh data=multipusht output_model_name=lewm_mpt2a_heuristic subdir=lewm_mpt2a_heuristic
```

This runs the unchanged LeWM recipe (random-init ViT-tiny, end-to-end, prediction loss + SIGReg, 100 epochs) on the full A100 of the `gpu` partition. It uses 12 loader workers, `prefetch_factor=1` instead of LeWM's 3, and non-persistent workers. These change only how batches are queued, not the results. The run peaks at 13 GB of VRAM and 26 GB of RAM, at the switch from training to validation. Preprocessed float32 batches of about 308 MB each sit in shared memory, which Slurm counts against `--mem`: with prefetch 3 training alone takes 33 GB, and with persistent workers the train and val worker pools are both alive at the switch, which was OOM-killed at 24 GB. It runs at 5 it/s, about 9 minutes per epoch and roughly 15 hours in total.

A crashed step is retried inside the same allocation: `train.py` resumes from the newest `last.ckpt` under `checkpoints/<subdir>/spt/` (stable-pretraining's run cache, kept per run), and a resumed run keeps the validation split it started with. Temp files go to `tmp/<job id>/` in the repo, not `/tmp`. `eval_multi.py` (`eval.scalers=train`) z-scores each policy's actions with the statistics of the data it was trained on, read from the `config.yaml` next to its checkpoint.

**Transfer from the single-agent LeWM.** LeWM's ViT is not an off-the-shelf backbone: it is trained from scratch together with the predictor. Transfer therefore starts from the whole official Push-T checkpoint. `init.ckpt` loads it, `init.freeze` keeps parts fixed (no gradient, eval mode, so the projector's BatchNorm keeps its source statistics), and `scalers_from` keeps the source model's action scaling. The expert data's action std is 0.208 and the heuristic data's is 0.164, so refitting the scalers would feed the pretrained action encoder actions 1.27× too large.

```bash
# predictor-only: frozen single-agent encoder + projector, train predictor, action encoder, pred_proj
sbatch scripts/train_slurm.sh trainer.max_epochs=30 init.ckpt=lewm/pusht/weights.pt 'init.freeze=[encoder,projector]' \
    scalers_from=pusht_expert_train.h5 output_model_name=lewm_mpt2a_ftpred subdir=lewm_mpt2a_ftpred
# full fine-tune from the same init
sbatch scripts/train_slurm.sh trainer.max_epochs=30 init.ckpt=lewm/pusht/weights.pt \
    scalers_from=pusht_expert_train.h5 output_model_name=lewm_mpt2a_ftfull subdir=lewm_mpt2a_ftfull
```

## Cooperative Push-T: force threshold

In the settings above, one agent can always do the whole task. The official physics is quasi-static: swm sets `space.damping = 0`, which in pymunk removes all velocity every step, so the T moves only while pushed and stops when contact ends. But it has no friction threshold, so any push moves it, and the agent is kinematic (here: 1000× heavier than the T) and cannot be stopped. Extra agents can only interfere. With `agent_force` set, the T is too heavy for one agent, so success requires two agents pushing together.

- **Agents.** Each agent is a light body (mass 1, like the T), pulled by a pivot joint of at most `agent_force` toward an invisible kinematic drive body. The drive body runs the original PD controller. In free space an agent stays within 2 px of the original motion. Against resistance it pushes with exactly `agent_force`, however small the action: a PD force would scale with the action, so small, expert-sized actions would push weakly.
- **T.** The T gets top-down floor friction from a pivot joint and a gear joint to the static body, with no position correction. It slides only under a net force above `block_friction` × `agent_force` (1.5) and turns only under a net torque above `block_torque_friction` × `agent_force` × its largest lever arm (1.1 × 76.5 px). One agent's torque is at most about 75 × `agent_force`, so with both ratios in (1, 2), one agent can neither slide nor turn the T, and two can.
- **Solver.** `solver_iterations=50`: with pymunk's default of 10, the agent → T → friction chain does not converge, and a lone push leaks through at up to 1 px per 30 steps.

Physics checks, with the T in the middle of the arena:

| test | T motion |
|---|---|
| 1 agent pushes a face for 200 steps | 0.3 px, 0.2° |
| 1 agent hammers the T for 200 steps (4 steps in, 4 out, \|a\| = 1) | 0.0 px |
| 1 agent at the bar tip or the stem end, \|a\| = 1, 30 steps | 0.0 px, 0.0° |
| 2 agents push the same face, \|a\| = 0.3, 30 steps | 128 px |
| 2 agents push the bar tips in opposite directions, 30 steps | 24 px, 32° |

```bash
python eval_multi.py env.n_agents=1 env.agent_force=1e5 env.solver_iterations=50
python collect.py env.agent_force=1e5 env.solver_iterations=50 policy=coop   # scripted team, see below
python scripts/coop_oracle.py                                                # scripted team on the eval episodes
```

`policy=coop` (`collect.CoopPolicy`) is a scripted team that plans with the same thresholds. It works in four steps:

1. List every contact spot on the T's outer boundary where an agent fits.
2. Score each pair of spots by the part of the two pushes' wrench that exceeds the friction thresholds, compared with the needed translation and rotation, minus a walking cost.
3. Walk both agents around the T and each other to their spots (8 px grid, wavefront).
4. Push together for 5 steps, then replan.

Once the T is placed, every agent walks to its goal spot. The team uses privileged state, so it serves as a solvability check and a data source, not as a baseline.

Results on the 50 episodes of the eval protocol. In 14 of them the T already starts within the success tolerance, so the success test only needs agent 0 to walk to its spot. *T must move* counts the other 36:

| run | budget | success | T must move |
|---|---|---|---|
| 1 LeWM (official checkpoint) | 50 | 24% | **0%** |
| 2 independent LeWMs | 50 | 24% | **0%** |
| scripted team (`coop_oracle.py`) | 50 / 100 / 150 / 300 | 38 / 70 / 80 / 90% | 17 / 58 / 72 / 86% |

Every LeWM success comes from an episode where the T starts in place. The two LeWMs touch the T together in 4% of steps, never at the right moment. Cooperating needs more steps than LeWM's 50-step budget, since both agents must first walk to their spots, so evals in this setting should use `eval.eval_budget=150`.

## Roadmap

| Level | Setting | Question |
|---|---|---|
| 0 | Push-T, 1 agent | Can the WM learn contact dynamics? |
| 1 | MultiPush-T, 2 agents, cooperative | Does N=1 physics transfer compositionally to N=2? |
| 2 | 2v2 | Can the WM support cooperative/competitive planning? |
| 3 | + movable tools | Does self-play produce emergent strategies? |
