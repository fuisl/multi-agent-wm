"""Roll out a policy in multi-agent Push-T and write a dataset that train.py reads unchanged.

Every env episode becomes N episodes in the single-agent Push-T layout, one per agent, each
seen from that agent's egocentric view:

    pixels (224, 224, 3) uint8, proprio (4,), state (7,), action (2,)   as pusht_expert_train.h5
    episode_idx, step_idx                                              one id per (scene, agent)
    scene_idx, agent_idx                                               env episode / which agent
    global_state (4N + 3,), joint_action (2N,)                         Markov-game state / joint action
    block_contact, agent_contact                                       contacts of this agent

action[t] is the action applied after observation t (as in the official dataset); the last row
of an episode has no action and holds NaN (train.py maps NaN actions to 0). Same for joint_action.

HDF5 is the default, in the layout of the official LeWM files: pixels losslessly Blosc-compressed
(lz4, level 5, byte shuffle) in 100-frame chunks, other columns uncompressed in 1000-row chunks.
Lance JPEG-encodes pixels, and render-vs-dataset pixel differences already cost episodes (see README).
"""

import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))  # ./data -> big disk, see scripts/setup_storage.sh

import hdf5plugin
import hydra
import numpy as np
import pymunk
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from world import MultiPushT


class BloscHDF5Writer(swm.data.formats.hdf5.HDF5Writer):
    """swm's HDF5Writer with the chunking and compression of the official LeWM datasets."""

    def _init_schema(self, sample_ep):
        for col, vals in sample_ep.items():
            sample = np.asarray(vals[0])
            image = col.startswith('pixels')
            self._f.create_dataset(
                col, shape=(0, *sample.shape), maxshape=(None, *sample.shape), dtype=sample.dtype,
                chunks=(100 if image else 1000, *sample.shape),
                **(hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE) if image else {}),
            )
        for col, dtype in (('ep_len', np.int32), ('ep_offset', np.int64)):
            self._f.create_dataset(col, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(1024,))


class RandomPolicy:
    """Uniform actions in [-1, 1]^2, as swm.policy.RandomPolicy."""

    def __init__(self, rng, **kwargs):
        self.rng = rng

    def reset(self, core, i):
        pass

    def __call__(self, core, i):
        return self.rng.uniform(-1, 1, size=2)


class HeuristicPolicy:
    """Noisy scripted pusher for one agent.

    Repeats segments: pick a point on the T's outer boundary, move to a spot just outside it,
    then push into that face (the contact point is fixed in the block frame, so the push
    follows the block as it turns) for a random number of steps. With probability p_wander a
    segment is a walk to a random spot instead, which also produces agent-agent contacts.
    Actions are EMA-smoothed with Gaussian noise; the step size matches the expert data
    (mean |a| ~ 0.15).
    """

    def __init__(self, rng, speed=0.25, push_speed=0.2, noise=0.05, smooth=0.5,
                 push_steps=(5, 25), p_wander=0.2, standoff=12.0, **kwargs):
        self.rng = rng
        self.speed, self.push_speed = speed, push_speed
        self.noise, self.smooth = noise, smooth
        self.push_steps, self.p_wander, self.standoff = push_steps, p_wander, standoff

    def reset(self, core, i):
        self.scale = core.action_scale  # relative actions are in units of 100 px
        self.prev = np.zeros(2)
        self._new_segment(core, i)

    def _new_segment(self, core, i):
        self.t = 0
        if self.rng.random() < self.p_wander:
            self.mode, self.target = 'wander', self.rng.uniform(50, 462, size=2)
            self.steps = int(self.rng.integers(*self.push_steps))
            return
        self.mode = 'approach'
        self.contact, self.normal = self._sample_face(core)
        self.steps = int(self.rng.integers(*self.push_steps))

    def _sample_face(self, core):
        """(contact point, outward normal) in the block frame, on the outer boundary of the T."""
        shapes = [s for s in core.block.shapes if isinstance(s, pymunk.Poly)]
        for _ in range(100):
            shape = shapes[self.rng.integers(len(shapes))]
            v = [np.array(p) for p in shape.get_vertices()]
            k = self.rng.integers(len(v))
            a, b = v[k], v[(k + 1) % len(v)]
            p = a + self.rng.uniform(0.1, 0.9) * (b - a)
            n = np.array([b[1] - a[1], a[0] - b[0]])
            n /= np.linalg.norm(n)
            if np.dot(n, p - np.mean(v, axis=0)) < 0:
                n = -n
            probe = core.block.local_to_world(tuple(p + 2 * n))  # skip faces shared by the two bars
            if all(s.point_query(probe).distance > 0 for s in core.block.shapes):
                return p, n
        return p, n

    def _toward(self, pos, target, speed):
        d = (target - pos) / self.scale
        norm = np.linalg.norm(d)
        return d if norm <= speed else d * speed / norm

    def __call__(self, core, i):
        body = core.agents[i]
        pos = np.array(body.position)
        radius = max(s.radius for s in body.shapes)
        contact = np.array(core.block.local_to_world(tuple(self.contact))) if self.mode != 'wander' else None
        normal = np.array(pymunk.Vec2d(*self.normal).rotated(core.block.angle)) if self.mode != 'wander' else None

        if self.mode == 'wander':
            a = self._toward(pos, self.target, self.speed)
            self.t += 1
        elif self.mode == 'approach':
            spot = contact + normal * (radius + self.standoff)
            a = self._toward(pos, spot, self.speed)
            if np.linalg.norm(spot - pos) < 8:
                self.mode = 'push'
            self.t += 0.25  # give up approaching after ~4x the push budget
        else:
            a = self._toward(pos, contact - normal * 30, self.push_speed)
            self.t += 1
            if np.linalg.norm(contact - pos) > radius + 3 * self.standoff:
                self.t = self.steps  # lost the face
        if self.t >= self.steps:
            self._new_segment(core, i)

        a = a + self.rng.normal(0, self.noise, size=2)
        a = self.smooth * self.prev + (1 - self.smooth) * a
        self.prev = a
        return np.clip(a, -1, 1)


POLICIES = {'random': RandomPolicy, 'heuristic': HeuristicPolicy}


def rollout(env, policies, seed):
    """One env episode -> list of N per-agent episodes (dicts of per-step lists)."""
    obs, infos = env.reset(seed=seed)
    core = env.core
    for i, p in enumerate(policies):
        p.reset(core, i)
    eps = [{k: [] for k in ('pixels', 'proprio', 'state', 'action', 'global_state', 'joint_action',
                            'block_contact', 'agent_contact')} for _ in policies]

    def record(obs, infos, state):
        for ep, ag in zip(eps, env.possible_agents):
            ep['pixels'].append(obs[ag]['pixels'])
            ep['proprio'].append(obs[ag]['proprio'].astype(np.float32))
            ep['state'].append(obs[ag]['state'].astype(np.float32))
            ep['global_state'].append(state.astype(np.float32))
            ep['block_contact'].append(np.int8(infos[ag]['block_contact']))
            ep['agent_contact'].append(np.int8(infos[ag]['agent_contact']))

    record(obs, infos, env.state())
    while env.agents:
        actions = {ag: np.asarray(p(core, i), dtype=np.float32)
                   for i, (ag, p) in enumerate(zip(env.possible_agents, policies))}
        joint = np.concatenate([actions[ag] for ag in env.possible_agents])
        for ep, ag in zip(eps, env.possible_agents):
            ep['action'].append(actions[ag])
            ep['joint_action'].append(joint)
        obs, _, _, _, infos = env.step(actions)
        record(obs, infos, env.state())
    for ep in eps:
        ep['action'].append(np.full(2, np.nan, dtype=np.float32))
        ep['joint_action'].append(np.full(2 * len(eps), np.nan, dtype=np.float32))
    return eps


@hydra.main(version_base=None, config_path="./config/collect", config_name="multipusht")
def run(cfg: DictConfig):
    """Roll out cfg.policy in MultiPushT(**cfg.env) and write $STABLEWM_HOME/datasets/<output.name>."""
    env = MultiPushT(**OmegaConf.to_container(cfg.env, resolve=True))
    N = env.core.n_agents
    rng = np.random.default_rng(cfg.seed)
    kwargs = OmegaConf.to_container(cfg.policy_kwargs, resolve=True)
    policies = [POLICIES[cfg.policy](np.random.default_rng([cfg.seed, i]), **kwargs) for i in range(N)]

    path = Path(os.environ['STABLEWM_HOME']) / 'datasets' / cfg.output.name
    stats = {'steps': 0, 'success': 0, 'block_contact': np.zeros(N), 'agent_contact': np.zeros(N)}
    if cfg.output.format == 'hdf5':
        writer = BloscHDF5Writer(path, mode=cfg.output.mode)
    else:
        writer = swm.data.get_format(cfg.output.format).open_writer(path, mode=cfg.output.mode)
    with writer as w:
        for scene in tqdm(range(cfg.num_episodes), desc='Recording'):
            eps = rollout(env, policies, seed=int(rng.integers(2**31)))
            T = len(eps[0]['action'])
            for i, ep in enumerate(eps):
                ep['episode_idx'] = [np.int64(scene * N + i)] * T
                ep['step_idx'] = list(np.arange(T, dtype=np.int64))
                ep['scene_idx'] = [np.int64(scene)] * T
                ep['agent_idx'] = [np.int64(i)] * T
                stats['block_contact'][i] += np.mean(ep['block_contact'])
                stats['agent_contact'][i] += np.mean(ep['agent_contact'])
                w.write_episode(ep)
            stats['steps'] += T - 1
            stats['success'] += T - 1 < env.max_episode_steps

    n = cfg.num_episodes
    print(f"wrote {path}: {n} scenes x {N} agents, {stats['steps']} env steps, success {stats['success'] / n:.1%}")
    print(f"block contact per agent {np.round(stats['block_contact'] / n, 3).tolist()}, "
          f"agent contact per agent {np.round(stats['agent_contact'] / n, 3).tolist()}")


if __name__ == "__main__":
    run()
