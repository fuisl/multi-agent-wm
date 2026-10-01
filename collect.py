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


class CoopPolicy:
    """Scripted team for the force-threshold env (world.PushTN(agent_force=...)): goal-directed
    joint pushes by agents 0 and 1; further agents stay still.

    Every agent calls it; agent 0's call plans the joint action for the step. Planning mirrors the
    physics: each candidate pair of contact spots on the T's outer boundary gives a wrench (two
    pushes of agent_force along the inward normals); only the part above the T's floor friction
    moves it. The pair whose excess wrench best matches the needed translation / rotation (minus a
    walking cost) is chosen, both agents walk around the T to their spots, push together for a few
    steps, and the team replans. Once the T is within tolerance, every agent walks to its goal spot.

    For data collection, p_solo > 0 makes each agent break away now and then: for solo_steps it
    follows its own HeuristicPolicy, so the data also shows lone pushes that do not move the T
    (and a partner pushing alone). Actions get Gaussian noise, then EMA smoothing, as in
    HeuristicPolicy.
    """

    joint = True

    def __init__(self, rng, speed=0.3, push_speed=0.25, push_steps=5, standoff=6.0, spacing=8.0,
                 noise=0.0, smooth=0.0, walk_cost=0.3, place_tol=(10.0, np.radians(8)),
                 p_solo=0.0, solo_steps=(5, 25), team_steps=(20, 60), **kwargs):
        self.rng = rng
        self.speed, self.push_speed, self.push_steps = speed, push_speed, push_steps
        self.standoff, self.spacing, self.noise, self.walk_cost = standoff, spacing, noise, walk_cost
        self.smooth, self.place_tol = smooth, place_tol
        self.p_solo, self.solo_steps, self.team_steps = p_solo, solo_steps, team_steps

    def reset(self, core, i):
        if i:
            return
        self.scale = core.action_scale
        self.radius = max(s.radius for s in core.agents[0].shapes)
        self.contacts = self._contacts(core)  # (K, 2) points, (K, 2) outward normals, block frame
        self.plan, self.mode, self.t = None, 'approach', 0
        self.tabu = {}
        self.actions = np.zeros((core.n_agents, 2))
        self.prev = np.zeros((core.n_agents, 2))
        self.step = 0
        self.solo = [HeuristicPolicy(self.rng) for _ in range(core.n_agents)]
        self.solo_until = np.zeros(core.n_agents, dtype=int)  # step until which agent k is solo
        self.team_until = np.array([self._duration(self.team_steps) for _ in range(core.n_agents)])

    def _duration(self, steps):
        return self.step + int(self.rng.integers(*steps))

    def _update_solo(self, core, acts):
        """Replace the team action of agents currently on a solo segment."""
        for k in range(core.n_agents):
            if self.step >= max(self.solo_until[k], self.team_until[k]):  # segment over: draw the next
                if self.rng.random() < self.p_solo:
                    self.solo_until[k] = self._duration(self.solo_steps)
                    self.solo[k].reset(core, k)
                else:
                    self.team_until[k] = self._duration(self.team_steps)
            if self.step < self.solo_until[k]:
                acts[k] = self.solo[k](core, k)
        return acts

    def _contacts(self, core):
        """Spots on the outer boundary of the T where an agent can stand and push (block frame)."""
        pts, nrm = [], []
        polys = [s for s in core.block.shapes if isinstance(s, pymunk.Poly)]
        for shape in polys:
            v = [np.array(p) for p in shape.get_vertices()]
            centre = np.mean(v, axis=0)
            for k in range(len(v)):
                a, b = v[k], v[(k + 1) % len(v)]
                n = np.array([b[1] - a[1], a[0] - b[0]])
                n /= np.linalg.norm(n)
                if np.dot(n, (a + b) / 2 - centre) < 0:
                    n = -n
                for f in np.arange(0, 1, self.spacing / np.linalg.norm(b - a)) + self.spacing / np.linalg.norm(b - a) / 2:
                    p = a + f * (b - a)
                    probe = core.block.local_to_world(tuple(p + 2 * n))
                    spot = core.block.local_to_world(tuple(p + n * (self.radius + 1)))
                    if all(s.point_query(probe).distance > 0 for s in polys) and \
                            all(s.point_query(spot).distance > self.radius - 0.5 for s in polys):
                        pts.append(p), nrm.append(n)
        return np.array(pts), np.array(nrm)

    def _errors(self, core):
        N = core.n_agents
        g, s = core.goal_state, core._get_obs()
        e = g[2 * N:2 * N + 2] - s[2 * N:2 * N + 2]
        dth = (g[2 * N + 2] - s[2 * N + 2] + np.pi) % (2 * np.pi) - np.pi
        return e, dth

    def _choose(self, core):
        """Best pair of contact spots (indices) for the current T error, or None."""
        e, dth = self._errors(core)
        rot = pymunk.Vec2d(1, 0).rotated(-core.block.angle)
        R = np.array([[rot.x, -rot.y], [rot.y, rot.x]])
        e_loc = R @ e  # T error in the block frame
        pts, nrm = self.contacts
        cog = np.array(core.block.center_of_gravity)
        d = -nrm
        tau = np.cross(pts - cog, d)
        lever = core.block_lever
        F = d[:, None, :] + d[None, :, :]
        T = tau[:, None] + tau[None, :]
        lin = np.linalg.norm(F, axis=-1)
        ex_lin = np.clip(lin - core.block_friction, 0, None)
        ex_rot = np.clip(np.abs(T) - core.block_torque_friction * lever, 0, None) / lever
        u = F / np.maximum(lin, 1e-6)[..., None]
        e_norm = np.linalg.norm(e)
        w_t = min(1.0, e_norm / 40)
        w_r = min(1.0, abs(dth) / np.radians(25))
        score = ex_lin * w_t * (u @ (e_loc / max(e_norm, 1e-6))) + ex_rot * np.sign(T) * np.sign(dth) * w_r
        score -= 0.3 * ex_rot * (1 - w_r)  # unwanted turning when the angle is already right
        # walking cost: best assignment of agents 0 / 1 to the two spots
        spots = np.array([core.block.local_to_world(tuple(p + n * (self.radius + self.standoff))) for p, n in zip(pts, nrm)])
        a0, a1 = (np.array(core.agents[k].position) for k in (0, 1))
        d0, d1 = np.linalg.norm(spots - a0, axis=1), np.linalg.norm(spots - a1, axis=1)
        walk = np.minimum(d0[:, None] + d1[None, :], d1[:, None] + d0[None, :])
        score -= self.walk_cost * walk / 500
        sep = np.linalg.norm(spots[:, None] - spots[None, :], axis=-1)
        score[sep < 2 * self.radius + 4] = -np.inf  # the two agents must fit side by side
        for (i, j), until in self.tabu.items():
            if self.step < until:
                score[i, j] = score[j, i] = -np.inf
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if not np.isfinite(score[i, j]) or score[i, j] <= 0:
            return None
        # agent 0 takes spot i or j, whichever is cheaper overall
        return (i, j) if d0[i] + d1[j] <= d1[i] + d0[j] else (j, i)

    def _toward(self, pos, target, speed):
        d = (target - pos) / self.scale
        norm = np.linalg.norm(d)
        return d if norm <= speed else d * speed / norm

    CELL = 8  # px, navigation grid
    GRID = (np.arange(512 // CELL) + 0.5) * CELL

    def _clearance(self, core, pts):
        """Distance from world points (..., 2) to the T (exact for its two rectangles)."""
        rel = pts - np.array(core.block.position)
        c, s = np.cos(-core.block.angle), np.sin(-core.block.angle)
        loc = np.stack([c * rel[..., 0] - s * rel[..., 1], s * rel[..., 0] + c * rel[..., 1]], -1)
        dist = np.inf
        for shape in core.block.shapes:
            v = np.array([tuple(p) for p in shape.get_vertices()])
            lo, hi = v.min(0), v.max(0)
            q = np.maximum(np.maximum(lo - loc, loc - hi), 0)
            dist = np.minimum(dist, np.linalg.norm(q, axis=-1))
        return dist

    def _walk(self, core, k, target):
        """Action for agent k toward target around the T and the other agents (grid wavefront)."""
        pos = np.array(core.agents[k].position)
        others = [np.array(b.position) for j, b in enumerate(core.agents) if j != k]
        seg = np.linspace(pos, target, 12)
        if (self._clearance(core, seg) > self.radius + 1).all() and \
                all((np.linalg.norm(seg - o, axis=1) > 2 * self.radius + 1).all() for o in others):
            return self._toward(pos, target, self.speed)

        gx, gy = np.meshgrid(self.GRID, self.GRID, indexing='ij')
        cells = np.stack([gx, gy], -1)
        free = self._clearance(core, cells) > self.radius + 4
        for o in others:
            free &= np.linalg.norm(cells - o, axis=-1) > 2 * self.radius + 2
        free &= (cells > self.radius + 8).all(-1) & (cells < 504 - self.radius).all(-1)
        near = lambda p, r: np.linalg.norm(cells - p, axis=-1) < r
        free |= near(target, 12) | near(pos, 12)  # goal spots may touch the T; the agent may already

        dist = np.full(free.shape, np.inf)
        dist[near(target, 6) & free] = 0
        moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.41), (-1, 1, 1.41), (1, -1, 1.41), (1, 1, 1.41)]
        start = tuple(np.clip((pos // self.CELL).astype(int), 0, len(self.GRID) - 1))
        for _ in range(4 * len(self.GRID)):
            new = dist.copy()
            for dx, dy, w in moves:
                shifted = np.full(dist.shape, np.inf)
                shifted[max(dx, 0):dist.shape[0] + min(dx, 0), max(dy, 0):dist.shape[1] + min(dy, 0)] = \
                    dist[max(-dx, 0):dist.shape[0] + min(-dx, 0), max(-dy, 0):dist.shape[1] + min(-dy, 0)] + w
                new = np.minimum(new, shifted)
            new[~free] = np.inf
            if np.array_equal(new, dist) or np.isfinite(new[start]):
                dist = new
                break
            dist = new
        if not np.isfinite(dist[start]):
            return self._toward(pos, target, self.speed)
        cell = start  # follow the distance field a few cells ahead
        for _ in range(3):
            nxt = min(((cell[0] + dx, cell[1] + dy) for dx, dy, _ in moves
                       if 0 <= cell[0] + dx < dist.shape[0] and 0 <= cell[1] + dy < dist.shape[1]),
                      key=lambda c: dist[c])
            if dist[nxt] >= dist[cell]:
                break
            cell = nxt
        return self._toward(pos, cells[cell], self.speed)

    def _plan_step(self, core):
        N = core.n_agents
        acts = np.zeros((N, 2))
        e, dth = self._errors(core)
        placed = np.linalg.norm(e) < self.place_tol[0] and abs(dth) < self.place_tol[1]
        if self.plan is None and not placed:
            self.plan, self.mode, self.t = self._choose(core), 'approach', 0
            self.err0 = (np.linalg.norm(e), abs(dth))
            # no pair helps any more: close enough, finish (within the success tolerance of 20 px / 20 deg)
            placed = self.plan is None and np.linalg.norm(e) < 20 and abs(dth) < np.radians(20)
        if placed:
            self.plan = None
            for k in range(N):
                acts[k] = self._walk(core, k, np.array(core.goal_state[2 * k:2 * k + 2]))
            return acts
        if self.plan is None:  # nothing useful to push: wait
            return acts

        pts, nrm = self.contacts
        spots, pushes = [], []
        for k, c in enumerate(self.plan):
            spots.append(np.array(core.block.local_to_world(tuple(pts[c] + nrm[c] * (self.radius + self.standoff)))))
            pushes.append(np.array(core.block.local_to_world(tuple(pts[c] - nrm[c] * 30))))
        pos = [np.array(core.agents[k].position) for k in (0, 1)]

        if self.mode == 'approach':
            ready = [np.linalg.norm(spots[k] - pos[k]) < 5 for k in (0, 1)]
            for k in (0, 1):
                acts[k] = np.zeros(2) if ready[k] else self._walk(core, k, spots[k])
            self.t += 1
            if all(ready):
                self.mode, self.t = 'push', 0
            elif self.t > 30:  # stuck on the way: try another pair for a while
                self.tabu[tuple(sorted(self.plan))] = self.step + 30
                self.plan = None
            return acts

        # push together; slower when the T is close to its goal
        closeness = max(np.linalg.norm(e) / 60, abs(dth) / np.radians(30))
        speed = float(np.clip(self.push_speed * closeness, 0.08, self.push_speed))
        for k in (0, 1):
            acts[k] = self._toward(pos[k], pushes[k], speed)
        self.t += 1
        if self.t >= self.push_steps:
            err = (np.linalg.norm(e), abs(dth))
            if err[0] > self.err0[0] - 2 and err[1] > self.err0[1] - np.radians(2):  # no progress
                self.tabu[tuple(sorted(self.plan))] = self.step + 30
            self.plan = None
        return acts

    def __call__(self, core, i):
        if i == 0:
            acts = self._plan_step(core)
            if self.p_solo:
                acts = self._update_solo(core, acts)
            if self.noise:
                acts = acts + self.rng.normal(0, self.noise, acts.shape)
            self.actions = self.prev = self.smooth * self.prev + (1 - self.smooth) * acts
            self.step += 1
        return np.clip(self.actions[i], -1, 1)


POLICIES = {'random': RandomPolicy, 'heuristic': HeuristicPolicy, 'coop': CoopPolicy}


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
    policy_cls = POLICIES[cfg.policy]
    if getattr(policy_cls, 'joint', False):  # one team planner shared by all agents
        policies = [policy_cls(np.random.default_rng(cfg.seed), **kwargs)] * N
    else:
        policies = [policy_cls(np.random.default_rng([cfg.seed, i]), **kwargs) for i in range(N)]

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
