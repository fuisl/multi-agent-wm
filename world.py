"""Multi-agent Push-T: N agents push one T-block, all acting at once (PettingZoo ParallelEnv).

Built on stable-worldmodel's single-agent PushT: same physics constants, T-block, walls,
goal and rendering, so a single-agent LeWM checkpoint can be plugged in for every agent.

Per-agent observation (same keys / shapes as single-agent Push-T):
    pixels  (224, 224, 3) uint8   this agent's egocentric view (see below)
    proprio (4,)  [x, y, vx, vy]                                   of this agent
    state   (7,)  [x, y, block_x, block_y, block_angle, vx, vy]    of this agent + block
    goal    (224, 224, 3) uint8   in infos: T at the goal pose + this agent at its goal position

Global state, env.state():
    [agent_xy * N, block_xy, block_angle, agent_vxvy * N]    (N=1: the original 7-d layout)

Egocentric rendering: shape = entity type, color = role relative to the viewer.
    self            circle, blue
    other agents    circle, orange (others='distinct'); same orange for every other agent,
                    so it means "another agent", not a fixed identity
    T-block / goal  gray / green, as in the original
others='visible' draws other agents like self (blue), others='hidden' leaves them out.
The audit render (env.render()) is a global view: one identity color + index per agent.

agent_collisions -- the original agent is a kinematic body, and pymunk never collides two
kinematic bodies, so agents would pass through each other. With agent_collisions=True every
agent becomes a heavy dynamic body whose velocity is still set by the PD controller.

agent_force -- cooperation by force threshold (None: original physics, where one agent moves
the T at will). Every agent becomes a light body (mass 1, like the T) pulled by a pivot joint
of at most agent_force toward an invisible kinematic drive body, which runs the original PD
controller. In free space the agent follows the original motion; against resistance it
pushes with exactly agent_force, however small the action. The T gets top-down floor
friction (pivot + gear joint to the static body, no position correction): it slides only
under a net force above block_friction x agent_force and turns only under a net torque above
block_torque_friction x agent_force x its largest lever arm. With both ratios in (1, 2), one
agent alone can neither slide nor turn the T, and two agents pushing together can.
"""

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pymunk.vec2d import Vec2d

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.pusht.env import DEFAULT_VARIATIONS, PushT
from stable_worldmodel.envs.utils import DrawOptions

# identity colors for the audit view: none of them is blue / orange (self / other) or gray / green (T / goal)
AUDIT_COLORS = ['MediumOrchid', 'Crimson', 'Gold', 'DeepPink', 'Sienna', 'DarkCyan']
OTHER_COLOR = 'DarkOrange'  # others='distinct'
OTHERS_MODES = ('visible', 'distinct', 'hidden')


class _DrawOptions(DrawOptions):
    """Skips circles whose color has alpha 0: hides agents without touching the physics."""

    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        if fill_color.a > 0:
            super().draw_circle(pos, angle, radius, outline_color, fill_color)


class PushTN(PushT):
    """swm PushT with N agents. Agent 0 is the original `self.agent`."""

    def __init__(
        self,
        n_agents=2,
        others='distinct',
        agent_collisions=True,
        agent_mass=1000.0,
        agent_force=None,
        block_friction=1.5,
        block_torque_friction=1.1,
        solver_iterations=None,
        success='pusht',
        audit_resolution=512,
        **kwargs,
    ):
        assert others in OTHERS_MODES, f'others must be one of {OTHERS_MODES}'
        assert success in ('block', 'pusht')
        assert agent_force is None or agent_collisions, 'agent_force needs dynamic agents (agent_collisions=True)'
        super().__init__(**kwargs)
        self.n_agents = n_agents
        self.others = others
        self.agent_collisions = agent_collisions
        self.agent_mass = agent_mass
        self.agent_force = agent_force
        self.block_friction = block_friction
        self.block_torque_friction = block_torque_friction
        self.solver_iterations = solver_iterations
        self.success = success
        self.audit_size = audit_resolution
        self.env_name = 'MultiPushT'

    ###########
    # physics #
    ###########

    def _setup(self):
        super()._setup()
        agent = self.variation_space['agent']
        params = {
            'position': agent['start_position'].value.tolist(),
            'angle': agent['angle'].value,
            'scale': agent['scale'].value,
            'color': agent['color'].value.tolist(),
            'shape': self.shapes[agent['shape'].value],
        }
        self.agents = [self.agent] + [self.add_shape(**params) for _ in range(self.n_agents - 1)]

        if self.agent_collisions:
            for body in self.agents:
                body.body_type = pymunk.Body.DYNAMIC
                body.mass = self.agent_mass
                body.moment = float('inf')
                body.velocity_func = _keep_velocity  # PD sets it; no damping (space.damping = 0)

        if self.solver_iterations is not None:
            self.space.iterations = self.solver_iterations
        self.drives = []
        if self.agent_force is not None:
            for body in self.agents:
                body.mass = 1.0
                drive = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                drive.position = body.position
                joint = pymunk.PivotJoint(drive, body, (0, 0), (0, 0))
                joint.max_force = self.agent_force
                self.space.add(drive, joint)
                self.drives.append(drive)

        self._owner = {s: i for i, b in enumerate(self.agents) for s in b.shapes}
        self._owner.update({s: 'block' for s in self.block.shapes})
        self.block_contact = np.zeros(self.n_agents, dtype=bool)
        self.agent_contact = np.zeros(self.n_agents, dtype=bool)

    def _add_floor_friction(self):
        """Top-down Coulomb friction on the T: joints to the static body that only resist velocity."""
        cog = self.block.center_of_gravity
        self.block_lever = max(
            (Vec2d(*v) - cog).length for s in self.block.shapes for v in s.get_vertices()
        )
        slide = pymunk.PivotJoint(self.space.static_body, self.block, (0, 0), cog)
        spin = pymunk.GearJoint(self.space.static_body, self.block, 0.0, 1.0)
        slide.max_force = self.block_friction * self.agent_force
        spin.max_force = self.block_torque_friction * self.agent_force * self.block_lever
        for joint in (slide, spin):
            joint.max_bias = 0  # no position correction: pure friction
            self.space.add(joint)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)
        a, b = (self._owner.get(s) for s in arbiter.shapes)
        for x, y in ((a, b), (b, a)):
            if type(x) is int:
                if y == 'block':
                    self.block_contact[x] = True
                elif type(y) is int:
                    self.agent_contact[x] = True

    def simulate(self, actions):
        """actions: (N, 2) in [-1, 1], one PushT control step for all agents at once."""
        self.n_contact_points = 0
        self.block_contact[:] = False
        self.agent_contact[:] = False
        self.latest_action = actions
        n_steps = int(1 / (self.dt * self.control_hz))

        targets = [
            body.position + a * self.action_scale if self.relative else Vec2d(*a)
            for body, a in zip(self.agents, actions)
        ]
        # with agent_force the PD moves each agent's drive body, which starts from the agent
        drivers = self.drives or self.agents
        self._sync_drives()
        for _ in range(n_steps):
            for body, target in zip(drivers, targets):
                acceleration = self.k_p * (target - body.position) + self.k_v * (Vec2d(0, 0) - body.velocity)
                body.velocity += acceleration * self.dt
            self.space.step(self.dt)
            if self.agent_collisions:
                self._keep_inside()

    def _keep_inside(self, wall_lo=7.0, wall_hi=504.0):
        """Walls for dynamic agents: pymunk lets a heavy PD-driven body sink into the thin wall
        segments, so clamp agent centers inside the arena and drop the outward velocity."""
        for body in self.agents:
            r = max(s.radius for s in body.shapes)
            (x, y), (vx, vy) = body.position, body.velocity
            lo, hi = wall_lo + r, wall_hi - r
            if not (lo <= x <= hi and lo <= y <= hi):
                vx = max(vx, 0.0) if x < lo else min(vx, 0.0) if x > hi else vx
                vy = max(vy, 0.0) if y < lo else min(vy, 0.0) if y > hi else vy
                body.position = (min(max(x, lo), hi), min(max(y, lo), hi))
                body.velocity = (vx, vy)

    def _sync_drives(self):
        for drive, body in zip(self.drives, self.agents):
            drive.position, drive.velocity = body.position, body.velocity

    #########
    # state #
    #########

    def _expand(self, state):
        """7-d single-agent state -> global state (agents 1..N-1 keep their current position)."""
        state = np.asarray(state, dtype=np.float64)
        N = self.n_agents
        if state.shape[0] == 4 * N + 3:
            return state.copy()
        assert state.shape[0] in (5, 7), f'bad state shape {state.shape}'
        pos = np.array([tuple(b.position) for b in self.agents])
        vel = np.zeros((N, 2))
        pos[0] = state[:2]
        if state.shape[0] == 7:
            vel[0] = state[5:7]
        return np.concatenate([pos.ravel(), state[2:5], vel.ravel()])

    def _set_state(self, state):
        state = self._expand(state)
        N = self.n_agents
        pos, vel = state[: 2 * N].reshape(N, 2), state[2 * N + 3 :].reshape(N, 2)
        for body, p, v in zip(self.agents, pos, vel):
            body.velocity = tuple(v)
            body.position = tuple(p)
        self.block.angle = state[2 * N + 2]
        self.block.position = tuple(state[2 * N : 2 * N + 2])
        self._sync_drives()
        self.space.step(self.dt)  # run physics to take effect

    def _set_goal_state(self, goal_state):
        self.goal_state = self._expand(goal_state)

    def _get_obs(self):
        """Global state (see module docstring)."""
        pos = [c for b in self.agents for c in b.position]
        vel = [c for b in self.agents for c in b.velocity]
        block = [*self.block.position, self.block.angle % (2 * np.pi)]
        return np.array(pos + block + vel, dtype=np.float64)

    def agent_obs(self, i, state=None):
        """Per-agent (proprio, state) in the single-agent layout."""
        s = self._get_obs() if state is None else state
        N = self.n_agents
        pos, vel = s[2 * i : 2 * i + 2], s[2 * N + 3 + 2 * i : 2 * N + 5 + 2 * i]
        return np.concatenate([pos, vel]), np.concatenate([pos, s[2 * N : 2 * N + 3], vel])

    def errors(self, goal_state, cur_state):
        """T position error, T angle error, and each agent's distance to its goal position."""
        N = self.n_agents
        block = np.linalg.norm(goal_state[2 * N : 2 * N + 2] - cur_state[2 * N : 2 * N + 2])
        angle = np.abs(goal_state[2 * N + 2] - cur_state[2 * N + 2])
        angle = np.minimum(angle, 2 * np.pi - angle)
        agents = np.linalg.norm((goal_state[: 2 * N] - cur_state[: 2 * N]).reshape(N, 2), axis=1)
        return float(block), float(angle), agents

    def eval_state(self, goal_state, cur_state):
        """Original PushT test ('pusht'): |[agent 0 xy, T xy] - goal| < 20 px and T angle < 20 deg,
        i.e. the T *and* agent 0 at their goal. 'block': T pose only (ends before the agent arrives)."""
        N = self.n_agents
        b = slice(2 * N, 2 * N + 2)
        pos_diff = goal_state[b] - cur_state[b]
        if self.success == 'pusht':
            pos_diff = np.concatenate([goal_state[:2] - cur_state[:2], pos_diff])
        pos_diff = np.linalg.norm(pos_diff)
        angle_diff = np.abs(goal_state[2 * N + 2] - cur_state[2 * N + 2])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        return bool(success), float(np.linalg.norm(goal_state - cur_state))

    def reset(self, seed=None, options=None):
        """options: state / goal_state (7-d or global), agent_starts ((N-1, 2) for agents 1..N-1)."""
        gym.Env.reset(self, seed=seed)
        self.rng = np.random.default_rng(seed)
        options = options or {}
        swm_spaces.reset_variation_space(self.variation_space, seed, options, DEFAULT_VARIATIONS)
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        if self.agent_force is not None:
            self._add_floor_friction()

        var = self.variation_space
        state = options.get('state')
        if state is None:
            state = np.concatenate([
                var['agent']['start_position'].value, var['block']['start_position'].value,
                [var['block']['angle'].value], var['agent']['velocity'].value,
            ])
        goal_state = options.get('goal_state')
        if goal_state is None:
            goal_state = np.concatenate([
                var['agent']['start_position'].sample(set_value=False),
                var['block']['start_position'].sample(set_value=False),
                [var['block']['angle'].sample(set_value=False)],
                var['agent']['velocity'].value,
            ])

        # place agents 1..N-1 (they keep these positions in the goal configuration too)
        state = np.asarray(state, dtype=np.float64)
        starts = options.get('agent_starts')
        if starts is None and state.shape[0] != 4 * self.n_agents + 3:
            starts = self._sample_starts(state[:2], state[2:4])
        for body, p in zip(self.agents[1:], starts if starts is not None else []):
            body.position = tuple(p)

        self._set_state(goal_state)
        self._set_goal_state(goal_state)
        self.goals = [self.render_view(i, others='hidden') for i in range(self.n_agents)]
        self._set_state(state)

    def _sample_starts(self, agent0, block, lo=50, hi=450, block_gap=110, agent_gap=60):
        placed = [np.asarray(agent0)]
        for _ in range(self.n_agents - 1):
            for _ in range(1000):
                p = self.np_random.uniform(lo, hi, size=2)
                if np.linalg.norm(p - block) > block_gap and all(np.linalg.norm(p - q) > agent_gap for q in placed):
                    break
            placed.append(p)
        return np.array(placed[1:])

    #############
    # rendering #
    #############

    def _draw(self, agent_colors, size):
        """Same drawing as PushT._render_frame, with one color (RGB or RGBA) per agent."""
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.variation_space['background']['color'].value)
        draw_options = _DrawOptions(canvas)
        # no joints: pymunk draws them by default (purple dots at the agent_force pivots and the T's
        # friction pivot, also for hidden agents), which leaks physics state into the pixels. Collision
        # points stay, as in the original PushT render (default flags) that LeWM was trained on
        draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES | pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS

        if bool(self.variation_space['rendering']['render_goal'].value) and self.with_target:
            goal_color = self.variation_space['goal']['color'].value
            goal_body = self._get_goal_pose_body(self.goal_pose)
            for shape in self.block.shapes:
                if isinstance(shape, pymunk.Circle):
                    c = pymunk.pygame_util.to_pygame(goal_body.local_to_world(shape.offset), canvas)
                    pygame.draw.circle(canvas, goal_color, (int(c[0]), int(c[1])), int(shape.radius))
                else:
                    pts = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), canvas) for v in shape.get_vertices()]
                    pygame.draw.polygon(canvas, goal_color, pts + [pts[0]])

        for body, color in zip(self.agents, agent_colors):
            self._set_body_color(body, color)
        self._set_body_color(self.block, self.variation_space['block']['color'].value.tolist())
        self.space.debug_draw(draw_options)

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        return cv2.resize(img, (size, size)) if size != self.window_size else img

    def render_view(self, i, others=None):
        """Agent i's egocentric pixels (224 px). With others='hidden' this is exactly the
        single-agent PushT frame."""
        own = self.variation_space['agent']['color'].value.tolist()
        other = {
            'visible': own,
            'distinct': list(pygame.Color(OTHER_COLOR))[:3],
            'hidden': [*own, 0],
        }[others or self.others]
        colors = [own if j == i else other for j in range(self.n_agents)]
        return self._draw(colors, self.render_size)

    def render_audit(self, text=None):
        """Full-resolution view with a distinct color and index per agent, for auditing."""
        colors = [list(pygame.Color(AUDIT_COLORS[j % len(AUDIT_COLORS)]))[:3] for j in range(self.n_agents)]
        img = self._draw(colors, self.audit_size).copy()
        k = self.audit_size / self.window_size
        for j, body in enumerate(self.agents):
            x, y = int(body.position[0] * k), int(body.position[1] * k)
            cv2.putText(img, str(j), (x - 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4)
            cv2.putText(img, str(j), (x - 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if text:
            cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return img

    def render(self):
        return self.render_audit()


def _keep_velocity(body, gravity, damping, dt):
    pass


class MultiPushT(ParallelEnv):
    """PettingZoo parallel env: every agent acts at each step; shared reward / termination."""

    metadata = {'name': 'multipusht_v0', 'render_modes': ['rgb_array'], 'render_fps': 10, 'is_parallelizable': True}

    def __init__(self, n_agents=2, max_episode_steps=300, render_mode='rgb_array', **kwargs):
        self.core = PushTN(n_agents=n_agents, render_mode=render_mode, **kwargs)
        self.possible_agents = [f'agent_{i}' for i in range(n_agents)]
        self.agents = []
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._t = 0

        single = self.core.observation_space
        res = self.core.render_size
        self._obs_space = spaces.Dict({
            'pixels': spaces.Box(0, 255, (res, res, 3), np.uint8),
            'proprio': single['proprio'],
            'state': single['state'],
        })
        self._act_space = self.core.action_space

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._t = 0
        self.core.reset(seed=seed, options=options)
        return self._obs(), self._infos()

    def step(self, actions):
        a = np.stack([np.asarray(actions[ag], dtype=np.float32) for ag in self.possible_agents])  # float32 like the original action space
        self.core.simulate(a)
        self._t += 1

        success, dist = self.core.eval_state(self.core.goal_state, self.core._get_obs())
        truncated = self._t >= self.max_episode_steps
        rewards = {ag: -dist for ag in self.agents}
        terminations = {ag: success for ag in self.agents}
        truncations = {ag: truncated for ag in self.agents}
        obs, infos = self._obs(), self._infos()
        if success or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def _obs(self):
        state = self.core._get_obs()
        obs = {}
        for i, ag in enumerate(self.possible_agents):
            proprio, s = self.core.agent_obs(i, state)
            obs[ag] = {'pixels': self.core.render_view(i), 'proprio': proprio, 'state': s}
        return obs

    def _infos(self):
        infos = {}
        for i, ag in enumerate(self.possible_agents):
            goal_proprio, goal_state = self.core.agent_obs(i, self.core.goal_state)
            infos[ag] = {
                'goal': self.core.goals[i],
                'goal_state': goal_state,
                'goal_proprio': goal_proprio,
                'block_contact': bool(self.core.block_contact[i]),
                'agent_contact': bool(self.core.agent_contact[i]),
            }
        return infos

    def render(self):
        return self.core.render_audit(f't={self._t}')

    def state(self):
        return self.core._get_obs()

    def close(self):
        self.core.close()
