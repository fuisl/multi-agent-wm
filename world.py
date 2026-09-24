"""MultiPush-T world: N agents cooperatively push a T-block to a goal pose.

Two layers:
  - MultiPushTScenario: VMAS scenario (true multi-agent physics, per-agent obs/reward).
    Usable directly with vmas.make_env(...) for decentralized / MARL / PettingZoo work.
  - MultiPushT: centralized Gymnasium wrapper for stable-worldmodel.
    Exposes a joint action a = [a^1, ..., a^N] and a dynamics-sufficient state S_t,
    so swm.World / collection / CEM planning work unchanged.

State layout (world-model state, NOT the per-agent policy obs):
    state   = [agent_1, ..., agent_N, block]              dim = 4N + 7
      agent_i = [x, y, vx, vy]
      block   = [x, y, vx, vy, sin(theta), cos(theta), omega]
    proprio = [agent_1, ..., agent_N]                     dim = 4N
    action  = [ux_1, uy_1, ..., ux_N, uy_N] in [-1, 1]    dim = 2N
"""

import gymnasium as gym
import numpy as np
import torch
import vmas
from gymnasium import spaces
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario

import stable_worldmodel as swm
from stable_worldmodel import spaces as swm_spaces

AGENT_DIM = 4
BLOCK_DIM = 7


###################
## VMAS scenario ##
###################

class MultiPushTScenario(BaseScenario):
    """VMAS scenario: N holonomic disc agents, one T-shaped rigid body, one goal pose."""

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """Build agents, the T-block (two jointed boxes or a compound body) and the goal landmark.
        kwargs: n_agents, agent_radius, block_mass, ...
        """
        raise NotImplementedError

    def reset_world_at(self, env_index=None):
        """Sample agent / block start poses and the goal pose for env `env_index` (all if None)."""
        raise NotImplementedError

    def observation(self, agent: Agent):
        """Decentralized obs o_i = g_i(S): egocentric relative positions / velocities
        of self, block, goal and other agents (optionally range-limited)."""
        raise NotImplementedError

    def reward(self, agent: Agent):
        """Shared team reward (e.g. goal coverage / -pose distance of the block)."""
        raise NotImplementedError

    def done(self):
        """(batch_dim,) bool: block pose within tolerance of the goal pose."""
        raise NotImplementedError

    def info(self, agent: Agent) -> dict:
        """Extra per-agent info (contacts, distances) for logging."""
        raise NotImplementedError

    def extra_render(self, env_index: int = 0):
        """Draw the goal T outline."""
        raise NotImplementedError


#################################
## Centralized Gym env for swm ##
#################################

class MultiPushT(gym.Env):
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 10,
    }

    def __init__(
        self,
        n_agents=2,
        resolution=224,
        render_mode='rgb_array',
        device='cpu',
        init_value=None,
    ):
        self.n_agents = n_agents
        self.render_size = resolution
        self.render_mode = render_mode
        self.device = device
        self.goal_state = None
        self.env = None  # vmas env, built lazily in reset()

        state_dim = AGENT_DIM * n_agents + BLOCK_DIM
        self.observation_space = spaces.Dict(
            {
                'proprio': spaces.Box(-np.inf, np.inf, (AGENT_DIM * n_agents,), np.float64),
                'state': spaces.Box(-np.inf, np.inf, (state_dim,), np.float64),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2 * n_agents,), np.float32)

        # TODO: factors of variation (start poses, block mass / friction, colors, ...)
        self.variation_space = swm_spaces.Dict({})
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

    def reset(self, seed=None, options=None):
        """Reset VMAS, sample (or take from options) `state` / `goal_state`,
        render the goal image into info['goal']. Returns (obs, info)."""
        super().reset(seed=seed, options=options)
        raise NotImplementedError

    def step(self, action):
        """Split joint action into per-agent actions, step VMAS.
        Returns (obs, reward, terminated, truncated, info)."""
        raise NotImplementedError

    def eval_state(self, goal_state, cur_state):
        """Returns (success, state_dist) from block position / angle error."""
        raise NotImplementedError

    def render(self):
        """RGB frame (H, W, 3) at self.render_size."""
        raise NotImplementedError

    def close(self):
        pass

    def _get_obs(self):
        """Centralized, dynamics-sufficient state S_t (see layout above)."""
        raise NotImplementedError

    def _get_info(self):
        """Must contain 'goal' (goal image) and 'goal_state' / 'goal_proprio' for swm."""
        raise NotImplementedError

    def _set_state(self, state):
        """Write a full state vector back into the VMAS world (used by swm eval callables)."""
        raise NotImplementedError

    def _set_goal_state(self, goal_state):
        self.goal_state = goal_state


swm.envs.register(
    id='swm/MultiPushT-v0',
    entry_point='world:MultiPushT',
)
