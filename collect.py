import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))  # ./data -> big disk, see scripts/setup_storage.sh

import hydra
import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig

import world  # noqa: F401  (registers swm/MultiPushT-v0)


class HeuristicPolicy(swm.policy.BasePolicy):
    """Scripted data-collection policy (e.g. noisy go-to-block-and-push per agent)."""

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.type = 'expert'

    def set_seed(self, seed):
        raise NotImplementedError

    def set_env(self, env):
        raise NotImplementedError

    def get_action(self, info_dict, **kwargs):
        """Returns the joint action (num_envs, 2 * n_agents)."""
        raise NotImplementedError


def get_policy(cfg):
    """random -> swm.policy.RandomPolicy, heuristic -> HeuristicPolicy."""
    raise NotImplementedError


@hydra.main(version_base=None, config_path="./config/collect", config_name="multipusht")
def run(cfg: DictConfig):
    """Roll out a policy in the world and write episodes to $STABLEWM_HOME/<cfg.output.name>."""
    # TODO: world = swm.World(**cfg.world, image_shape=(cfg.img_size, cfg.img_size))
    # TODO: world.set_policy(get_policy(cfg))
    # TODO: world.collect(path, episodes=cfg.num_episodes, seed=cfg.seed, format=cfg.output.format)
    raise NotImplementedError


if __name__ == "__main__":
    run()
