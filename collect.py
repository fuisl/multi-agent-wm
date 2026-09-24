import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))  # ./data -> big disk, see scripts/setup_storage.sh

import hydra
import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig

from world import MultiPushT  # noqa: F401


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
    # TODO: roll out MultiPushT(**cfg.env) (PettingZoo parallel API) with get_policy(cfg) per agent
    # TODO: write episodes with a swm writer: swm.data.get_format(cfg.output.format).open_writer(path)
    raise NotImplementedError


if __name__ == "__main__":
    run()
