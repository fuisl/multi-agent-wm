import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

import world  # noqa: F401  (registers swm/MultiPushT-v0)

def img_transform(cfg):
    """ToImage -> float -> ImageNet normalize -> resize to cfg.eval.img_size."""
    raise NotImplementedError


def get_episodes_length(dataset, episodes):
    raise NotImplementedError


def get_dataset(cfg, dataset_name):
    raise NotImplementedError

@hydra.main(version_base=None, config_path="./config/eval", config_name="multipusht")
def run(cfg: DictConfig):
    """Plan with the world model (CEM / Adam MPC) from dataset start states to dataset goals."""
    # TODO: world = swm.World(**cfg.world, image_shape=(224, 224))
    # TODO: fit StandardScaler per cached column (action, proprio, state)
    # TODO: policy = swm.policy.WorldModelPolicy(solver, config, process, transform) or RandomPolicy
    # TODO: sample valid (episode, start_step) pairs, world.evaluate(dataset=..., callables=...)
    # TODO: append config + metrics to cfg.output.filename
    raise NotImplementedError


if __name__ == "__main__":
    run()
