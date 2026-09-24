import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses.
    loss = pred_loss (next-embedding MSE) + cfg.loss.sigreg.weight * sigreg_loss
    """
    raise NotImplementedError

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # TODO: swm.data.load_dataset(cfg.data.dataset.name, ...)
    # TODO: pixel preprocessor + z-score normalizer per non-pixel column
    # TODO: cfg.model.action_encoder.input_dim = frameskip * dataset.get_dim("action")
    # TODO: train / val split + DataLoaders

    ##############################
    ##       model / optim      ##
    ##############################

    # TODO: world_model = hydra.utils.instantiate(cfg.model)
    # TODO: spt.Module(model=..., sigreg=SIGReg(...), forward=partial(lejepa_forward, cfg=cfg), optim=...)

    ##########################
    ##       training       ##
    ##########################

    # TODO: WandbLogger (optional), SaveCkptCallback, pl.Trainer, spt.Manager(...)()
    raise NotImplementedError


if __name__ == "__main__":
    run()
