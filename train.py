import os
from functools import partial
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))  # ./data -> big disk, see scripts/setup_storage.sh

import h5py
import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def split_by_episode(dataset, train_split, seed):
    """Train/val split of whole episodes, and of whole scenes for multi-agent data (collect.py writes one
    episode per agent, and the agents' episodes of a scene show the same events). The official random
    window split puts near-copies of training windows in val (neighbouring windows share 3 of 4 frames),
    so its val loss tracks memorization: 0.003 there vs 0.017 on unseen scenes for lewm_coop_ftfull."""
    with h5py.File(dataset.h5_path, "r") as f:
        offsets = f["ep_offset"][:]
        group = f["scene_idx"][:][offsets] if "scene_idx" in f else np.arange(len(offsets))
    units = np.random.default_rng(seed).permutation(np.unique(group))
    val_units = units[: round(len(units) * (1 - train_split))]
    is_val = np.isin(group[[ep for ep, _ in dataset.clip_indices]], val_units)
    return spt.data.Subset(dataset, np.nonzero(~is_val)[0].tolist()), spt.data.Subset(dataset, np.nonzero(is_val)[0].tolist())


def widen_action_encoder(state, model, frameskip):
    """Load a single-agent action encoder into a joint-action one, keeping its function.

    The Embedder maps a step's frameskip actions, packed time-major, through a 1x1 conv (in -> smoothed)
    and an MLP. Joint data packs [self, partner] per env step, so source input column k*a + d (step k,
    coordinate d of the a-dim action) becomes k*A + d, with A the joint action dim; the partner columns
    are zero in the source rows. The new conv rows keep their random init and the MLP's columns for them
    are zero, so the widened model computes exactly the source function, and gradient still reaches the
    new path (zero-init on the output side only, as in ControlNet's zero convolution).
    """
    own = model.state_dict()
    conv, lin = "action_encoder.patch_embed", "action_encoder.embed.0"
    if state[f"{conv}.weight"].shape == own[f"{conv}.weight"].shape:
        return state
    w, b, lw = state[f"{conv}.weight"], state[f"{conv}.bias"], state[f"{lin}.weight"]
    s_out, s_in = w.shape[:2]
    a, A = s_in // frameskip, own[f"{conv}.weight"].shape[1] // frameskip
    cols = torch.tensor([k * A + d for k in range(frameskip) for d in range(a)])
    new_w, new_b, new_lw = own[f"{conv}.weight"].clone(), own[f"{conv}.bias"].clone(), own[f"{lin}.weight"].clone()
    new_w[:s_out] = 0.0
    new_w[:s_out, cols] = w
    new_b[:s_out] = b
    new_lw.zero_()
    new_lw[:, :s_out] = lw
    print(f"widened action encoder: conv {tuple(w.shape)} -> {tuple(new_w.shape)}, embed {tuple(lw.shape)} -> {tuple(new_lw.shape)}")
    return {**state, f"{conv}.weight": new_w, f"{conv}.bias": new_b, f"{lin}.weight": new_lw}


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # frozen parts stay in eval mode (projector BatchNorm keeps its source statistics); Lightning
    # switches the whole module back to train mode at every epoch start
    for name in cfg.init.freeze:
        getattr(self.model, name).eval()

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    stats_dataset = dataset
    if cfg.scalers_from:
        stats_dataset = swm.data.load_dataset(
            cfg.scalers_from, transform=None, cache_dir=cache_dir, **{**dataset_cfg, "keys_to_load": dataset_cfg["keys_to_cache"]}
        )

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            # joint-action data ([self, partner] per step) with single-agent scalers: z-score each agent's half alike
            tile = dataset.get_dim(col) // stats_dataset.get_dim(col) if col == "action" else 1
            normalizer = get_column_normalizer(stats_dataset, col, col, tile=tile)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    spt_cache = run_dir / "spt"
    last_ckpts = sorted(spt_cache.glob("runs/*/*/*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    if last_ckpts and (run_dir / "config.yaml").exists():
        # a resumed run keeps the split it started with (runs from before split_by used windows)
        with open_dict(cfg):
            cfg.split_by = OmegaConf.select(OmegaConf.load(run_dir / "config.yaml"), "split_by", default="window")
    if cfg.split_by == "episode":
        train_set, val_set = split_by_episode(dataset, cfg.train_split, cfg.seed)
    else:  # "window": the official LeWM split
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
    print(f"split_by={cfg.split_by}: {len(train_set)} train / {len(val_set)} val windows", flush=True)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    if cfg.init.ckpt:
        ckpt = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.init.ckpt)
        world_model.load_state_dict(widen_action_encoder(torch.load(ckpt, map_location="cpu"), world_model, cfg.data.dataset.frameskip))
    for name in cfg.init.freeze:
        getattr(world_model, name).requires_grad_(False)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )
    # full training state (optimizer, scheduler, epoch) in last.ckpt, written before validation so
    # a crash at the train -> val switch loses nothing; rerunning with the same subdir resumes from
    # it. spt.Manager redirects every ModelCheckpoint into its cache_dir, so keep that cache per run
    # (on the big disk, not ~/.cache) and resume from the newest last.ckpt in it
    spt.set(cache_dir=str(spt_cache))
    resume_callback = ModelCheckpoint(
        dirpath=run_dir, filename="last", save_on_train_epoch_end=True, enable_version_counter=False,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, resume_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = last_ckpts[-1] if last_ckpts else None
    print(f"resuming from {ckpt_path}" if ckpt_path else "no last.ckpt, starting fresh", flush=True)
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
        weights_only=False,
    )

    manager()
    return


if __name__ == "__main__":
    run()
