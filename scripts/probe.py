"""Probe physical quantities from a world model's latent embedding (LeWM paper, Sec. 5.1 / App. F.2).

    python scripts/probe.py --data pusht_expert_train.h5 --ckpt lewm/pusht/weights.pt          # reproduce Tab. 1
    python scripts/probe.py --data multipusht_2a_heuristic_heldout.h5 --ckpt lewm/pusht/weights.pt \\
        --ckpt lewm_mpt2a_heuristic/weights_epoch_100.pt --ckpt random

Protocol, as far as the paper specifies it:
    embedding   z_t = projector(CLS of the last ViT layer), 192-d: what the predictor and the planner use
    probes      linear, and a non-linear MLP (ours: 2 hidden layers of 512, ReLU), one per quantity
    metrics     test MSE on z-scored targets (mean +- std over samples) and Pearson r (mean over dims)
    quantities  Push-T: agent location, block location, block angle (paper: scalar angle, z-scored)
The paper gives no probe optimizer settings or sample counts; ours: AdamW (lr 1e-3, wd 1e-4), batch 256,
early stopping on a validation split (patience 20 epochs), ~100k training frames (2000 episodes, stride 2: with
400 episodes at stride 4 the MLP probe undertrains, r 0.956 vs 0.989 on block angle), splits by episode (by scene for multi-agent data,
whose per-agent episodes of one scene are near-duplicates), every --stride-th frame.

Multi-agent datasets (collect.py) add "other agent location" (the nearest other agent, from global_state).
"--ckpt random" is an untrained encoder of the same architecture (floor for how much a random ViT exposes).
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parents[1] / "data"))

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import hdf5plugin  # noqa: F401  (Blosc-compressed pixels)
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

ROOT = Path(os.environ["STABLEWM_HOME"])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet stats, as utils.get_img_preprocessor
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def select_frames(h, n_episodes, stride, seed):
    """Episodes (or scenes) split 80/10/10 -> {split: sorted list of (start, stop)} row ranges."""
    off, length = h["ep_offset"][:], h["ep_len"][:]
    rng = np.random.default_rng(seed)
    if "scene_idx" in h:  # split by scene: both agents' episodes of a scene land in the same split
        scene_of_ep = h["scene_idx"][:][off]
        units = rng.permutation(np.unique(scene_of_ep))[:n_episodes]
        groups = [np.nonzero(scene_of_ep == u)[0] for u in units]
    else:
        groups = [[e] for e in rng.permutation(len(off))[:n_episodes]]
    n = len(groups)
    cut = {"train": groups[: int(0.8 * n)], "val": groups[int(0.8 * n): int(0.9 * n)], "test": groups[int(0.9 * n):]}
    return {k: sorted((int(off[e]), int(off[e] + length[e])) for g in v for e in g) for k, v in cut.items()}


def targets(h, rows):
    """Physical quantities for the given rows (dict name -> (n, d) float array)."""
    if "global_state" in h:
        gs, ag = h["global_state"][rows[0]:rows[1]], h["agent_idx"][rows[0]:rows[1]]
        n_agents = (gs.shape[1] - 3) // 4
        pos = gs[:, : 2 * n_agents].reshape(len(gs), n_agents, 2)
        own = pos[np.arange(len(gs)), ag]
        dist = np.linalg.norm(pos - own[:, None], axis=-1)
        dist[np.arange(len(gs)), ag] = np.inf
        other = pos[np.arange(len(gs)), dist.argmin(1)]
        block, angle = gs[:, 2 * n_agents: 2 * n_agents + 2], gs[:, 2 * n_agents + 2]
        out = {"agent location": own, "other agent location": other}
    else:
        st = h["state"][rows[0]:rows[1]]
        block, angle = st[:, 2:4], st[:, 4]
        out = {"agent location": st[:, :2]}
    out["block location"] = block
    out["block angle"] = angle[:, None]
    out["block angle (sin, cos)"] = np.stack([np.sin(angle), np.cos(angle)], 1)
    return out


def load_model(name, device):
    if name == "random":
        cfg = json.load(open(ROOT / "checkpoints/lewm/pusht/config.json"))
        torch.manual_seed(0)
        return hydra.utils.instantiate(OmegaConf.create(cfg)).to(device).eval()
    path = ROOT / "checkpoints" / name
    cfg = json.load(open(path.parent / "config.json"))
    model = hydra.utils.instantiate(OmegaConf.create(cfg))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def embed(model, h, ranges, stride, device, batch=128):
    feats, tgts = [], []
    for lo, hi in ranges:
        rows = np.arange(lo, hi, stride)
        for s in range(0, len(rows), batch):
            r = rows[s: s + batch]
            px = torch.from_numpy(h["pixels"][r[0]: r[-1] + 1][r - r[0]]).to(device)  # contiguous read, then subsample
            px = px.permute(0, 3, 1, 2).float().div(255)
            px = (px - MEAN.to(device)) / STD.to(device)
            if px.shape[-1] != 224:
                px = nn.functional.interpolate(px, size=224, mode="bilinear", antialias=True)
            cls = model.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            feats.append(model.projector(cls).float().cpu())
        t = targets(h, (lo, hi))
        tgts.append({k: v[rows - lo] for k, v in t.items()})
    return torch.cat(feats), {k: torch.from_numpy(np.concatenate([t[k] for t in tgts])).float() for k in tgts[0]}


def make_probe(kind, d_in, d_out):
    if kind == "linear":
        return nn.Linear(d_in, d_out)
    return nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, d_out))


def fit_probe(kind, X, Y, device, seed, max_epochs=500, patience=20, batch=256):
    torch.manual_seed(seed)
    probe = make_probe(kind, X["train"].shape[1], Y["train"].shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr = X["train"].to(device), Y["train"].to(device)
    Xva, Yva = X["val"].to(device), Y["val"].to(device)
    best, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        probe.train()
        for idx in torch.randperm(len(Xtr), device=device).split(batch):
            loss = (probe(Xtr[idx]) - Ytr[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            val = (probe(Xva) - Yva).pow(2).mean().item()
        if val < best - 1e-5:
            best, best_state, wait = val, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    probe.load_state_dict(best_state)
    return probe.eval()


def evaluate(probe, X, Y, device):
    with torch.no_grad():
        P = probe(X.to(device)).cpu()
    per_sample = (P - Y).pow(2).mean(1)
    r = [np.corrcoef(P[:, j], Y[:, j])[0, 1] for j in range(Y.shape[1])]
    return float(per_sample.mean()), float(per_sample.std()), float(np.mean(r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dataset under $STABLEWM_HOME/datasets")
    parser.add_argument("--ckpt", action="append", required=True, help="checkpoint under $STABLEWM_HOME/checkpoints, or random")
    parser.add_argument("--episodes", type=int, default=2000, help="episodes (scenes for multi-agent data) to sample")
    parser.add_argument("--stride", type=int, default=2, help="use every stride-th frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="results json (default: $STABLEWM_HOME/probes/<data>.json)")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    h = h5py.File(ROOT / "datasets" / args.data, "r")
    split = select_frames(h, args.episodes, args.stride, args.seed)
    print(f"{args.data}: {', '.join(f'{k} {len(v)} episodes' for k, v in split.items())}, stride {args.stride}")

    results = {}
    for name in args.ckpt:
        model = load_model(name, device)
        X, Y = {}, {}
        for k, ranges in split.items():
            X[k], Y[k] = embed(model, h, ranges, args.stride, device)
        del model
        mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
        X = {k: (v - mu) / sd for k, v in X.items()}
        results[name] = {}
        for q in Y["train"]:
            tm, ts = Y["train"][q].mean(0), Y["train"][q].std(0) + 1e-6
            Yq = {k: (v[q] - tm) / ts for k, v in Y.items()}
            results[name][q] = {kind: evaluate(fit_probe(kind, X, Yq, device, args.seed), X["test"], Yq["test"], device)
                                for kind in ("linear", "mlp")}
        n_test = len(X["test"])
        print(f"\n### {name}  (train {len(X['train'])}, val {len(X['val'])}, test {n_test} frames)")
        print("| quantity | linear MSE | linear r | MLP MSE | MLP r |\n|---|---|---|---|---|")
        for q, res in results[name].items():
            (lm, ls, lr), (mm, ms, mr) = res["linear"], res["mlp"]
            print(f"| {q} | {lm:.3f}±{ls:.3f} | {lr:.3f} | {mm:.3f}±{ms:.3f} | {mr:.3f} |", flush=True)

    out = Path(args.out) if args.out else ROOT / "probes" / f"{Path(args.data).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "results": results}, open(out, "w"), indent=1)
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
