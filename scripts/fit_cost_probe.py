"""Fit the probe that probe-cost planning reads (eval_multi.py cost=probe): z -> [self x, y, T x, y, sin, cos].

An MLP probe (probe.make_probe) on the model's own latents of the held-out coop scenes (latent_audit.py data and
scene split), so the planner can score candidates by the success test's own quantities instead of ||z - z_goal||^2.

    python scripts/fit_cost_probe.py lewm_coop2_ftfull/weights_epoch_30.pt
    # -> $STABLEWM_HOME/checkpoints/lewm_coop2_ftfull/cost_probe_weights_epoch_30.pt
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch

import latent_audit as la
from probe import fit_probe, load_model

TARGETS = ["self x", "self y", "T x", "T y", "T sin", "T cos"]


def out_path(ckpt):
    p = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / ckpt
    return p.parent / f"cost_probe_{p.stem}.pt"


def main(ckpt):
    out = out_path(ckpt)
    model = load_model(ckpt, la.dev)
    Z = la.encode_all(model)
    Y = np.concatenate([la.self_pos, la.T_pos, np.sin(la.T_ang)[:, None], np.cos(la.T_ang)[:, None]], 1).astype(np.float32)
    m, s = Y[la.split["train"]].mean(0), Y[la.split["train"]].std(0)
    Ys = torch.from_numpy((Y - m) / s)
    probe = fit_probe("mlp", {k: Z[v] for k, v in la.split.items()}, {k: Ys[v] for k, v in la.split.items()},
                      la.dev, seed=0, max_epochs=300)
    with torch.no_grad():
        P = probe(Z[la.split["test"]].to(la.dev)).cpu().numpy() * s + m
    T = Y[la.split["test"]]
    r = {n: float(np.corrcoef(P[:, j], T[:, j])[0, 1]) for j, n in enumerate(TARGETS)}
    err = {"self px": float(np.median(np.linalg.norm(P[:, :2] - T[:, :2], axis=1))),
           "T px": float(np.median(np.linalg.norm(P[:, 2:4] - T[:, 2:4], axis=1))),
           "T deg": float(np.median(np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(P[:, 4], P[:, 5]) - np.arctan2(T[:, 4], T[:, 5]))))))))}
    print(f"{ckpt}: test r {r}\n   median error {err}", flush=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")  # parallel jobs may fit the same probe: write whole files only
    torch.save({"kind": "mlp", "in_dim": Z.shape[1], "out_dim": Y.shape[1], "state_dict": probe.cpu().state_dict(),
                "mean": m, "std": s, "targets": TARGETS, "test_r": r, "test_median_err": err, "ckpt": ckpt}, tmp)
    os.replace(tmp, out)
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1])
