"""Latent audit of the trained encoders on held-out coop scenes (no new rendering).

H1 content   linear / MLP probes: self pos, partner pos, T pos, T angle, self contact, partner contact
H2 geometry  planning cost ||z_t - z_goal||^2 (goal 25 steps later, as eval) regressed on squared state deltas
H3 physics   real transitions: self pushes alone / both push / no contact. Predicted T displacement
             (T probe on the predicted next latent) vs true; counterfactuals with own action zeroed and,
             for joint-action models ([self, partner] per step), partner action zeroed and both zeroed
Probes / splits by scene (70/10/20), test numbers on held-out-of-probe scenes.

    python scripts/latent_audit.py           # coop2 models (H100) -> data/multipusht/coop/latent_audit_coop2/
    AUDIT_SET=coop1 python scripts/latent_audit.py    # the A100 coop models -> data/multipusht/coop/latent_audit/
"""
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("STABLEWM_HOME", str(REPO / "data"))
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import h5py, hdf5plugin  # noqa
import numpy as np, torch
from omegaconf import OmegaConf
import stable_worldmodel as swm
from probe import load_model, fit_probe, MEAN, STD

torch.set_num_threads(4)
dev = "cuda"
AUDIT_SET = os.environ.get("AUDIT_SET", "coop2")
OUT = REPO / "data/multipusht/coop" / ("latent_audit" if AUDIT_SET == "coop1" else f"latent_audit_{AUDIT_SET}")
DATA = REPO / "data/datasets/multipusht_2a_coop_heldout.h5"
MODELS_COOP1 = {
    "lewm (= ftpred encoder)": "lewm/pusht/weights.pt",
    "coop_ftpred": "lewm_coop_ftpred/weights_epoch_30.pt",
    "coop_ftfull": "lewm_coop_ftfull/weights_epoch_30.pt",
    "coop_scratch": os.environ.get("SCRATCH", "lewm_coop_scratch/weights_epoch_18.pt"),
    "mpt2a_heuristic": "lewm_mpt2a_heuristic/weights_epoch_100.pt",
    "random": "random",
}
MODELS_COOP2 = {
    "lewm": "lewm/pusht/weights.pt",
    "coop2_ftfull": "lewm_coop2_ftfull/weights_epoch_30.pt",
    "coop2_joint": "lewm_coop2_joint/weights_epoch_30.pt",
    "random": "random",
}
MODELS = MODELS_COOP1 if AUDIT_SET == "coop1" else MODELS_COOP2
JOINT = {"coop2_joint"}  # action input = [self, partner] per env step
ENCODER_SAME_AS = {"coop_ftpred": "lewm (= ftpred encoder)"}  # frozen encoder: skip H1/H2

# ---------------------------------------------------------------- data
h = h5py.File(DATA, "r")
N = len(h["agent_idx"])
gs, ag, scene, step = h["global_state"][:], h["agent_idx"][:], h["scene_idx"][:], h["step_idx"][:]
act_raw = h["action"][:]
_joint = h["joint_action"][:]  # agent-index order -> egocentric [self, partner], as make_joint_dataset.py
act_joint = np.where(ag[:, None] == 0, _joint, _joint[:, [2, 3, 0, 1]])
contact = h["block_contact"][:].astype(np.float32)
off, length = h["ep_offset"][:], h["ep_len"][:]
pos = gs[:, :4].reshape(N, 2, 2)
self_pos = pos[np.arange(N), ag]
partner_pos = pos[np.arange(N), 1 - ag]
T_pos, T_ang = gs[:, 4:6], gs[:, 6]
key = {(s, t, a): i for i, (s, t, a) in enumerate(zip(scene, step, ag))}
partner_contact = np.array([contact[key[(s, t, 1 - a)]] for s, t, a in zip(scene, step, ag)], dtype=np.float32)

TARGETS = {
    "self position": self_pos, "partner position": partner_pos, "T position": T_pos,
    "T angle (sin,cos)": np.stack([np.sin(T_ang), np.cos(T_ang)], 1),
    "self touches T": contact[:, None], "partner touches T": partner_contact[:, None],
}
rng = np.random.default_rng(0)
scenes = rng.permutation(np.unique(scene))
cut = {"train": scenes[:70], "val": scenes[70:80], "test": scenes[80:]}
split = {k: np.nonzero(np.isin(scene, v))[0] for k, v in cut.items()}
print({k: len(v) for k, v in split.items()}, "frames", flush=True)


@torch.no_grad()
def encode_all(model, batch=256):
    Z = []
    for lo in range(0, N, batch):
        px = torch.from_numpy(h["pixels"][lo:lo + batch]).to(dev).permute(0, 3, 1, 2).float().div(255)
        px = (px - MEAN.to(dev)) / STD.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = model.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            Z.append(model.projector(cls).float().cpu())
    return torch.cat(Z)


def standardize(Y, idx):
    m, s = Y[idx].mean(0), Y[idx].std(0) + 1e-8
    return (Y - m) / s


def probes(Z):
    res, T_probe = {}, None
    for name, Y in TARGETS.items():
        Ys = torch.from_numpy(standardize(Y.astype(np.float32), split["train"]))
        X = {k: Z[v] for k, v in split.items()}
        Yd = {k: Ys[v] for k, v in split.items()}
        res[name] = {}
        for kind in ("linear", "mlp"):
            p = fit_probe(kind, X, Yd, dev, seed=0, max_epochs=300)
            with torch.no_grad():
                P = p(X["test"].to(dev)).cpu()
            r = np.mean([np.corrcoef(P[:, j], Yd["test"][:, j])[0, 1] for j in range(P.shape[1])])
            res[name][kind] = {"mse": float((P - Yd["test"]).pow(2).mean()), "r": float(r)}
        print(f"   {name:20s} linear r {res[name]['linear']['r']:.3f}  mlp r {res[name]['mlp']['r']:.3f}", flush=True)
    return res


def t_pose_probe(Z):
    """MLP probe z -> [T x, T y, sin, cos] in px / unit, trained on train scenes (used for H3)."""
    Y = np.concatenate([T_pos, np.sin(T_ang)[:, None], np.cos(T_ang)[:, None]], 1).astype(np.float32)
    m, s = Y[split["train"]].mean(0), Y[split["train"]].std(0)
    Ys = torch.from_numpy((Y - m) / s)
    p = fit_probe("mlp", {k: Z[v] for k, v in split.items()}, {k: Ys[v] for k, v in split.items()}, dev, seed=0, max_epochs=300)
    return lambda z: p(z.to(dev)).cpu() * torch.from_numpy(s) + torch.from_numpy(m)


def geometry(Z):
    """||z_t - z_{t+25}||^2 (same agent episode, as eval goal offset) on squared state deltas, test scenes."""
    rows = []
    for e in range(len(off)):
        if scene[off[e]] not in cut["test"]:
            continue
        for t in range(0, length[e] - 25, 2):
            rows.append((off[e] + t, off[e] + t + 25))
    a, b = np.array(rows).T
    d = (Z[a] - Z[b]).pow(2).sum(1).numpy()
    dang = np.abs(np.angle(np.exp(1j * (T_ang[a] - T_ang[b]))))
    F = np.stack([((self_pos[a] - self_pos[b]) ** 2).sum(1), ((partner_pos[a] - partner_pos[b]) ** 2).sum(1),
                  ((T_pos[a] - T_pos[b]) ** 2).sum(1), dang ** 2], 1)
    names = ["self Δpos²", "partner Δpos²", "T Δpos²", "T Δangle²"]
    Fs = F / F.std(0)
    X = np.c_[Fs, np.ones(len(Fs))]
    w, *_ = np.linalg.lstsq(X, d, rcond=None)
    pred = X @ w
    r2 = 1 - ((d - pred) ** 2).sum() / ((d - d.mean()) ** 2).sum()
    # share of the explained cost each factor carries on average (weights * mean feature)
    contrib = w[:4] * Fs.mean(0)
    share = contrib / contrib.sum()
    single_r = {n: float(np.corrcoef(Fs[:, i], d)[0, 1]) for i, n in enumerate(names)}
    # unique R2: drop one factor at a time
    uniq = {}
    for i, n in enumerate(names):
        Xi = np.delete(X, i, axis=1)
        wi, *_ = np.linalg.lstsq(Xi, d, rcond=None)
        r2i = 1 - ((d - Xi @ wi) ** 2).sum() / ((d - d.mean()) ** 2).sum()
        uniq[n] = float(r2 - r2i)
    out = {"n_pairs": len(d), "r2": float(r2), "share": dict(zip(names, share.tolist())),
           "unique_r2": uniq, "single_r": single_r, "std_weights": dict(zip(names, w[:4].tolist()))}
    print("   geometry R2 %.3f | share " % r2 + "  ".join(f"{n} {s:.2f}" for n, s in zip(names, share))
          + " | unique R2 " + "  ".join(f"{n} {u:.3f}" for n, u in uniq.items()), flush=True)
    return out


def scaler_source(name):
    p = REPO / "data/checkpoints" / name
    cfg = p.parent / "config.yaml"
    if not cfg.exists():
        return "pusht_expert_train.h5"
    c = OmegaConf.load(cfg)
    return OmegaConf.select(c, "scalers_from") or OmegaConf.select(c, "data.dataset.name")


_st = {}
def action_stats(ds):
    if ds not in _st:
        d = swm.data.load_dataset(ds, transform=None, num_steps=4, frameskip=5, keys_to_load=["action"], keys_to_cache=["action"])
        a = torch.from_numpy(np.array(d.get_col_data("action")))
        a = a[~torch.isnan(a).any(1)]
        _st[ds] = (a.mean(0).numpy(), a.std(0).numpy())
    return _st[ds]


@torch.no_grad()
def physics(model, Z, decode, ck, name):
    """windows of 4 frames (stride 5) in test scenes; predict frame 3; classify transition 2->3 by contacts."""
    mean, std = action_stats(scaler_source(ck))
    joint = name in JOINT
    A, d = (act_joint, 4) if joint else (act_raw, 2)
    if joint:  # single-agent scaler tiled over [self, partner], as train.py
        mean, std = np.tile(mean, 2), np.tile(std, 2)
    W = []
    for e in range(len(off)):
        if scene[off[e]] not in cut["test"]:
            continue
        for s in range(0, length[e] - 16, 3):
            W.append(off[e] + s)
    W = np.array(W)
    frames = W[:, None] + 5 * np.arange(4)
    acts = np.stack([A[W + 5 * k + j] for k in range(4) for j in range(5)], 1).reshape(len(W), 4, 5, d)
    acts = np.nan_to_num((acts - mean) / std).astype(np.float32)
    zero = ((0 - mean) / std).astype(np.float32)  # a zero env action, z-scored (d,)
    tr = frames[:, 2][:, None] + np.arange(5)  # rows of transition 2 -> 3
    sc, pc = contact[tr].max(1), partner_contact[tr].max(1)
    cls = np.where((sc > 0) & (pc > 0), "both push", np.where(sc > 0, "self alone", np.where(pc > 0, "partner alone", "no contact")))
    true_T = np.c_[T_pos, np.sin(T_ang), np.cos(T_ang)]

    def pred_T(a):
        P = []
        for i in range(0, len(W), 256):
            z = Z[frames[i:i + 256]].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ae = model.action_encoder(torch.from_numpy(a[i:i + 256]).to(dev))
                p = model.predict(z[:, :3], ae[:, :3]).float()
            P.append(p[:, 2].cpu())
        return decode(torch.cat(P)).numpy()

    def disp(A, B):  # px translation + degrees rotation
        dpos = np.linalg.norm(A[:, :2] - B[:, :2], axis=1)
        dang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(A[:, 2], A[:, 3]) - np.arctan2(B[:, 2], B[:, 3]))))))
        return dpos, dang

    now = decode(Z[frames[:, 2]]).numpy()
    flat = lambda a: a.reshape(len(W), 4, 5 * d)
    p_true = pred_T(flat(acts))
    a0 = acts.copy(); a0[:, 2, :, :2] = zero[:2]  # own action zeroed in the transition 2 -> 3
    p_zero = pred_T(flat(a0))
    cf = {}
    if joint:
        ap = acts.copy(); ap[:, 2, :, 2:] = zero[2:]
        ab = acts.copy(); ab[:, 2] = zero
        cf = {"partner_zeroed": disp(pred_T(flat(ap)), now)[0], "both_zeroed": disp(pred_T(flat(ab)), now)[0]}
    tpos, tang = disp(true_T[frames[:, 3]], true_T[frames[:, 2]])
    ppos, pang = disp(p_true, now)
    zpos, zang = disp(p_zero, now)
    out = {}
    for c in ["no contact", "self alone", "partner alone", "both push"]:
        m = cls == c
        out[c] = {"n": int(m.sum()), "true_px": float(np.median(tpos[m])), "pred_px": float(np.median(ppos[m])),
                  "pred_own_action_zeroed_px": float(np.median(zpos[m])),
                  "true_deg": float(np.median(tang[m])), "pred_deg": float(np.median(pang[m])),
                  "zeroed_deg": float(np.median(zang[m]))}
        for k, v in cf.items():
            out[c][f"pred_{k}_px"] = float(np.median(v[m]))
        o = out[c]
        print(f"   {c:14s} n={o['n']:5d}  T moves: true {o['true_px']:5.1f}px {o['true_deg']:4.1f}°  "
              f"predicted {o['pred_px']:5.1f}px {o['pred_deg']:4.1f}°  own action zeroed {o['pred_own_action_zeroed_px']:5.1f}px {o['zeroed_deg']:4.1f}°"
              + "".join(f"  {k.replace('_', ' ')} {o[f'pred_{k}_px']:5.1f}px" for k in cf), flush=True)
    return out


def pca_plot(Zs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = split["test"][::3]
    cols = {"self x": self_pos[idx, 0], "partner x": partner_pos[idx, 0], "T angle": np.mod(T_ang[idx], 2 * np.pi)}
    fig, axes = plt.subplots(len(Zs), 3, figsize=(10, 3.1 * len(Zs)))
    for r, (name, Z) in enumerate(Zs.items()):
        X = Z[idx].numpy(); X = X - X.mean(0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        P = X @ Vt[:2].T
        for c, (cn, cv) in enumerate(cols.items()):
            ax = axes[r, c]
            ax.scatter(P[:, 0], P[:, 1], c=cv, s=2, cmap="twilight" if cn == "T angle" else "viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(f"colored by {cn}")
            if c == 0: ax.set_ylabel(name)
    fig.tight_layout(); fig.savefig(OUT / "latent_pca.png", dpi=110)


def spectrum(Z):
    X = Z[split["test"]].numpy(); X = X - X.mean(0)
    ev = np.linalg.eigvalsh(np.cov(X.T))[::-1].clip(min=0)
    p = ev / ev.sum()
    return {"effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum())), "top10_var": float(p[:10].sum()),
            "mean_norm": float(np.linalg.norm(Z[split['test']].numpy(), axis=1).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, Zs = {}, {}
    for name, ck in MODELS.items():
        print(f"== {name} ({ck})", flush=True)
        model = load_model(ck, dev)
        Z = encode_all(model)
        r = {"ckpt": ck}
        if name not in ENCODER_SAME_AS:
            Zs[name] = Z
            r["spectrum"] = spectrum(Z); print("   spectrum", r["spectrum"], flush=True)
            r["probes"] = probes(Z)
            r["geometry"] = geometry(Z)
        if name != "random":
            r["physics"] = physics(model, Z, t_pose_probe(Z), ck, name)
        results[name] = r
        del model; torch.cuda.empty_cache()
        (OUT / "latent_audit.json").write_text(json.dumps(results, indent=1))
    pca_plot(Zs)
    print("->", OUT / "latent_audit.json", OUT / "latent_pca.png")


if __name__ == "__main__":
    main()
