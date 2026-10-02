"""Open-loop audit: roll the scripted team's own actions through each world model, on held-out coop scenes.

For every window (start frame t of one agent's episode, one history frame as in planning) the true action
blocks of the next K model steps (5 env steps each) are rolled out from z_t and compared to the latent of the
real frame at t + 5K. If the model is right, the true actions should bring the prediction close to that goal
and rank above other action sequences, which is what CEM needs to find them.

  cost     ||z_hat_K - z_goal||^2 for: the true actions, all-zero actions, 300 CEM-prior samples (N(0, 1) in
           normalized action space), 300 true sequences from other scenes, and CEM's plan (300 x 30, top 30)
  rank     fraction of candidates scoring below the true actions (0 = true actions are best)
  T pose   T probe (latent_audit) on the final predicted latent vs the real T at t + 5K
Windows are split by how much the T really moves over the horizon (only joint pushes move it).

    CUDA_VISIBLE_DEVICES=0 python scripts/open_loop.py coop_ftfull    # one model -> data/multipusht/coop/open_loop/
    python scripts/open_loop.py --summary
"""
import json, sys
from pathlib import Path

import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/multipusht/coop/open_loop"
MODELS = {
    "lewm": "lewm/pusht/weights.pt",
    "coop_ftpred": "lewm_coop_ftpred/weights_epoch_30.pt",
    "coop_ftfull": "lewm_coop_ftfull/weights_epoch_30.pt",
    "coop_scratch": "lewm_coop_scratch/weights_epoch_100.pt",
    "mpt2a_heuristic": "lewm_mpt2a_heuristic/weights_epoch_100.pt",
}
KS = (5, 10)          # horizons in model steps (5 = planning horizon, goal 25 env steps ahead as in eval)
S = 300               # candidates per set, as CEM's num_samples
CHUNK = 8192          # sequences per predictor batch


def summary():
    rows = {}
    for name in MODELS:
        f = OUT / f"{name}.json"
        if f.exists():
            rows[name] = json.loads(f.read_text())
    for K in KS:
        print(f"\n=== horizon {K} model steps ({5 * K} env steps)")
        print(f"{'model':16s} {'windows':10s} {'n':>5s} {'progress':>9s} {'true/zero':>9s} {'rank rand':>9s} {'rank repl':>9s}"
              f" {'T err pred':>10s} {'T err stay':>10s} {'T disp pred/true':>16s}" + (f" {'cem/true':>8s} {'cem<true':>8s} {'T disp cem':>10s}" if K == 5 else ""))
        for name, r in rows.items():
            for cat, s in r[str(K)].items():
                line = (f"{name:16s} {cat:10s} {s['n']:5d} {s['progress']:9.2f} {s['true_over_zero']:9.2f} {s['rank_random']:9.3f} {s['rank_replay']:9.3f}"
                        f" {s['T_err_pred_px']:10.1f} {s['T_err_stay_px']:10.1f} {s['T_disp_pred_px']:7.1f}/{s['T_disp_true_px']:<8.1f}")
                if K == 5:
                    line += f" {s['cem_over_true']:8.2f} {s['cem_beats_true']:8.2f} {s['T_disp_cem_px']:10.1f}"
                print(line)


def main(name):
    sys.path.insert(0, str(REPO / "scripts"))
    import latent_audit as la  # data, split, encoder, T probe and action scalers of the latent audit
    from probe import load_model

    dev = la.dev
    model = load_model(MODELS[name], dev)
    Z = la.encode_all(model)
    probe_T = la.t_pose_probe(Z)
    decode = torch.no_grad()(probe_T)
    mean, std = la.action_stats(la.scaler_source(MODELS[name]))
    Kmax = max(KS)

    # windows: test-scene episodes, every 5 frames, with Kmax true action blocks and the goal frame inside
    W = np.array([la.off[e] + s for e in range(len(la.off)) if la.scene[la.off[e]] in la.cut["test"]
                  for s in range(0, la.length[e] - 5 * Kmax - 1, 5)])
    n = len(W)
    rows = W[:, None] + np.arange(5 * Kmax)
    true = ((la.act_raw[rows] - mean) / std).reshape(n, Kmax, 10).astype(np.float32)
    assert not np.isnan(true).any()
    zero = np.broadcast_to(((0 - mean) / std).astype(np.float32), (Kmax, 5, 2)).reshape(Kmax, 10)
    g = torch.Generator().manual_seed(0)
    rand = torch.randn(n, S, Kmax, 10, generator=g)
    other_scene = la.scene[W][None, :] != la.scene[W][:, None]
    rng = np.random.default_rng(0)
    replay_idx = np.stack([rng.choice(np.nonzero(m)[0], S, replace=False) for m in other_scene])
    replay = torch.from_numpy(true)[torch.from_numpy(replay_idx)]                            # (n, S, Kmax, 10)
    cands = torch.cat([torch.from_numpy(true)[:, None], torch.from_numpy(zero)[None, None].expand(n, 1, -1, -1),
                       rand, replay], 1)                                                     # (n, 2 + 2S, Kmax, 10)

    @torch.no_grad()
    def rollout(z0, acts, keep=KS):
        """jepa.rollout with one history frame on precomputed z0: (B, D), acts (B, K, 10) -> {k: z_hat_k}."""
        out = {k: [] for k in keep}
        for i in range(0, len(z0), CHUNK):
            emb, a = z0[i:i + CHUNK, None].to(dev), acts[i:i + CHUNK].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(a.shape[1]):
                    ae = model.action_encoder(a[:, :t + 1])
                    emb = torch.cat([emb, model.predict(emb[:, -3:], ae[:, -3:])[:, -1:].float()], 1)
            for k in keep:
                out[k].append(emb[:, k].float().cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    C = cands.shape[1]
    z0 = Z[W]
    pred = rollout(z0.repeat_interleave(C, 0), cands.reshape(n * C, Kmax, 10))
    res = {}
    T_true = np.c_[la.T_pos, np.sin(la.T_ang), np.cos(la.T_ang)]
    T_now = decode(z0).numpy()

    def tdist(A, B):
        return np.linalg.norm(A[:, :2] - B[:, :2], axis=1)

    for K in KS:
        zg = Z[W + 5 * K]
        cost = (pred[K].view(n, C, -1) - zg[:, None]).pow(2).sum(-1).numpy()                 # (n, C)
        c_true, c_zero = cost[:, 0], cost[:, 1]
        c_stay = (z0 - zg).pow(2).sum(-1).numpy()
        rank_rand = (cost[:, 2:2 + S] < c_true[:, None]).mean(1)
        rank_repl = (cost[:, 2 + S:] < c_true[:, None]).mean(1)
        T_pred = decode(pred[K].view(n, C, -1)[:, 0]).numpy()
        T_goal = T_true[W + 5 * K]
        dpos_true = tdist(T_goal, T_true[W])
        dang_true = np.degrees(np.abs(np.angle(np.exp(1j * (la.T_ang[W + 5 * K] - la.T_ang[W])))))
        per = {"c_true": c_true, "c_zero": c_zero, "c_stay": c_stay, "rank_random": rank_rand, "rank_replay": rank_repl,
               "T_err_pred_px": tdist(T_pred, T_goal), "T_err_stay_px": tdist(T_now, T_goal),
               "T_disp_pred_px": tdist(T_pred, T_now), "T_disp_true_px": dpos_true}

        if K == 5:  # CEM as in eval (300 samples, 30 iterations, top 30, std init 1, plan = final mean)
            gen = torch.Generator(device=dev).manual_seed(42)
            mu, sd = torch.zeros(n, K, 10, device=dev), torch.ones(n, K, 10, device=dev)
            for _ in range(30):
                c = torch.randn(n, S, K, 10, device=dev, generator=gen) * sd[:, None] + mu[:, None]
                c[:, 0] = mu
                p = rollout(z0.repeat_interleave(S, 0), c.reshape(n * S, K, 10).cpu(), keep=(K,))[K]
                cc = (p.view(n, S, -1) - zg[:, None]).pow(2).sum(-1).to(dev)
                elite = c[torch.arange(n, device=dev)[:, None], cc.topk(30, 1, largest=False).indices]
                mu, sd = elite.mean(1), elite.std(1)
            p = rollout(z0, mu.cpu(), keep=(K,))[K]
            per["c_cem"] = (p - zg).pow(2).sum(-1).numpy()
            per["T_disp_cem_px"] = tdist(decode(p).numpy(), T_now)
            per["cem_action_dist"] = (mu.cpu() - torch.from_numpy(true[:, :K])).pow(2).mean((1, 2)).sqrt().numpy()

        cats = {"T static": (dpos_true < 1) & (dang_true < 1), "T moves": (dpos_true >= 10) | (dang_true >= 5)}
        cats["all"] = np.ones(n, bool)
        res[str(K)] = {}
        for cat, m in cats.items():
            s = {"n": int(m.sum()),
                 "progress": float(np.median(1 - per["c_true"][m] / per["c_stay"][m])),
                 "true_over_zero": float(np.median(per["c_true"][m] / per["c_zero"][m])),
                 **{k: float(np.median(per[k][m])) for k in ["rank_random", "rank_replay", "T_err_pred_px", "T_err_stay_px",
                                                             "T_disp_pred_px", "T_disp_true_px"]},
                 "true_best_of_random": float((per["rank_random"][m] == 0).mean())}
            if K == 5:
                s.update({"cem_over_true": float(np.median(per["c_cem"][m] / per["c_true"][m])),
                          "cem_beats_true": float((per["c_cem"][m] < per["c_true"][m]).mean()),
                          "T_disp_cem_px": float(np.median(per["T_disp_cem_px"][m])),
                          "cem_action_rms_from_true": float(np.median(per["cem_action_dist"][m]))})
            res[str(K)][cat] = s
            print(f"[{name}] K={K} {cat:9s} " + "  ".join(f"{k} {v:.3g}" for k, v in s.items()), flush=True)
        np.savez_compressed(OUT / f"{name}_K{K}.npz", W=W, **per)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    summary() if sys.argv[1] == "--summary" else main(sys.argv[1])
