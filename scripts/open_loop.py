"""Open-loop audit: roll the scripted team's own actions through each world model, on held-out coop scenes.

For every window (start frame t of one agent's episode, one history frame as in planning) the true action
blocks of the next K model steps (5 env steps each) are rolled out from z_t and compared to the latent of the
real frame at t + 5K. If the model is right, the true actions should bring the prediction close to that goal
and rank above other action sequences, which is what CEM needs to find them.

  cost     ||z_hat_K - z_goal||^2 for: the true actions, all-zero actions, 300 CEM-prior samples (N(0, 1) in
           normalized action space), 300 true sequences from other scenes, and CEM's plan (300 x 30, top 30)
  rank     fraction of candidates scoring below the true actions (0 = true actions are best)
  T pose   T probe (latent_audit) on the final predicted latent vs the real T at t + 5K
  sim      (horizon 5) CEM's plan executed in the simulator from the window's true global state: own half of the
           plan, partner replaying its recorded actions (and, for joint-action models, both halves of the plan).
           The real final frame is encoded: does the plan that wins in the model win in reality (F12)? The true
           actions are executed the same way, to measure how faithfully the reset reproduces the dataset.
Windows are split by how much the T really moves over the horizon (only joint pushes move it).

    python scripts/open_loop.py coop2_joint    # one model -> data/multipusht/coop/open_loop_coop2/ (A100 models: open_loop/)
    python scripts/open_loop.py --summary
"""
import json, os, sys
from pathlib import Path

import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
MODELS_COOP1 = {
    "lewm": "lewm/pusht/weights.pt",
    "coop_ftpred": "lewm_coop_ftpred/weights_epoch_30.pt",
    "coop_ftfull": "lewm_coop_ftfull/weights_epoch_30.pt",
    "coop_scratch": "lewm_coop_scratch/weights_epoch_100.pt",
    "mpt2a_heuristic": "lewm_mpt2a_heuristic/weights_epoch_100.pt",
}
MODELS_COOP2 = {
    "lewm": "lewm/pusht/weights.pt",
    "coop2_ftfull": "lewm_coop2_ftfull/weights_epoch_30.pt",
    "coop2_joint": "lewm_coop2_joint/weights_epoch_30.pt",
}
JOINT = {"coop2_joint"}
AUDIT_SET = os.environ.get("AUDIT_SET", "coop2")
MODELS = MODELS_COOP1 if AUDIT_SET == "coop1" else MODELS_COOP2
OUT = REPO / "data/multipusht/coop" / ("open_loop" if AUDIT_SET == "coop1" else f"open_loop_{AUDIT_SET}")
ENV = dict(n_agents=2, others="distinct", agent_force=1e5, solver_iterations=50, jpeg_quality=95)  # config/collect/coop.yaml
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


def simulate(W, variants, dev, model):
    """Each window's true global state -> run each variant's env actions (n, steps, 4 = [self, partner]) -> encode
    the real final own view. Returns {variant: (z_real (n, 192), real T pose [x, y, sin, cos] (n, 4))}."""
    import latent_audit as la
    from world import MultiPushT
    env = MultiPushT(max_episode_steps=10 ** 6, **ENV)
    out = {}
    for v, acts in variants.items():
        px, T = [], []
        for i, w in enumerate(W):
            a = la.ag[w]
            env.reset(seed=0, options={"state": la.gs[w].astype(np.float64)})
            for t in range(acts.shape[1]):
                own, partner = acts[i, t, :2], acts[i, t, 2:]
                env.core.simulate(np.stack([own, partner] if a == 0 else [partner, own]).astype(np.float32))
            px.append(env.core.render_view(a))
            g = env.core._get_obs()
            T.append([g[4], g[5], np.sin(g[6]), np.cos(g[6])])
        px = np.stack(px)
        Z = []
        with torch.no_grad():
            for lo in range(0, len(px), 256):
                x = torch.from_numpy(px[lo:lo + 256]).to(dev).permute(0, 3, 1, 2).float().div(255)
                x = (x - la.MEAN.to(dev)) / la.STD.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Z.append(model.projector(model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float().cpu())
        out[v] = (torch.cat(Z), np.array(T))
        print(f"   simulated {v}: {len(W)} windows", flush=True)
    return out


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
    joint = name in JOINT
    A, d = (la.act_joint, 4) if joint else (la.act_raw, 2)
    if joint:
        mean, std = np.tile(mean, 2), np.tile(std, 2)
    D = 5 * d  # action input per model step
    Kmax = max(KS)

    # windows: test-scene episodes, every 5 frames, with Kmax true action blocks and the goal frame inside
    W = np.array([la.off[e] + s for e in range(len(la.off)) if la.scene[la.off[e]] in la.cut["test"]
                  for s in range(0, la.length[e] - 5 * Kmax - 1, 5)])
    n = len(W)
    rows = W[:, None] + np.arange(5 * Kmax)
    true = ((A[rows] - mean) / std).reshape(n, Kmax, D).astype(np.float32)
    assert not np.isnan(true).any()
    zero = np.broadcast_to(((0 - mean) / std).astype(np.float32), (Kmax, 5, d)).reshape(Kmax, D)
    g = torch.Generator().manual_seed(0)
    rand = torch.randn(n, S, Kmax, D, generator=g)
    other_scene = la.scene[W][None, :] != la.scene[W][:, None]
    rng = np.random.default_rng(0)
    replay_idx = np.stack([rng.choice(np.nonzero(m)[0], S, replace=False) for m in other_scene])
    replay = torch.from_numpy(true)[torch.from_numpy(replay_idx)]                            # (n, S, Kmax, D)
    cands = torch.cat([torch.from_numpy(true)[:, None], torch.from_numpy(zero)[None, None].expand(n, 1, -1, -1),
                       rand, replay], 1)                                                     # (n, 2 + 2S, Kmax, D)

    @torch.no_grad()
    def rollout(z0, acts, keep=KS):
        """jepa.rollout with one history frame on precomputed z0: (B, 192), acts (B, K, D) -> {k: z_hat_k}."""
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
    pred = rollout(z0.repeat_interleave(C, 0), cands.reshape(n * C, Kmax, D))
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
            mu, sd = torch.zeros(n, K, D, device=dev), torch.ones(n, K, D, device=dev)
            for _ in range(30):
                c = torch.randn(n, S, K, D, device=dev, generator=gen) * sd[:, None] + mu[:, None]
                c[:, 0] = mu
                p = rollout(z0.repeat_interleave(S, 0), c.reshape(n * S, K, D).cpu(), keep=(K,))[K]
                cc = (p.view(n, S, -1) - zg[:, None]).pow(2).sum(-1).to(dev)
                elite = c[torch.arange(n, device=dev)[:, None], cc.topk(30, 1, largest=False).indices]
                mu, sd = elite.mean(1), elite.std(1)
            p = rollout(z0, mu.cpu(), keep=(K,))[K]
            per["c_cem"] = (p - zg).pow(2).sum(-1).numpy()
            per["T_disp_cem_px"] = tdist(decode(p).numpy(), T_now)
            per["cem_action_dist"] = (mu.cpu() - torch.from_numpy(true[:, :K])).pow(2).mean((1, 2)).sqrt().numpy()
            # execute in the simulator (env actions, unnormalized and clipped like the env's action space)
            plan_env = np.clip(mu.cpu().numpy().reshape(n, 5 * K, d) * std + mean, -1, 1)
            true_env = np.nan_to_num(A[rows[:, :5 * K]])                                        # (n, 5K, d)
            partner_env = la.act_joint[rows[:, :5 * K]][..., 2:]                                 # recorded partner actions
            variants = {"true": np.concatenate([true_env[..., :2], partner_env], -1),
                        "cem_own": np.concatenate([plan_env[..., :2], partner_env], -1)}
            if joint:
                variants["cem_both"] = plan_env
            sim = simulate(W, variants, dev, model)
            for v, (zr, Tr) in sim.items():
                per[f"sim_{v}_c"] = (zr - zg).pow(2).sum(-1).numpy()
                per[f"sim_{v}_T_err_px"] = tdist(Tr, T_goal)
                per[f"sim_{v}_T_disp_px"] = tdist(Tr, T_true[W])

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
                for k in [k for k in per if k.startswith("sim_")]:
                    s[k] = float(np.median(per[k][m]))
                # does the plan that wins in the model also win in reality? (both measured in the real frame's latent)
                s["sim_cem_own_beats_true"] = float((per["sim_cem_own_c"][m] < per["sim_true_c"][m]).mean())
                s["model_cem_over_sim_cem_own"] = float(np.median(per["c_cem"][m] / per["sim_cem_own_c"][m]))
            res[str(K)][cat] = s
            print(f"[{name}] K={K} {cat:9s} " + "  ".join(f"{k} {v:.3g}" for k, v in s.items()), flush=True)
        np.savez_compressed(OUT / f"{name}_K{K}.npz", W=W, **per)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    summary() if sys.argv[1] == "--summary" else main(sys.argv[1])
