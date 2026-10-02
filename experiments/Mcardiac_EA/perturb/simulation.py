#!/usr/bin/env python3
"""Exploratory target-free Stage2 simulation for GSE282547.

No target quotas, target occupancy, normal coordinate advection, or teacher-latent
replacement is used after the E10.5 state is initialized.
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

import sys
sys.dont_write_bytecode = True
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR.parents[1] / "src"))
from workflow import read_training_manifest
from stvirtual.perturb import prepare_runtime
prepare_runtime(EXPERIMENT_DIR)
from stvirtual.models import stage2_3d_transition as s2


def clone_state(state):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}


def subset_state(state, keep):
    n = int(keep.numel())
    return {
        k: (v[keep] if torch.is_tensor(v) and v.ndim > 0 and int(v.shape[0]) == n else v)
        for k, v in state.items()
    }


def as_numpy(x, dtype=None):
    y = x.detach().cpu().numpy()
    return y.astype(dtype) if dtype else y


class NicheMotionMLP(torch.nn.Module):
    """Small E10.5-only local niche update model.

    It is fitted to local E10.5 neighbor relaxation targets; no E12.5 target
    coordinates, target occupancy, teacher latent, or KNN target latent is used.
    """
    def __init__(self, latent_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 4, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
        )
        self.latent_head = torch.nn.Linear(hidden, latent_dim)
        self.xy_head = torch.nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.net(x)
        return self.latent_head(h), self.xy_head(h)


@torch.no_grad()
def local_niche_features(z, xy, lr, time_value, k=16):
    n = int(xy.shape[0])
    if n <= 1:
        dmean = torch.zeros((n, 1), device=xy.device, dtype=xy.dtype)
        zdiff_norm = torch.zeros((n, 1), device=xy.device, dtype=xy.dtype)
        zmean = z.clone()
        xymean = xy.clone()
    else:
        kk = min(int(k), n - 1)
        d = torch.cdist(xy, xy)
        d.fill_diagonal_(float("inf"))
        vals, idx = torch.topk(d, kk, largest=False)
        zmean = z[idx].mean(dim=1)
        xymean = xy[idx].mean(dim=1)
        dmean = vals.mean(dim=1, keepdim=True)
        zdiff_norm = (zmean - z).norm(dim=1, keepdim=True)
    tcol = torch.full((n, 1), float(time_value), device=xy.device, dtype=z.dtype)
    x = torch.cat([z, lr.view(-1, 1), dmean.to(z.dtype), zdiff_norm.to(z.dtype), tcol], dim=1)
    return x, zmean - z, xymean - xy


def fit_niche_motion(z, xy, lr, nn_scale, device, seed=2026):
    torch.manual_seed(int(seed))
    latent_dim = int(z.shape[1])
    model = NicheMotionMLP(latent_dim).to(device)
    x0, dz_target, dxy_target = local_niche_features(z, xy, lr, 0.0)
    xs, zs, ys = [], [], []
    for t in torch.linspace(0.0, 1.0, 5, device=device):
        scale = 0.35 + 0.65 * t
        xs.append(x0.clone()); xs[-1][:, -1] = t
        zs.append(dz_target * scale); ys.append(dxy_target * scale)
    x = torch.cat(xs); y_lat = torch.cat(zs); y_xy = torch.cat(ys)
    # Preserve the original magnitude-aware latent regression target.
    zstd = z.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_lat = torch.clamp(y_lat, -0.05 * zstd, 0.05 * zstd)
    y_xy = torch.clamp(y_xy, -0.25 * nn_scale, 0.25 * nn_scale)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(250):
        pz, pxy = model(x)
        loss = torch.nn.functional.smooth_l1_loss(pz, y_lat) + torch.nn.functional.smooth_l1_loss(pxy, y_xy)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, zstd.squeeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-manifest", required=True, type=Path)
    p.add_argument(
        "--ablation-mode", choices=["initial"], default="initial",
        help="Apply Wt1-weighted partial ablation once at the initial E10.5 frame.",
    )
    p.add_argument("--out-subdir", default="counterfactual_free_simulation")
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument(
        "--ablation-fraction", type=float, default=0.4,
        help="Fraction of initial E10.5 epicardial cells selected for ablation.",
    )
    p.add_argument("--birth-hazard-max", type=float, default=0.10)
    p.add_argument("--death-hazard-max", type=float, default=0.04)
    p.add_argument("--growth-target-count", type=int, default=0)
    p.add_argument("--latent-noise", type=float, default=0.005)
    p.add_argument("--lr-latent-alpha", type=float, default=0.0)
    p.add_argument("--latent-step-fraction", type=float, default=0.08)
    p.add_argument("--coord-step-fraction", type=float, default=0.15)
    p.add_argument("--message-passing-substeps", type=int, default=3)
    p.add_argument("--latent-update-gain", type=float, default=1.2)
    p.add_argument(
        "--ablation-only", action="store_true",
        help="Run only the perturbed simulation; do not generate normal_free.",
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    manifest = read_training_manifest(a.training_manifest)
    a.source, a.target, a.steps = manifest["source"], manifest["target"], int(manifest["steps"])
    out = a.output_root.resolve() / a.out_subdir
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg = s2.StageCfg(
        src=a.source, tgt=a.target,
        out_npz_path=manifest["stage1_trace"], bound_dir=manifest["bound_dir"],
        decoder_checkpoint=manifest["decoder_checkpoint"],
        latent_key=manifest["latent_key"], use_latent=True, use_lr=True,
        lr_source="decoder", counts_layer="counts", layer_col="cell_type",
        diff_csv=manifest["transition_prior"],
    )
    ctx = s2.build_global_ctx(
        adata_path=manifest["adata_path"], lr_pairs_path=manifest["lr_pairs"],
        ckpt_3dslice=manifest["stage1_checkpoint"], device=dev, layer_col="cell_type",
    )
    pack = s2.prepare_one_stage(ctx, cfg, layer_col="cell_type", sample_key="sample", AUTO_PRINT=True)
    policy_path = Path(manifest["policy_checkpoint"])
    checkpoint = torch.load(policy_path, map_location=dev, weights_only=False)
    alpha_net = None
    if hasattr(s2, "_load_policy_net"):
        policy = s2._load_policy_net(s2, str(policy_path), n_layers=pack.n_layers, device=dev)
    else:
        policy, alpha_net = s2._load_policy_and_alpha_net(
            s2, str(policy_path), n_layers=pack.n_layers, device=dev, latent_dim=pack.state0["latent"].shape[1]
        )
    policy.eval()
    if alpha_net is not None:
        alpha_net.eval()

    aa = ad.read_h5ad(manifest["adata_path"])
    src = aa[aa.obs["sample"].astype(str) == a.source].copy()
    target_count_observed = int((aa.obs["sample"].astype(str) == a.target).sum())
    growth_target_count = int(a.growth_target_count if a.growth_target_count > 0 else 0)
    if not 0.0 <= float(a.ablation_fraction) <= 1.0:
        raise ValueError("--ablation-fraction must be between 0 and 1")

    ct = src.obs["cell_type"].astype(str).to_numpy()
    xy = np.c_[src.obs["cx_aligned"], src.obs["cy_aligned"]].astype(float)
    epi = np.flatnonzero(ct == "epicardial")
    gene_idx = src.var_names.get_indexer(["Wt1"])
    if int(gene_idx[0]) < 0:
        raise KeyError("The input AnnData does not contain the required Wt1 gene")
    wt1_col = src.layers["counts"][:, int(gene_idx[0])]
    wt1_expr = np.asarray(wt1_col.toarray()).ravel() if hasattr(wt1_col, "toarray") else np.asarray(wt1_col).ravel()
    wt1_expr = wt1_expr.astype(np.float32, copy=False)
    wt1_epi = wt1_expr[epi]
    if len(epi) == 0:
        raise ValueError("No E10.5 epicardial cells were found")

    # Monotone Wt1 propensity. The floor keeps low-Wt1 cells selectable.
    wmin = float(wt1_epi.min())
    wrange = float(wt1_epi.max() - wmin)
    wt1_score_epi = np.ones(len(epi), dtype=np.float64) if wrange <= 1e-12 else (
        0.25 + 0.75 * (wt1_epi.astype(np.float64) - wmin) / wrange
    )
    ablation_probability_epi = np.clip(
        float(a.ablation_fraction) * wt1_score_epi / max(float(wt1_score_epi.mean()), 1e-12),
        0.0, 1.0,
    ).astype(np.float32)

    # Weighted sampling without replacement: exact fraction, stochastic, reproducible.
    rng = np.random.default_rng(int(a.seed))
    n_ablate = int(round(float(a.ablation_fraction) * len(epi)))
    if n_ablate > 0:
        gumbel = rng.gumbel(size=len(epi))
        chosen_epi_local = np.argsort(np.log(wt1_score_epi) + gumbel)[-n_ablate:]
        targets = np.sort(epi[chosen_epi_local]).astype(np.int64)
    else:
        targets = np.empty(0, dtype=np.int64)

    ablation_probability = np.zeros(src.n_obs, dtype=np.float32)
    ablation_probability[epi] = ablation_probability_epi
    ablation_target = np.zeros(src.n_obs, dtype=bool)
    ablation_target[targets] = True
    ablation_status = np.full(src.n_obs, "not_epicardial", dtype=object)
    ablation_status[epi] = "retained_epicardial"
    ablation_status[targets] = "ablated"

    masked = src.copy()
    masked.obs["source_local_index"] = np.arange(masked.n_obs, dtype=np.int64)
    masked.obs["Wt1_expression"] = wt1_expr
    masked.obs["ablation_probability"] = ablation_probability
    masked.obs["ablation_target"] = ablation_target
    masked.obs["ablation_status"] = pd.Categorical(ablation_status)
    masked.obs["ablation_weight"] = np.float32(1.0)
    masked.obs.loc[masked.obs["ablation_target"], "ablation_weight"] = np.float32(0.0)
    masked.write_h5ad(out / "E10.5_counterfactual_partial_ablation_selection.h5ad")

    state0 = clone_state(pack.state0)
    n0 = int(state0["coords"].shape[0])
    state0["uid"] = torch.arange(n0, device=dev, dtype=torch.long)
    state0["parent_uid"] = torch.full((n0,), -1, device=dev, dtype=torch.long)
    state0["is_birth"] = torch.zeros(n0, device=dev, dtype=torch.bool)
    state0["born_step"] = torch.full((n0,), -1, device=dev, dtype=torch.int32)
    state0["has_divided"] = torch.zeros(n0, device=dev, dtype=torch.bool)
    state0["is_diff"] = torch.zeros(n0, device=dev, dtype=torch.bool)
    state0["diff_mid"] = torch.zeros(n0, device=dev, dtype=torch.float32)
    state0["diff_alpha"] = torch.zeros(n0, device=dev, dtype=torch.float32)
    state0["diff_tgt_layer"] = torch.full((n0,), -1, device=dev, dtype=torch.long)
    state0["diff_enter_step"] = torch.full((n0,), -100000, device=dev, dtype=torch.int32)
    state0["commit_step"] = torch.full((n0,), -1, device=dev, dtype=torch.int32)
    epi_layer_idx = int(ctx.layer_to_idx["epicardial"])
    fibro_layer_idx = int(ctx.layer_to_idx["fibroblasts"])
    state0["lr"] = pack.recompute_lr_fn(state0["latent"], state0["coords"])

    # Freeze policy calibration on the unperturbed E10.5 state. Both arms use it.
    grid0 = pack.grid_cache_dev[0]
    occ0, _ = s2.get_occ_and_crowd(state0["coords"], state0["layers"], grid0,
                                    pack.H, pack.W, pack.n_layers, cap=pack.shell_need_cap)
    density_map0 = occ0.sum(dim=0)
    density0 = s2.gather_map_at_coords_fast(
        density_map0, state0["coords"], grid0, pack.H, pack.W, default=0.0
    ).float() / max(float(pack.shell_need_cap), 1.0)
    lr_mu = state0["lr"].mean()
    lr_sd = state0["lr"].std().clamp_min(1e-6)
    with torch.no_grad():
        out0 = policy(state0["coords"], lr=((state0["lr"] - lr_mu) / lr_sd).view(-1, 1),
                      density=density0.view(-1, 1))
    logit_mu = out0[:, :2].mean(dim=0)
    logit_sd = out0[:, :2].std(dim=0).clamp_min(1e-6)

    # Coordinate jitter is based only on E10.5 nearest-neighbour spacing.
    with torch.no_grad():
        dmat = torch.cdist(state0["coords"], state0["coords"])
        dmat.fill_diagonal_(float("inf"))
        nn_scale = torch.quantile(dmat.min(dim=1).values, 0.5).clamp_min(1e-4)

    motion_model, latent_step_scale = fit_niche_motion(
        state0["latent"], state0["coords"], state0["lr"], nn_scale, dev, seed=a.seed
    )

    def simulation(label, ablate):
        state = clone_state(state0)
        next_uid = n0
        if ablate:
            kill_u = torch.tensor(targets, device=dev, dtype=torch.long)
            state = subset_state(state, ~torch.isin(state["uid"], kill_u))
            state["lr"] = pack.recompute_lr_fn(state["latent"], state["coords"])

        fields = {k: [] for k in [
            "coords", "layers", "latent", "lr", "uid", "parent_uid", "is_birth",
            "is_diff", "diff_mid", "diff_alpha", "diff_tgt_layer",
            "diff_enter_step", "commit_step",
        ]}
        rows = []

        def capture(frame):
            for k in fields:
                if k in ("coords", "latent", "lr", "diff_mid", "diff_alpha"):
                    typ = np.float32
                elif k in ("is_birth", "is_diff"):
                    typ = np.bool_
                else:
                    typ = np.int64
                fields[k].append(as_numpy(state[k], typ))
            lay = as_numpy(state["layers"], np.int64)
            for i, name in enumerate(ctx.layers_list):
                rows.append({"simulation": label, "frame": frame, "cell_type": name,
                             "count": int((lay == i).sum())})

        capture(0)
        for t in range(a.steps):
            # Free local state/motion update. This is the only developmental
            # motion source: current latent + local niche/LR + time. It does
            # not use target coordinates, target occupancy, teacher latent,
            # target KNN, or a target direction field.
            if state["coords"].shape[0] > 0:
                n_sub = max(int(a.message_passing_substeps), 1)
                for sub in range(n_sub):
                    # Decoder-backed LR is recomputed after every substep and
                    # therefore becomes the input to the next local update.
                    motion_x, _, dxy_niche = local_niche_features(
                        state["latent"], state["coords"], state["lr"],
                        (float(t) + float(sub) / n_sub) / max(a.steps, 1), k=16
                    )
                    with torch.no_grad():
                        dz_raw, dxy_raw = motion_model(motion_x)
                        dz = (
                            torch.tanh(dz_raw) * latent_step_scale.view(1, -1)
                            * (float(a.latent_update_gain) / n_sub)
                        )
                        dxy_direction = dxy_raw / dxy_raw.norm(dim=1, keepdim=True).clamp_min(1e-6)
                        coord_gap = dxy_niche.norm(dim=1, keepdim=True) / nn_scale.clamp_min(1e-6)
                        dxy = dxy_direction * (
                            (float(a.coord_step_fraction) / n_sub)
                            * nn_scale * torch.tanh(coord_gap)
                        )
                        state["latent"] = state["latent"] + dz
                        state["coords"] = state["coords"] + dxy
                    state["lr"] = pack.recompute_lr_fn(state["latent"], state["coords"])
            coords, layers, latent, lr = state["coords"], state["layers"], state["latent"], state["lr"]
            n = int(coords.shape[0])
            occ, _ = s2.get_occ_and_crowd(coords, layers, grid0, pack.H, pack.W,
                                           pack.n_layers, cap=pack.shell_need_cap)
            density_map = occ.sum(dim=0)
            density = s2.gather_map_at_coords_fast(
                density_map, coords, grid0, pack.H, pack.W, default=0.0
            ).float() / max(float(pack.shell_need_cap), 1.0)
            with torch.no_grad():
                pred = policy(coords, lr=((lr - lr_mu) / lr_sd).view(-1, 1), density=density.view(-1, 1))
            hz = torch.sigmoid((pred[:, :2] - logit_mu) / logit_sd)
            p_birth = a.birth_hazard_max * hz[:, 0]
            p_death = a.death_hazard_max * hz[:, 1]

            # Execute the transition-trained transition head without target guidance.
            # The only allowed transition here is epicardial -> fibroblasts.
            if alpha_net is not None and n > 0:
                with torch.no_grad():
                    can_start = (
                        (~state["is_diff"])
                        & (~state["is_birth"])
                        & (state["layers"] == epi_layer_idx)
                    )
                    diff_gen = torch.Generator(device=dev)
                    diff_gen.manual_seed(a.seed * 2000 + t + (200000 if ablate else 0))
                    p_diff = torch.sigmoid((pred[:, 2] - 0.2)).clamp(0.0, 1.0)
                    start_diff = can_start & (
                        torch.rand(n, generator=diff_gen, device=dev) < p_diff
                    )
                    if bool(start_diff.any()):
                        state["is_diff"] = state["is_diff"].clone()
                        state["diff_mid"] = state["diff_mid"].clone()
                        state["diff_alpha"] = state["diff_alpha"].clone()
                        state["diff_tgt_layer"] = state["diff_tgt_layer"].clone()
                        state["diff_enter_step"] = state["diff_enter_step"].clone()
                        state["commit_step"] = state["commit_step"].clone()
                        state["is_diff"][start_diff] = True
                        state["diff_mid"][start_diff] = 0.0
                        state["diff_alpha"][start_diff] = 0.0
                        state["diff_tgt_layer"][start_diff] = fibro_layer_idx
                        state["diff_enter_step"][start_diff] = t + 1
                        state["commit_step"][start_diff] = -1

                    active = state["is_diff"] & (
                        state["diff_tgt_layer"] == fibro_layer_idx
                    )
                    if bool(active.any()):
                        idx_active = active.nonzero(as_tuple=True)[0]
                        lr_alpha = ((lr - lr.mean()) / (lr.std() + 1e-6)).view(-1, 1)
                        raw_h = alpha_net(
                            coords[idx_active],
                            lr=lr_alpha[idx_active],
                            density=torch.log1p(density[idx_active]).view(-1, 1),
                        ).float()
                        h01 = torch.nn.functional.softplus(raw_h[:, 0])
                        h12 = torch.nn.functional.softplus(raw_h[:, 1])
                        age = (t + 1 - state["diff_enter_step"][idx_active].to(torch.int64)).clamp_min(0).float()
                        tau_gate = max(0.03 * max(a.steps, 1), 1.0)
                        gate12 = ((age - 1.0) / tau_gate).clamp(0.0, 1.0)
                        gate01 = (((age + 1.0) / tau_gate).clamp(0.0, 1.0)) ** 2
                        q01 = (1.0 - torch.exp(-h01)) * gate01
                        q12 = (1.0 - torch.exp(-h12)) * gate12
                        p_mid = state["diff_mid"][idx_active].clamp(0.0, 1.0)
                        p_tgt = (state["diff_alpha"][idx_active] - 0.5 * p_mid).clamp(0.0, 1.0)
                        p_src = (1.0 - p_mid - p_tgt).clamp(0.0, 1.0)
                        flow01 = p_src * q01
                        flow12 = p_mid * q12
                        p_mid_new = (p_mid + flow01 - flow12).clamp(0.0, 1.0)
                        p_tgt_new = (p_tgt + flow12).clamp(0.0, 1.0)
                        total_p = (1.0 - flow01 + flow12).clamp_min(1e-12)
                        p_mid_new = (p_mid_new / total_p).clamp(0.0, 1.0)
                        p_tgt_new = (p_tgt_new / total_p).clamp(0.0, 1.0)
                        alpha_new = (0.5 * p_mid_new + p_tgt_new).clamp(0.0, 1.0)
                        state["diff_mid"] = state["diff_mid"].clone()
                        state["diff_alpha"] = state["diff_alpha"].clone()
                        state["diff_mid"][idx_active] = p_mid_new
                        state["diff_alpha"][idx_active] = alpha_new

                        commit = (alpha_new >= 0.80) & (age >= 2.0)
                        if bool(commit.any()):
                            idx_commit = idx_active[commit]
                            state["layers"] = state["layers"].clone()
                            state["layers"][idx_commit] = fibro_layer_idx
                            state["is_diff"] = state["is_diff"].clone()
                            state["is_diff"][idx_commit] = False
                            state["diff_mid"] = state["diff_mid"].clone()
                            state["diff_mid"][idx_commit] = 0.0
                            state["diff_alpha"] = state["diff_alpha"].clone()
                            state["diff_alpha"][idx_commit] = 1.0
                            state["diff_tgt_layer"] = state["diff_tgt_layer"].clone()
                            state["diff_tgt_layer"][idx_commit] = -1
                            state["diff_enter_step"] = state["diff_enter_step"].clone()
                            state["diff_enter_step"][idx_commit] = -100000
                            state["commit_step"] = state["commit_step"].clone()
                            state["commit_step"][idx_commit] = t + 1

            gen = torch.Generator(device=dev)
            gen.manual_seed(a.seed * 1000 + t + (100000 if ablate else 0))
            die = torch.rand(n, generator=gen, device=dev) < p_death
            keep = ~die
            state = subset_state(state, keep)
            pred_surv = pred[keep]
            p_birth_surv = p_birth[keep]
            eligible = ~state["has_divided"]
            # Global developmental growth calibration only: policy scores choose
            # parents, but there is no type-specific or spatial target quota.
            if growth_target_count > 0:
                n_now = int(state["coords"].shape[0])
                n0_growth = int(state0["coords"].shape[0])
                desired_next = int(round(n0_growth * (growth_target_count / max(n0_growth, 1)) ** ((t + 1) / max(a.steps, 1))))
                births_needed = max(0, min(int(eligible.sum().item()), desired_next - n_now))
                eligible_idx = eligible.nonzero(as_tuple=True)[0]
                if births_needed > 0 and eligible_idx.numel() > 0:
                    weights = p_birth_surv[eligible_idx].clamp_min(1e-6)
                    parent = eligible_idx[torch.multinomial(weights, births_needed, replacement=False, generator=gen)]
                else:
                    parent = torch.empty(0, device=dev, dtype=torch.long)
            else:
                born_parent = eligible & (torch.rand(int(eligible.shape[0]), generator=gen, device=dev) < p_birth_surv)
                parent = born_parent.nonzero(as_tuple=True)[0]

            if parent.numel() > 0:
                nb = int(parent.numel())
                state["has_divided"] = state["has_divided"].clone()
                state["has_divided"][parent] = True
                mu = torch.tanh(pred_surv[parent, 2:4]) * nn_scale
                rho = torch.sigmoid(pred_surv[parent, 4:6]) * nn_scale
                eps = torch.randn((nb, 2), generator=gen, device=dev)
                new_coords = state["coords"][parent] + mu + rho * eps
                lat_sd = state["latent"].std(dim=0, keepdim=True).clamp_min(1e-6)
                new_latent = state["latent"][parent] + a.latent_noise * lat_sd * torch.randn(
                    (nb, state["latent"].shape[1]), generator=gen, device=dev
                )
                new_uid = torch.arange(next_uid, next_uid + nb, device=dev, dtype=torch.long)
                next_uid += nb
                additions = {
                    "coords": new_coords,
                    "layers": state["layers"][parent].clone(),
                    "latent": new_latent,
                    "uid": new_uid,
                    "parent_uid": state["uid"][parent].clone(),
                    "is_birth": torch.ones(nb, device=dev, dtype=torch.bool),
                    "born_step": torch.full((nb,), t + 1, device=dev, dtype=torch.int32),
                    "has_divided": torch.zeros(nb, device=dev, dtype=torch.bool),
                    "is_diff": torch.zeros(nb, device=dev, dtype=torch.bool),
                    "diff_mid": torch.zeros(nb, device=dev, dtype=torch.float32),
                    "diff_alpha": torch.zeros(nb, device=dev, dtype=torch.float32),
                    "diff_tgt_layer": torch.full((nb,), -1, device=dev, dtype=torch.long),
                    "diff_enter_step": torch.full((nb,), -100000, device=dev, dtype=torch.int32),
                    "commit_step": torch.full((nb,), -1, device=dev, dtype=torch.int32),
                }
                for k, v in additions.items():
                    state[k] = torch.cat([state[k], v], dim=0)

            # Existing latent and coordinates are not replaced or advected toward E12.5.
            state["lr"] = pack.recompute_lr_fn(state["latent"], state["coords"])
            capture(t + 1)

        fields["t"] = np.linspace(0, 1, a.steps + 1, dtype=np.float32)
        fields["T"] = a.steps
        fields["n_layers"] = pack.n_layers
        np.savez_compressed(out / f"{label}_simulation.npz", **{
            k: (np.asarray(v, dtype=object) if isinstance(v, list) else v) for k, v in fields.items()
        })
        return rows

    rows = simulation("ablation_free", True) if a.ablation_only else (
        simulation("normal_free", False) + simulation("ablation_free", True)
    )
    pd.DataFrame(rows).to_csv(out / "cell_counts_by_frame.csv", index=False)
    pd.DataFrame({
        "source_local_index": targets,
        "obs_name": src.obs_names[targets].astype(str),
        "cell_type": ct[targets], "x": xy[targets, 0], "y": xy[targets, 1],
        "Wt1_expression": wt1_expr[targets],
        "ablation_probability": ablation_probability[targets],
        "ablation_target": ablation_target[targets],
        "ablation_status": ablation_status[targets],
    }).to_csv(out / "ablation_targets.csv", index=False)
    pd.DataFrame({
        "source_local_index": np.arange(src.n_obs, dtype=np.int64),
        "obs_name": src.obs_names.astype(str),
        "cell_type": ct,
        "Wt1_expression": wt1_expr,
        "ablation_probability": ablation_probability,
        "ablation_target": ablation_target,
        "ablation_status": ablation_status,
    }).to_csv(out / "ablation_selection_by_cell.csv", index=False)
    config = vars(a).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config.update({
        "mode": "target-free exploratory counterfactual",
        "disabled": ["birth/death quota", "target occupancy", "normal advection", "teacher latent", "target latent KNN"],
        "ablation_targets": int(len(targets)),
        "ablation_fraction": float(a.ablation_fraction),
        "ablation_selection": "initial E10.5 epicardial only; Wt1-weighted stochastic sampling without replacement",
        "ablation_wt1_gene": "Wt1",
        "ablation_seed": int(a.seed),
        "transition_prior": manifest["transition_prior"],
        "transition_simulation_mode": "transition diff head + alpha_net executed; target-free transition to fibroblasts; no target quota/occupancy/teacher/KNN guidance",
        "transition_transition": "epicardial -> fibroblasts; start from policy diff logit; kinetics from alpha_net; commit when alpha>=0.8 and age>=2",
        "growth_target_count": int(growth_target_count),
        "growth_calibration": "global total-count schedule; no type/spatial target quota",
        "ablation_mode": a.ablation_mode,
        "fixed_policy_lr_mean": float(lr_mu.cpu()),
        "fixed_policy_lr_std": float(lr_sd.cpu()),
        "nearest_neighbor_scale": float(nn_scale.cpu()),
        "lr_latent_residual": "disabled; LR enters niche-motion MLP features and policy hazard features",
        "motion_model": "E10.5-only magnitude-aware latent MLP + normalized free coordinate update",
        "motion_inputs": "current latent + fixed-scale LR + local neighbor distance + local latent-niche difference + time",
        "motion_outputs": "magnitude-aware delta latent (10D) + normalized gap-limited delta xy (2D)",
        "message_passing_substeps": int(a.message_passing_substeps),
        "latent_update_gain": float(a.latent_update_gain),
        "motion_coord_step_max_per_frame": float((a.coord_step_fraction * nn_scale).cpu()),
        "motion_latent_step_scale_mean": float(latent_step_scale.mean().cpu()),
    })
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
