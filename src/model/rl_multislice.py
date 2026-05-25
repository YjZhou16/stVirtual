import gc, glob, os, re, json, inspect
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from matplotlib.path import Path as MplPath
from tqdm import trange
from geomloss import SamplesLoss
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from model.lr_dec import NoisyCountDecoder
from utils.compute_lr import compute_lr_potential_gpu
from utils.rl_utils import *


def _filter_kwargs(cls, kw):
    sig = inspect.signature(cls.__init__)
    ok = set(sig.parameters.keys()); ok.discard("self")
    return {k: v for k, v in kw.items() if k in ok}

# ============================================================
# 1) GrowthPolicyNet
# ============================================================
class GrowthPolicyNet(nn.Module):
    def __init__(self, n_layers: int, use_lr: bool = True, hidden_dim: int = 128, use_t: bool = True):
        super().__init__()
        self.n_layers = int(n_layers)
        self.use_lr = bool(use_lr)
        self.use_t = bool(use_t)

        input_dim = 3 + (1 if self.use_lr else 0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        coords,
        lr=None,
        t_frac=None,
        *,
        density=None,
        need=None,
        overlap=None,
        crowd=None,
    ):
        device = coords.device
        N = int(coords.shape[0])

        if density is None:
            density = torch.zeros((N, 1), device=device, dtype=torch.float32)
        else:
            density = density.to(device=device, dtype=torch.float32)
            if density.ndim == 1:
                density = density.view(N, 1)

        feats = [coords[:, :2].to(dtype=torch.float32), density]

        if self.use_lr:
            if lr is None:
                lr = torch.zeros((N, 1), device=device, dtype=torch.float32)
            else:
                lr = lr.to(device=device, dtype=torch.float32)
                if lr.ndim == 1:
                    lr = lr.view(N, 1)
            feats.append(lr)

        x = torch.cat(feats, dim=1)
        return self.net(x)

# =========================
# (C) SimulationEnv: step() 
# =========================
class SimulationEnv:
    def __init__(self, state0: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor], device: torch.device,
                 base_coords_seq: List[torch.Tensor], base_Z_seq: List[torch.Tensor], base_layers0: torch.Tensor,
                 grid_cache_np: list, grid_cache_dev: list, shells_norm: list, T: int, n_layers: int, H: int, W: int,
                 recompute_lr_fn, t0: int = 0, shell_need_cap: int = 2, use_lr: bool = True,
                 advect_dist_q: float = 0.90, advect_dist_min: float = 0.02, advect_dist_max: float = 0.35, advect_blend_temp: float = 0.02,
                 turnover_max_frac: float = 0.15, turnover_cap_frac: float = 0.20, turnover_p: float = 1.0, turnover_mismatch_mul: float = 0.5,
                 w_need: float = 1.0, w_overlap: float = 2.0, w_gap: float = 1.0, gap_ema_beta: float = 0.90):
        self.device = device
        self.base_coords_seq = base_coords_seq
        self.base_Z_seq = base_Z_seq
        self.base_layers0 = base_layers0.to(device)
        self.grid_cache_np = grid_cache_np
        self.grid_cache_dev = grid_cache_dev
        self.shells_norm = shells_norm
        self.shell_need_cap = int(shell_need_cap)
        self.T = int(T); self.t = int(t0)
        self.n_layers = int(n_layers); self.H = int(H); self.W = int(W)
        self.use_lr = bool(use_lr)
        self.recompute_lr = recompute_lr_fn

        self.advect_dist_q = float(advect_dist_q)
        self.advect_dist_min = float(advect_dist_min)
        self.advect_dist_max = float(advect_dist_max)
        self.advect_blend_temp = float(advect_blend_temp)

        self.turnover_max_frac = float(turnover_max_frac)
        self.turnover_cap_frac = float(turnover_cap_frac)
        self.turnover_p = float(turnover_p)
        self.turnover_mismatch_mul = float(turnover_mismatch_mul)

        self.w_need = float(w_need); self.w_overlap = float(w_overlap); self.w_gap = float(w_gap)
        self.gap_ema_beta = float(gap_ema_beta)
        self.gap_ema = torch.zeros((self.n_layers,), device=device, dtype=torch.float32)

        self.tgt_coords = target_state["coords"].to(device)
        self.tgt_layers = target_state["layers"].to(device)
        self.tgt_latent = target_state.get("latent", None); self.tgt_latent = None if self.tgt_latent is None else self.tgt_latent.to(device)
        self.tgt_is_new = target_state.get("is_new", None); self.tgt_is_new = None if self.tgt_is_new is None else self.tgt_is_new.to(device)

        self.state = {"coords": state0["coords"].to(device).clone(),
                      "layers": state0["layers"].to(device).clone(),
                      "anchor": state0["anchor"].to(device).clone()}
        lat0 = state0.get("latent", None); self.state["latent"] = None if lat0 is None else lat0.to(device).clone()
        ex0 = state0.get("expr_union", None); self.state["expr_union"] = None if ex0 is None else ex0.to(device).clone()
        lr0 = state0.get("lr", None)
        if lr0 is None: lr0 = torch.zeros((self.state["coords"].shape[0],), device=device, dtype=torch.float32)
        self.state["lr"] = lr0.to(device).clone()

        N0 = int(self.state["coords"].shape[0])
        self.state["is_birth"] = torch.zeros(N0, dtype=torch.bool, device=device)
        self.state["born_step"] = torch.full((N0,), -1, dtype=torch.int32, device=device)
        self.state["has_divided"] = torch.zeros(N0, dtype=torch.bool, device=device)

        # lineage ids
        self.state["uid"] = torch.arange(N0, device=device, dtype=torch.long)
        self.state["parent_uid"] = torch.full((N0,), -1, device=device, dtype=torch.long)
        self._next_uid = int(N0)

        self._tgt_occ_cache = {}

    def _get_occ_tgt_raw(self, ti: int) -> torch.Tensor:
        if ti not in self._tgt_occ_cache:
            grid_item = self.grid_cache_dev[int(ti)]
            v = grid_occupancy_by_layer(self.tgt_coords, self.tgt_layers, grid_item, self.H, self.W, self.n_layers, normalize=False)
            self._tgt_occ_cache[ti] = v
        return self._tgt_occ_cache[ti]

    def step(self, policy_net, birth_quota_layer: np.ndarray, death_quota_layer: np.ndarray, dir_step: int,
            advect_latent: bool = True, seed_step: int = 2025, *, TAU_BIRTH: float, TAU_DEATH: float, LATENT_NOISE_SCALE: float,
            NEED_GAMMA: float = 2.0, LOCAL_NEED_RADIUS: int = 16, PARENT_COOLDOWN: int = 0,
            BIRTH_HOT_FRAC: float = 0.6, CUR_DILATE_MAX: int = 3, LAMBDA_PARENT: float = 1.0, CROWD_DEATH_W: float = 2.0):
        dev = self.device; assert dir_step in (+1, -1)
        t_now = int(self.t); t_next = t_now + int(dir_step)
        if t_next < 0 or t_next > int(self.T):
            z0 = torch.zeros(1, device=dev).squeeze(); return z0, z0

        coords = self.state["coords"]; layers = self.state["layers"]; anchor = self.state["anchor"]; lr_vals = self.state["lr"]
        latent = self.state.get("latent", None); expr_union = self.state.get("expr_union", None)
        born_step = self.state["born_step"]; uid = self.state["uid"]; parent_uid = self.state["parent_uid"]
        has_divided = self.state.get("has_divided", torch.zeros((coords.shape[0],), dtype=torch.bool, device=dev))
        last_parent_step = self.state.get("last_parent_step", None)
        if last_parent_step is None or int(last_parent_step.numel()) != int(coords.shape[0]):
            last_parent_step = torch.full((coords.shape[0],), -100000, dtype=torch.int32, device=dev)

        N = int(coords.shape[0])
        if N <= 0:
            self.t = t_next; z0 = torch.zeros(1, device=dev).squeeze(); return z0, z0

        def _fix_centers_len(centers, ref_xy):
            if centers is None or (not torch.is_tensor(centers)) or centers.ndim != 2 or centers.shape[1] != 2: return ref_xy
            m = int(centers.shape[0]); n = int(ref_xy.shape[0])
            if m == n: return centers
            if m > n: return centers[:n]
            return torch.cat([centers, ref_xy[m:n]], dim=0)

        def _dilate_hw(mask_hw_bool: torch.Tensor, px: int):
            px = int(px)
            if px <= 0: return mask_hw_bool
            k = 2 * px + 1
            x = mask_hw_bool.to(torch.float32)[None, None, :, :]
            y = F.max_pool2d(x, kernel_size=k, stride=1, padding=px)
            return (y[0, 0] > 0)

        @torch.no_grad()
        def _knn_mean_target_latent(q_xy: torch.Tensor, li: int, *, k: int, chunk_q: int, max_ref: int, seed: int):
            if self.tgt_latent is None or self.tgt_latent.numel() == 0: return None
            ref_idx = (self.tgt_layers == int(li)).nonzero(as_tuple=True)[0]
            if ref_idx.numel() == 0: ref_idx = torch.arange(self.tgt_coords.shape[0], device=dev, dtype=torch.long)
            if ref_idx.numel() > max_ref:
                g = torch.Generator(device=dev); g.manual_seed(int(seed))
                ref_idx = ref_idx[torch.randperm(ref_idx.numel(), generator=g, device=dev)[:max_ref]]
            ref_xy = self.tgt_coords[ref_idx]
            ref_z = self.tgt_latent[ref_idx]
            kk = min(int(k), int(ref_xy.shape[0]))
            out = torch.empty((q_xy.shape[0], ref_z.shape[1]), device=dev, dtype=ref_z.dtype)
            for s in range(0, int(q_xy.shape[0]), int(chunk_q)):
                qq = q_xy[s:s + int(chunk_q)]
                d2 = (qq[:, None, :] - ref_xy[None, :, :]).pow(2).sum(dim=2)
                idx = torch.topk(d2, k=kk, largest=False).indices  # (B,kk)
                out[s:s + int(chunk_q)] = ref_z[idx].mean(dim=1)
            return out

        # ===== (0) plan quotas (keep turnover in plan) =====
        bq_plan_total = int(np.asarray(birth_quota_layer, np.int64).sum()) if birth_quota_layer is not None else 0
        dq_plan_total = int(np.asarray(death_quota_layer, np.int64).sum()) if death_quota_layer is not None else 0
        B_base = int(max(bq_plan_total, 0))
        D_base = int(max(dq_plan_total, 0)); D_base = min(D_base, N)
        deltaN = int(B_base - D_base)
        turn_plan = int(min(B_base, D_base))

        # ===== (1) advect coords (latent update moved to the end) =====
        base_t = self.base_coords_seq[t_now]; base_tp1 = self.base_coords_seq[t_next]
        try:
            coords_adv, alpha_knn = advect_coords_hybrid_soft(
                coords, anchor=anchor, base_t=base_t, base_tp1=base_tp1,
                dist_q=self.advect_dist_q, dist_min=self.advect_dist_min, dist_max=self.advect_dist_max,
                blend_temp=self.advect_blend_temp, knn_k=6, knn_max_ref=20000, knn_seed=100000 + int(seed_step), return_alpha=True
            )
            alpha_knn = alpha_knn.clamp(0.0, 1.0).view(-1)
        except TypeError:
            coords_adv = advect_coords_hybrid_soft(
                coords, anchor=anchor, base_t=base_t, base_tp1=base_tp1,
                dist_q=self.advect_dist_q, dist_min=self.advect_dist_min, dist_max=self.advect_dist_max,
                blend_temp=self.advect_blend_temp, knn_k=6, knn_max_ref=20000, knn_seed=100000 + int(seed_step)
            )
            alpha_knn = torch.zeros((N,), device=dev, dtype=torch.float32)

        is_new_now = (born_step.to(dev).view(-1) == int(t_now))
        if bool(is_new_now.any()): alpha_knn = torch.where(is_new_now, torch.ones_like(alpha_knn), alpha_knn)  # keep for debug
        coords = coords_adv

        # ===== (2) occupancy / need / overlap =====
        grid_item_dev = self.grid_cache_dev[t_next]
        occ_tgt_raw = self._get_occ_tgt_raw(t_next)
        allow_shell = grid_item_dev[4].bool()

        x0, y0, dx, dy = grid_item_dev[0], grid_item_dev[1], grid_item_dev[2], grid_item_dev[3]
        j = torch.round((coords[:, 0] - x0) / dx).to(torch.long)
        i = torch.round((coords[:, 1] - y0) / dy).to(torch.long)
        inside = (i >= 0) & (i < int(self.H)) & (j >= 0) & (j < int(self.W))
        ii, jj = i.clamp(0, int(self.H) - 1), j.clamp(0, int(self.W) - 1)
        good = inside & allow_shell[ii, jj]
        bad_idx = (~good).nonzero(as_tuple=True)[0]
        if bad_idx.numel() > 0:
            good_idx = good.nonzero(as_tuple=True)[0]
            if good_idx.numel() > 0:
                occ_good, _ = get_occ_and_crowd(coords[good_idx], layers[good_idx], grid_item_dev, self.H, self.W, self.n_layers, cap=self.shell_need_cap)
                tissue = (occ_good.sum(dim=0) > 0)
                tissue_d = _dilate_hw(tissue, px=2) & allow_shell
                ring = (tissue_d & (~tissue)) & allow_shell
                target_hw = ring if bool(ring.any()) else (tissue_d if bool(tissue_d.any()) else allow_shell)
            else:
                target_hw = allow_shell
            coords_bad = project_to_allowed_mask(coords[bad_idx], target_hw, grid_item_dev, self.H, self.W, chunk_q=4096, max_ref=12000, seed=int(123 + seed_step))
            coords = coords.clone(); coords[bad_idx] = coords_bad

        occ_cur_raw, crowd_all = get_occ_and_crowd(coords, layers, grid_item_dev, self.H, self.W, self.n_layers, cap=self.shell_need_cap)
        need_map = (occ_tgt_raw - occ_cur_raw).clamp_min(0.0)
        overlap_map = (occ_cur_raw - occ_tgt_raw).clamp_min(0.0)

        desired_layer = occ_tgt_raw.sum(dim=(1, 2)).float()
        cur_layer = occ_cur_raw.sum(dim=(1, 2)).float()
        gap = desired_layer - cur_layer
        beta = float(self.gap_ema_beta)
        self.gap_ema = beta * self.gap_ema + (1.0 - beta) * gap.detach()

        desired_total = float(desired_layer.sum().clamp_min(1.0).item())
        mismatch = float((need_map.sum() + overlap_map.sum()).detach().cpu().item()) / desired_total
        mismatch01 = float(np.clip(mismatch, 0.0, 1.0))
        t_frac = float(t_now) / max(int(self.T), 1)
        warm_start, warm_end = 0.10, 0.70
        warm = float(np.clip((t_frac - warm_start) / max(warm_end - warm_start, 1e-6), 0.0, 1.0)); warm = warm * warm

        # ===== (3) turnover trigger: include overcap (not only overlap) =====
        cap = float(self.shell_need_cap)
        total_occ = occ_cur_raw.sum(dim=0)
        overcap_map = (total_occ - cap).clamp_min(0.0)
        overcap = float(overcap_map.sum().detach().cpu().item())

        missing = float(need_map.sum().detach().cpu().item())
        excess = float(overlap_map.sum().detach().cpu().item())
        move_cap = int(max(0.0, round(min(missing, excess + overcap))))

        K_goal = int(round(move_cap * (0.10 + 0.90 * mismatch01)))
        K_cap = int(round(self.turnover_cap_frac * N))
        turn_sched = float(0.2 + 0.2 * t_frac ** 1.5)
        K_raw = int(round(turn_sched * K_goal))
        K_extra = int(np.clip(K_raw - turn_plan, 0, K_cap))

        D_total = min(N, D_base + K_extra)
        B_total = int(B_base + K_extra)

        # ===== (4) pointwise features for policy =====
        # ---- coords + density + lr ----
        total_occ_map = occ_cur_raw.sum(dim=0)  
        density_here = gather_map_at_coords_fast(
            total_occ_map, coords, grid_item_dev, self.H, self.W, default=0.0
        ).float()
        density_here = density_here / max(float(self.shell_need_cap), 1.0)

        lr_in = None
        if self.use_lr and (lr_vals is not None) and (lr_vals.numel() > 0):
            lr_z = (lr_vals - lr_vals.mean()) / (lr_vals.std() + 1e-6)
            lr_in = lr_z.detach().view(N, 1)

        out = policy_net(
            coords.detach(),
            lr=lr_in,
            density=density_here.view(N, 1),
        )

        need_here = torch.zeros((N,), device=dev, dtype=torch.float32)
        overlap_here = torch.zeros((N,), device=dev, dtype=torch.float32)
        for li in range(self.n_layers):
            idx = (layers == li).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            need_here[idx] = gather_map_at_coords_fast(
                need_map[li], coords[idx], grid_item_dev, self.H, self.W, default=0.0
            ).float()
            overlap_here[idx] = gather_map_at_coords_fast(
                overlap_map[li], coords[idx], grid_item_dev, self.H, self.W, default=0.0
            ).float()
        birth_logits = out[:, 0]; death_logits = out[:, 1]
        place_mu_raw = out[:, 2:4]; place_rho_raw = out[:, 4:6]
        ent = torch.zeros(1, device=dev)

        # ===== (5) death quotas: weight overlap + crowd + gap =====
        overlap_sum_layer = overlap_map.sum(dim=(1, 2)).float()
        cur_count_layer = torch.bincount(layers.long().clamp(0, self.n_layers - 1), minlength=self.n_layers).float()

        crowd_sum_layer = torch.zeros((self.n_layers,), device=dev, dtype=torch.float32)
        for li in range(self.n_layers):
            idx = (layers == li).nonzero(as_tuple=True)[0]
            if idx.numel() > 0: crowd_sum_layer[li] = crowd_all[idx].sum()

        wD = self.w_gap * (-self.gap_ema).clamp_min(0.0) + self.w_overlap * overlap_sum_layer + float(CROWD_DEATH_W) * crowd_sum_layer
        if float(wD.sum().item()) <= 1e-8: wD = (cur_count_layer > 0).float()
        dq_t = quota_from_weights(D_total, wD, max_cap=cur_count_layer.long())
        D_total = int(dq_t.sum().item())
        K_eff = int(D_total - D_base)
        B_total = int(B_base + max(K_eff, 0))

        # ===== (6) execute deaths =====
        death_logp = torch.zeros(1, device=dev)
        keep_mask = torch.ones(N, dtype=torch.bool, device=dev)
        if D_total > 0:
            idx_die_all, D_done = [], 0
            for li in range(self.n_layers):
                d_need = int(dq_t[li].item())
                if d_need <= 0: continue
                idx_li = (layers == li).nonzero(as_tuple=True)[0]
                if idx_li.numel() == 0: continue
                take = min(d_need, int(idx_li.numel()))
                ov = overlap_here[idx_li]; cr = crowd_all[idx_li]
                score = ov + float(CROWD_DEATH_W) * cr
                cand = idx_li[(score > 0)]
                if int(cand.numel()) < take:
                    k2 = min(int(max(take * 3, take)), int(idx_li.numel()))
                    cand = idx_li[torch.topk(score, k=k2, largest=True).indices]
                probs = F.softmax(death_logits[cand] / float(TAU_DEATH), dim=0)
                idx_local, logpD = sample_no_replace(probs, take)
                death_logp = death_logp + logpD
                if idx_local.numel() > 0: idx_die_all.append(cand[idx_local]); D_done += int(idx_local.numel())

            D_left = int(D_total - D_done)
            if D_left > 0:
                killed = torch.cat(idx_die_all, dim=0) if len(idx_die_all) else torch.empty(0, device=dev, dtype=torch.long)
                cand_mask = torch.ones(N, dtype=torch.bool, device=dev)
                if killed.numel() > 0: cand_mask[killed] = False
                idx_cand = cand_mask.nonzero(as_tuple=True)[0]
                if idx_cand.numel() > 0:
                    ov = overlap_here[idx_cand]; cr = crowd_all[idx_cand]
                    score = ov + float(CROWD_DEATH_W) * cr
                    cand = idx_cand[(score > 0)]
                    if int(cand.numel()) < D_left:
                        k2 = min(int(max(D_left * 3, D_left)), int(idx_cand.numel()))
                        cand = idx_cand[torch.topk(score, k=k2, largest=True).indices]
                    take = min(D_left, int(cand.numel()))
                    probs = F.softmax(death_logits[cand] / float(TAU_DEATH), dim=0)
                    idx_local, logpD = sample_no_replace(probs, take)
                    death_logp = death_logp + logpD
                    if idx_local.numel() > 0: idx_die_all.append(cand[idx_local])

            if len(idx_die_all) > 0:
                idx_die = torch.cat(idx_die_all, dim=0)
                keep_mask[idx_die] = False

        # ===== (7) survivors =====
        surv_idx = keep_mask.nonzero(as_tuple=True)[0]
        coords_surv = coords[surv_idx]; layers_surv = layers[surv_idx]; anchor_surv = anchor[surv_idx]
        birth_logits_surv = birth_logits[surv_idx]; mu_surv = place_mu_raw[surv_idx]; rho_surv = place_rho_raw[surv_idx]
        latent_surv = None if latent is None else latent[surv_idx]
        expr_surv = None if expr_union is None else expr_union[surv_idx]
        uid_surv = uid[surv_idx]; parent_uid_surv = parent_uid[surv_idx]
        born_step_surv = born_step[surv_idx].clone()
        has_div_surv = has_divided[surv_idx].clone()
        last_parent_surv = last_parent_step[surv_idx].clone()
        is_new_surv_now = is_new_now[surv_idx]

        # ===== (8) build birth-need map =====
        occ_cur_raw2, crowd_surv = get_occ_and_crowd(coords_surv, layers_surv, grid_item_dev, self.H, self.W, self.n_layers, cap=self.shell_need_cap)
        need_map2 = (occ_tgt_raw - occ_cur_raw2).clamp_min(0.0)

        total_occ2 = occ_cur_raw2.sum(dim=0)
        free_hw = (cap - total_occ2).clamp_min(0.0) / cap
        tgt_any_hw = (occ_tgt_raw.sum(dim=0) > 0)
        hole_hw = (free_hw > 0) & tgt_any_hw

        cur_allow = (occ_cur_raw2 > 0)
        dil_px = int(max(1, round(float(CUR_DILATE_MAX) * (1.0 - warm))))
        cur_allow_d = torch.zeros_like(cur_allow)
        for li in range(self.n_layers): cur_allow_d[li] = _dilate_hw(cur_allow[li], dil_px)

        tgt_allow = (occ_tgt_raw > 0)
        allow = cur_allow_d.float() + (1.0 - warm) * hole_hw.float().unsqueeze(0) + warm * tgt_allow.float()
        allow = allow.clamp(0.0, 1.0)
        need_map_birth = need_map2 * allow * (0.10 + free_hw).pow(2.0).unsqueeze(0)

        tissue_surv = (total_occ2 > 0)
        tissue_surv_d = _dilate_hw(tissue_surv, px=2) & allow_shell
        ring_surv = (tissue_surv_d & (~tissue_surv)) & allow_shell
        target_hw_surv = ring_surv if bool(ring_surv.any()) else (tissue_surv_d if bool(tissue_surv_d.any()) else allow_shell)

        # ===== (9) birth quotas by layer =====
        desired_layer2 = occ_tgt_raw.sum(dim=(1, 2)).float()
        cur_layer2 = torch.bincount(layers_surv.long().clamp(0, self.n_layers - 1), minlength=self.n_layers).float()
        gap2 = desired_layer2 - cur_layer2
        self.gap_ema = beta * self.gap_ema + (1.0 - beta) * gap2.detach()

        need_sum_layer = need_map_birth.sum(dim=(1, 2)).float()
        wB = self.w_gap * (self.gap_ema).clamp_min(0.0) + self.w_need * need_sum_layer
        if float(wB.sum().item()) <= 1e-8: wB = (desired_layer2 > 0).float().clamp_min(0.0)

        B_total = int(max(D_total + deltaN, 0))
        bq_t = quota_from_weights(B_total, wB, max_cap=None)

        # ===== (10) sample parents + centers =====
        birth_logp = torch.zeros(1, device=dev)
        centers_list, anchors_list, new_layers_list, mu_list, rho_list = [], [], [], [], []
        latent_src_list, expr_src_list, parent_mark_local_all, parent_uid_list = [], [], [], []
        Ns = int(coords_surv.shape[0])

        if B_total > 0 and Ns > 0:
            cool_ok = (t_now - last_parent_surv.to(torch.int32)) > int(PARENT_COOLDOWN)
            elig = cool_ok & (~is_new_surv_now)
            if bool(elig.any()):
                idx_elig = elig.nonzero(as_tuple=True)[0]
                B_done_by_layer = torch.zeros((self.n_layers,), device=dev, dtype=torch.long)
                hot_frac = float(BIRTH_HOT_FRAC) * (1.0 - warm) + 0.2 * warm
                for li in range(self.n_layers):
                    b_need = int(bq_t[li].item())
                    if b_need <= 0: continue
                    idx_li = idx_elig[(layers_surv[idx_elig] == li)]
                    if idx_li.numel() == 0: continue
                    take = min(b_need, int(idx_li.numel()))

                    pneed = gather_map_at_coords_fast(need_map_birth[li], coords_surv[idx_li], grid_item_dev, self.H, self.W, default=0.0).float()
                    logits_adj = birth_logits_surv[idx_li] + float(LAMBDA_PARENT) * torch.log1p(pneed)
                    probsB = F.softmax(logits_adj / float(TAU_BIRTH), dim=0)

                    parent_local, logpB = sample_no_replace(probsB, take)
                    birth_logp = birth_logp + logpB
                    if parent_local.numel() == 0: continue
                    parent_idx = idx_li[parent_local]

                    gen = torch.Generator(device=dev); gen.manual_seed(int(700000 + seed_step + 1000 * li))
                    perm = torch.randperm(parent_idx.numel(), generator=gen, device=dev)
                    parent_idx = parent_idx[perm]
                    cnt = int(parent_idx.numel()); cnt_hot = int(round(cnt * hot_frac)); cnt_loc = cnt - cnt_hot
                    idx_loc = parent_idx[:cnt_loc]; idx_hot = parent_idx[cnt_loc:]
                    parent_idx_all = torch.cat([idx_loc, idx_hot], dim=0) if (cnt_loc > 0 and cnt_hot > 0) else (idx_loc if cnt_hot == 0 else idx_hot)
                    parent_mark_local_all.append(parent_idx_all); B_done_by_layer[li] += int(parent_idx_all.numel())

                    rad_eff = int(LOCAL_NEED_RADIUS)
                    gam_eff = float(NEED_GAMMA) * (0.50 + 0.50 * warm)

                    if cnt_loc > 0:
                        pc = coords_surv[idx_loc]; pl = layers_surv[idx_loc]
                        cent_loc = sample_local_centers(parent_coords=pc, parent_layers=pl, need_map_by_layer=need_map_birth,
                                                        grid_item_dev=grid_item_dev, H=self.H, W=self.W, radius=rad_eff, gamma=gam_eff,
                                                        seed=int(100000 + seed_step + 1000 * li))
                        cent_loc = _fix_centers_len(cent_loc, pc)
                    else:
                        cent_loc = torch.empty((0, 2), device=dev, dtype=coords_surv.dtype)

                    if cnt_hot > 0:
                        cent_hot = sample_centers_map(need_map_birth[li], grid_item_dev, self.H, self.W, cnt_hot,
                                                    gamma=gam_eff * 1.5, seed=int(110000 + seed_step + 1000 * li), smooth_k=3)
                        if cent_hot is None or int(cent_hot.shape[0]) != cnt_hot: cent_hot = coords_surv[idx_hot].clone()
                    else:
                        cent_hot = torch.empty((0, 2), device=dev, dtype=coords_surv.dtype)

                    centers_all = torch.cat([cent_loc, cent_hot], dim=0) if (cnt_loc > 0 and cnt_hot > 0) else (cent_loc if cnt_hot == 0 else cent_hot)
                    centers_all = _fix_centers_len(centers_all, coords_surv[parent_idx_all])

                    need_at = gather_map_at_coords_fast(need_map_birth[li], centers_all, grid_item_dev, self.H, self.W, default=0.0).float()
                    bad = (need_at <= 0.0)
                    if bool(bad.any()):
                        cnt_bad = int(bad.sum().item())
                        cen2 = sample_centers_map(need_map_birth[li], grid_item_dev, self.H, self.W, cnt_bad,
                                                gamma=gam_eff * 1.5, seed=int(120000 + seed_step + 1000 * li), smooth_k=3)
                        if cen2 is not None:
                            cen2 = _fix_centers_len(cen2, centers_all[bad])
                            centers_all = centers_all.clone(); centers_all[bad] = cen2

                    centers_list.append(centers_all)
                    anchors_list.append(anchor_surv[parent_idx_all])
                    new_layers_list.append(layers_surv[parent_idx_all].clone())
                    mu_list.append(mu_surv[parent_idx_all]); rho_list.append(rho_surv[parent_idx_all])
                    parent_uid_list.append(uid_surv[parent_idx_all].clone())
                    if latent_surv is not None: latent_src_list.append(latent_surv[parent_idx_all])
                    if expr_surv is not None: expr_src_list.append(expr_surv[parent_idx_all])

                left = (bq_t - B_done_by_layer).clamp_min(0)
                if int(left.sum().item()) > 0:
                    genA = torch.Generator(device=dev); genA.manual_seed(int(900000 + seed_step))
                    for li in range(self.n_layers):
                        cnt = int(left[li].item())
                        if cnt <= 0: continue
                        pool = (self.base_layers0 == li).nonzero(as_tuple=True)[0]
                        if pool.numel() == 0: pool = torch.arange(self.base_layers0.numel(), device=dev)
                        ridx = torch.randint(0, pool.numel(), (cnt,), generator=genA, device=dev)
                        seed_anchor_li = pool[ridx].long()
                        gam_eff = float(NEED_GAMMA) * (0.50 + 0.50 * warm)
                        cen = sample_centers_map(need_map_birth[li], grid_item_dev, self.H, self.W, cnt,
                                                gamma=gam_eff * 1.5, seed=int(910000 + seed_step + 1000 * li), smooth_k=3)
                        if cen is None or int(cen.shape[0]) != cnt: cen = self.base_coords_seq[t_next][seed_anchor_li].clone()
                        centers_list.append(cen); anchors_list.append(seed_anchor_li)
                        new_layers_list.append(torch.full((cnt,), li, dtype=torch.long, device=dev))
                        mu_list.append(torch.zeros((cnt, 2), device=dev, dtype=coords.dtype))
                        rho_list.append(torch.zeros((cnt, 2), device=dev, dtype=coords.dtype))
                        parent_uid_list.append(torch.full((cnt,), -1, device=dev, dtype=torch.long))
                        if latent_surv is not None and latent_surv.numel() > 0:
                            src_pool = (layers_surv == li).nonzero(as_tuple=True)[0]
                            if src_pool.numel() == 0: src_pool = torch.arange(layers_surv.numel(), device=dev)
                            ridx2 = torch.randint(0, src_pool.numel(), (cnt,), generator=genA, device=dev)
                            latent_src_list.append(latent_surv[src_pool[ridx2]])
                        if expr_surv is not None and expr_surv.numel() > 0:
                            src_pool = (layers_surv == li).nonzero(as_tuple=True)[0]
                            if src_pool.numel() == 0: src_pool = torch.arange(layers_surv.numel(), device=dev)
                            ridx2 = torch.randint(0, src_pool.numel(), (cnt,), generator=genA, device=dev)
                            expr_src_list.append(expr_surv[src_pool[ridx2]])

        # ===== (11) materialize births =====
        did_birth = (len(centers_list) > 0)
        disp = (self.base_coords_seq[t_next] - self.base_coords_seq[t_now])

        if not did_birth:
            next_coords, next_layers, next_anchor = coords_surv, layers_surv, anchor_surv
            next_is_birth = torch.zeros(next_coords.shape[0], dtype=torch.bool, device=dev)
            next_latent, next_expr = latent_surv, expr_surv
            next_born_step = born_step_surv
            next_uid = uid_surv; next_parent_uid = parent_uid_surv
            next_has_div = has_div_surv
            next_last_parent = last_parent_surv
        else:
            centers_all = torch.cat(centers_list, dim=0)
            new_anchor = torch.cat(anchors_list, dim=0)
            new_layers = torch.cat(new_layers_list, dim=0)
            mu_all = torch.cat(mu_list, dim=0)
            rho_all = torch.cat(rho_list, dim=0)
            new_parent_uid = torch.cat(parent_uid_list, dim=0) if parent_uid_list else torch.full((new_anchor.numel(),), -1, device=dev, dtype=torch.long)

            Bc, Bm, Br, Ba = int(centers_all.shape[0]), int(mu_all.shape[0]), int(rho_all.shape[0]), int(new_anchor.shape[0])
            B = min(Bc, Bm, Br, Ba)
            if Bc != B: centers_all = centers_all[:B]
            if Bm != B: mu_all = mu_all[:B]
            if Br != B: rho_all = rho_all[:B]
            if Ba != B: new_anchor = new_anchor[:B]
            if int(new_layers.shape[0]) != B: new_layers = new_layers[:B]
            if int(new_parent_uid.shape[0]) != B: new_parent_uid = new_parent_uid[:B]

            nb = int(new_anchor.numel())
            new_uid = torch.arange(self._next_uid, self._next_uid + nb, device=dev, dtype=torch.long)
            self._next_uid += nb

            new_coords, logp_place = sample_birth_locations(
                centers=centers_all, mu_raw=mu_all, rho_raw=rho_all, dir_vec=disp[new_anchor], dx=dx, dy=dy, seed=int(2025 + int(seed_step))
            )
            birth_logp = birth_logp + logp_place

            jn = torch.round((new_coords[:, 0] - x0) / dx).to(torch.long)
            in_ = torch.round((new_coords[:, 1] - y0) / dy).to(torch.long)
            inside_n = (in_ >= 0) & (in_ < int(self.H)) & (jn >= 0) & (jn < int(self.W))
            iin, jjn = in_.clamp(0, int(self.H) - 1), jn.clamp(0, int(self.W) - 1)
            good_n = inside_n & allow_shell[iin, jjn]
            bad_n_idx = (~good_n).nonzero(as_tuple=True)[0]
            if bad_n_idx.numel() > 0:
                new_bad = project_to_allowed_mask(new_coords[bad_n_idx], target_hw_surv, grid_item_dev, self.H, self.W,
                                                chunk_q=4096, max_ref=12000, seed=int(456 + seed_step))
                new_coords = new_coords.clone(); new_coords[bad_n_idx] = new_bad

            next_latent = latent_surv
            if latent_surv is not None and len(latent_src_list) > 0:
                latent_src = torch.cat(latent_src_list, dim=0)
                if int(latent_src.shape[0]) != nb: latent_src = latent_src[:nb]
                if latent_src.numel() > 0:
                    lat_std = latent_surv.std(dim=0, keepdim=True).clamp_min(1e-6)
                    noise = torch.randn_like(latent_src) * (float(LATENT_NOISE_SCALE) * lat_std)
                    next_latent = torch.cat([latent_surv, latent_src + noise], dim=0)

            next_expr = expr_surv
            if expr_surv is not None and len(expr_src_list) > 0:
                expr_src = torch.cat(expr_src_list, dim=0)
                if int(expr_src.shape[0]) != nb: expr_src = expr_src[:nb]
                if expr_src.numel() > 0: next_expr = torch.cat([expr_surv, expr_src], dim=0)

            next_coords = torch.cat([coords_surv, new_coords], dim=0)
            next_layers = torch.cat([layers_surv, new_layers], dim=0)
            next_anchor = torch.cat([anchor_surv, new_anchor], dim=0)
            next_is_birth = torch.cat([torch.zeros(coords_surv.shape[0], dtype=torch.bool, device=dev),
                                    torch.ones(new_coords.shape[0], dtype=torch.bool, device=dev)], dim=0)
            next_born_step = torch.cat([born_step_surv, torch.full((new_coords.shape[0],), int(t_next), device=dev, dtype=torch.int32)], dim=0)

            next_has_div = torch.cat([has_div_surv, torch.zeros((new_coords.shape[0],), dtype=torch.bool, device=dev)], dim=0)
            next_last_parent = torch.cat([last_parent_surv, torch.full((new_coords.shape[0],), -100000, dtype=torch.int32, device=dev)], dim=0)
            if len(parent_mark_local_all) > 0:
                parent_mark_local = torch.cat(parent_mark_local_all, dim=0)
                if parent_mark_local.numel() > 0:
                    next_has_div[parent_mark_local] = True
                    next_last_parent[parent_mark_local] = int(t_now)

            next_uid = torch.cat([uid_surv, new_uid], dim=0)
            next_parent_uid = torch.cat([parent_uid_surv, new_parent_uid], dim=0)

        # ===== (11.5) latent update (as requested) =====
        # old (born_step<0): update by anchor-teacher at t_next
        # born (born_step>=0): blend toward target latent (KNN within same layer)
        if advect_latent and (next_latent is not None) and (next_latent.numel() > 0):
            teacher_all = self.base_Z_seq[t_next][next_anchor]
            legacy_mask = (next_born_step < 0)
            if bool(legacy_mask.any()):
                next_latent = next_latent.clone()
                next_latent[legacy_mask] = teacher_all[legacy_mask]

            born_mask = (next_born_step >= 0)
            if bool(born_mask.any()) and (self.tgt_latent is not None) and (self.tgt_latent.numel() > 0):
                alpha0 = float(getattr(self, "newborn_target_alpha0", 0.97))  # smaller -> faster to target
                knn_k = int(getattr(self, "newborn_target_knn_k", 16))
                max_ref = int(getattr(self, "newborn_target_knn_max_ref", 20000))
                chunk_q = int(getattr(self, "newborn_target_knn_chunk_q", 2048))

                idx_born = born_mask.nonzero(as_tuple=True)[0]
                # age at t_next: newborn created this step has age=0, older births age>=1
                age = (int(t_next) - next_born_step[idx_born].to(torch.int64)).clamp_min(0).to(torch.float32)
                a_cell = torch.clamp(torch.pow(torch.tensor(alpha0, device=dev), age + 1.0), 0.0, 1.0).to(next_latent.dtype)

                next_latent = next_latent.clone()
                for li in range(self.n_layers):
                    idx_li = idx_born[(next_layers[idx_born] == li)]
                    if idx_li.numel() == 0: continue
                    z_tgt = _knn_mean_target_latent(next_coords[idx_li], li, k=knn_k, chunk_q=chunk_q, max_ref=max_ref, seed=int(91000 + seed_step + 77 * li))
                    if z_tgt is None: continue
                    a = a_cell[(next_layers[idx_born] == li)].view(-1, 1)
                    next_latent[idx_li] = next_latent[idx_li] * a + z_tgt.to(next_latent.dtype) * (1.0 - a)

        # ===== (12) recompute lr + commit state =====
        if (not self.use_lr) or (self.recompute_lr is None):
            next_lr = torch.zeros((next_coords.shape[0],), device=dev, dtype=torch.float32)
        else:
            next_lr = self.recompute_lr(next_expr, next_coords) if (next_expr is not None) else self.recompute_lr(next_latent, next_coords)

        self.state["coords"] = next_coords; self.state["layers"] = next_layers; self.state["anchor"] = next_anchor; self.state["lr"] = next_lr
        self.state["is_birth"] = next_is_birth; self.state["latent"] = next_latent; self.state["expr_union"] = next_expr
        self.state["born_step"] = next_born_step; self.state["has_divided"] = next_has_div; self.state["uid"] = next_uid; self.state["parent_uid"] = next_parent_uid
        self.state["last_parent_step"] = next_last_parent
        self.t = t_next

        if hasattr(self, "trace") and isinstance(self.trace, list):
            self.trace.append({"t": int(self.t), "uid": next_uid.detach().cpu(), "parent_uid": next_parent_uid.detach().cpu(),
                            "born_step": next_born_step.detach().cpu(), "anchor": next_anchor.detach().cpu(),
                            "layers": next_layers.detach().cpu(), "coords": next_coords.detach().cpu(),
                            "is_birth": next_is_birth.detach().cpu()})
        return (birth_logp + death_logp).squeeze(), ent.squeeze()
    
# ============================================================
# 2) TrainingRL
# ============================================================
def train_policy_rl(*, state0, target_state, shells_norm, coords_seq_torch, Z_seq_torch, base_layers0,
                    grid_cache_np, grid_cache_dev, B_step_layer, D_step_layer, T, device, n_layers, H, W, recompute_lr_fn,
                    policy_cls, env_cls, USE_LR=True, EPOCHS=300, LR=1e-4,
                    W_ENT=0.1, EMA_BETA=0.9, ADVECT_LATENT=True, shell_need_cap=3, hidden_dim=128,
                    best_ckpt_path=None, save_best=True, empty_cache_every=1, return_history=True,
                    W_XY_TGT=2.0, W_Z_TGT=0.05,
                    W_OCC=0.1, W_OCC_IOU=0.1,
                    TAU_BIRTH=1.0, TAU_DEATH=1.0, LATENT_NOISE_SCALE=0.02,
                    OLD_MASS_SCALE=0.1):
    device=torch.device(device)
    kw=dict(n_layers=int(n_layers), use_lr=bool(USE_LR), hidden_dim=int(hidden_dim), use_t=True)
    policy_net=policy_cls(**_filter_kwargs(policy_cls, kw)).to(device)
    optimizer=torch.optim.Adam(policy_net.parameters(), lr=float(LR))
    best_reward=-1e18; reward_ema=None
    best_ckpt_path=str(best_ckpt_path) if best_ckpt_path is not None else None
    history={"reward": [], "adv": [], "tgt": [], "occ": [], "iou": [], "ent": [], "logp": []} if return_history else None

    uot_xy=SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.9, backend="multiscale")
    uot_z =SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.9, backend="online")

    tgt_is_new=target_state.get("is_new", None)
    if tgt_is_new is not None: tgt_is_new=tgt_is_new.to(device).bool()

    xt=target_state["coords"].to(device); lt=target_state["layers"].to(device)
    zt=target_state.get("latent", None); zt=None if zt is None else zt.to(device)
    x0=state0["coords"].to(device)

    grid_item_T=grid_cache_dev[int(T)]  

    pbar=trange(1, int(EPOCHS)+1, desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        env_f=env_cls(state0=state0, target_state=target_state, device=device,
                      base_coords_seq=coords_seq_torch, base_Z_seq=Z_seq_torch, base_layers0=base_layers0,
                      grid_cache_np=grid_cache_np, grid_cache_dev=grid_cache_dev, shells_norm=shells_norm,
                      T=int(T), n_layers=int(n_layers), H=int(H), W=int(W), recompute_lr_fn=recompute_lr_fn,
                      t0=0, shell_need_cap=int(shell_need_cap), use_lr=bool(USE_LR))
        logp_f_list=[]; ent_f_list=[]
        for t in range(int(T)):
            logp, ent=env_f.step(policy_net,
                                 birth_quota_layer=B_step_layer[t], death_quota_layer=D_step_layer[t],
                                 dir_step=+1, advect_latent=bool(ADVECT_LATENT), seed_step=int(epoch*1000+t),
                                 TAU_BIRTH=float(TAU_BIRTH), TAU_DEATH=float(TAU_DEATH),
                                 LATENT_NOISE_SCALE=float(LATENT_NOISE_SCALE))
            logp_f_list.append(logp); ent_f_list.append(ent)

        with torch.no_grad():

            xs=env_f.state["coords"]; ls=env_f.state["layers"]
            zs=env_f.state.get("latent", None)

            born_step=env_f.state.get("born_step", None)
            new_s=(born_step >= 0) if born_step is not None else env_f.state.get("is_birth", torch.zeros(xs.shape[0], device=device, dtype=torch.bool)).bool()

            if tgt_is_new is not None:
                new_t=tgt_is_new
            else:
                N_new_expected=max(int(xt.shape[0]-x0.shape[0]), 0)
                if N_new_expected <= 0:
                    new_t=torch.zeros((xt.shape[0],), device=device, dtype=torch.bool)
                else:
                    mind=torch.full((xt.shape[0],), 1e9, device=device)
                    chunk=2048
                    for s in range(0, xt.shape[0], chunk):
                        D=torch.cdist(xt[s:s+chunk], x0)
                        mind[s:s+chunk]=D.min(dim=1).values
                    _, idx=torch.topk(mind, k=min(N_new_expected, xt.shape[0]), largest=True)
                    new_t=torch.zeros((xt.shape[0],), device=device, dtype=torch.bool); new_t[idx]=True

            ms=torch.ones((xs.shape[0],), device=device, dtype=torch.float32)
            mt=torch.ones((xt.shape[0],), device=device, dtype=torch.float32)
            ms[~new_s]=float(OLD_MASS_SCALE); mt[~new_t]=float(OLD_MASS_SCALE)
            ms=ms/ms.sum().clamp_min(1e-12); mt=mt/mt.sum().clamp_min(1e-12)
            
            # uot loss
            tgt_xy=uot_xy(ms, xs, mt, xt).view(1)
            tgt_z =uot_z(ms, zs, mt, zt).view(1) if (zs is not None and zt is not None) else torch.zeros(1, device=device)
            tgt_loss=float(W_XY_TGT)*tgt_xy + float(W_Z_TGT)*tgt_z

            # per-layer occupancy penalty
            occ_cur_cap, _ = get_occ_and_crowd(xs, ls, grid_item_T, int(H), int(W), int(n_layers), cap=int(shell_need_cap))
            occ_tgt_raw = grid_occupancy_by_layer(xt, lt, grid_item_T, int(H), int(W), int(n_layers), normalize=False)
            occ_tgt_cap = occ_tgt_raw.clamp_max(float(shell_need_cap))
            occ_l1 = (occ_cur_cap - occ_tgt_cap).abs().sum() / occ_tgt_cap.sum().clamp_min(1.0)

            iou_loss=torch.zeros(1, device=device)
            if float(W_OCC_IOU) > 0:
                cur_bin=(occ_cur_cap.sum(dim=0) > 0).float()
                tgt_bin=(occ_tgt_cap.sum(dim=0) > 0).float()
                inter=(cur_bin*tgt_bin).sum()
                uni=((cur_bin+tgt_bin) > 0).float().sum().clamp_min(1.0)
                iou=inter/uni
                iou_loss=(1.0 - iou).view(1)
            occ_loss = occ_l1.view(1) + float(W_OCC_IOU)*iou_loss
            final_loss = tgt_loss + float(W_OCC)*occ_loss
            reward = -final_loss

        r_det=reward.detach()
        reward_ema=r_det if reward_ema is None else float(EMA_BETA)*reward_ema + (1.0-float(EMA_BETA))*r_det
        adv=(reward - reward_ema).detach()

        logp_total=torch.stack(logp_f_list).sum()
        ent_total=torch.stack(ent_f_list).sum()
        loss= -logp_total*adv - float(W_ENT)*ent_total
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            r_val=float(reward.item())
            if r_val > best_reward:
                best_reward=r_val
                if save_best and (best_ckpt_path is not None):
                    p_args={"n_layers": int(n_layers), "use_lr": bool(USE_LR), "use_t": True, "hidden_dim": int(hidden_dim)}
                    torch.save({"epoch": int(epoch), "model_state": policy_net.state_dict(),
                                "best_reward": float(best_reward), "grid_size": (int(H), int(W)),
                                "advect_latent": bool(ADVECT_LATENT), "T": int(T),
                                "use_lr": bool(USE_LR), "policy_args": p_args}, best_ckpt_path)

        if return_history:
            history["reward"].append(float(reward.item()))
            history["adv"].append(float(adv.item()))
            history["tgt"].append(float(tgt_loss.item()))
            history["occ"].append(float(occ_l1.item()))
            history["iou"].append(float((1.0 - iou_loss).item()) if float(W_OCC_IOU) > 0 else float("nan"))
            history["ent"].append(float(ent_total.item()))
            history["logp"].append(float(logp_total.item()))

        pbar.set_postfix({"rew": f"{float(reward.item()):.4f}", "best": f"{best_reward:.4f}",
                          "tgt": f"{float(tgt_loss.item()):.4f}", "occ": f"{float(occ_l1.item()):.4f}"})

        del env_f, logp_f_list, ent_f_list, loss, reward
        if empty_cache_every and empty_cache_every > 0 and (epoch % int(empty_cache_every) == 0):
            torch.cuda.empty_cache(); gc.collect()

    out={"policy": policy_net, "best_reward": float(best_reward), "best_ckpt_path": best_ckpt_path}
    if return_history: out["history"]=history
    return out

# -------------------- orchestration --------------------
@dataclass
class GlobalCtx:
    device: torch.device
    adata_all: sc.AnnData
    lr_pairs: pd.DataFrame
    xy_mu: np.ndarray
    xy_s: np.ndarray
    layers_list: List[str]
    layer_to_idx: Dict[str, int]
    latent_cache: dict
    layer_col: str

@dataclass
class StageCfg:
    src: str
    tgt: str
    out_npz_path: str
    bound_dir: str
    stat_json: Optional[str] = None
    ckpt_path_dec: Optional[str] = None
    scanvi_dir: Optional[str] = None
    model_type: str = "scanvi"
    latent_fallback_obsm: str = "X_scVI"
    latent_fallback_layer: Optional[str] = None
    z_csv_offset: int = 0
    layer_col: Optional[str] = None
    use_latent: bool = True
    use_lr: bool = True
    lr_source: str = "decoder"
    counts_layer: str = "counts"

@dataclass
class StagePack:
    cfg: StageCfg
    n_layers: int
    T: int
    H: int
    W: int
    shells_norm: List[np.ndarray]
    coords_seq_torch: List[torch.Tensor]
    Z_seq_torch: List[torch.Tensor]
    base_layers0: torch.Tensor
    grid_cache_np: list
    grid_cache_dev: list
    B_step_layer: np.ndarray
    D_step_layer: np.ndarray
    state0: Dict[str, torch.Tensor]
    target_state: Dict[str, torch.Tensor]
    recompute_lr_fn: Any
    shell_need_cap: int = 3

def build_global_ctx(*, adata_path: str, lr_pairs_path: str, ckpt_3dslice: str, device: torch.device, layer_col: str) -> GlobalCtx:
    ckpt = torch.load(ckpt_3dslice, map_location="cpu")
    adata_all = sc.read_h5ad(adata_path)
    lr_pairs = pd.read_csv(lr_pairs_path)
    xy_mu, xy_s = get_xy_norm_from_ckpt(ckpt)
    layers_all = adata_all.obs[layer_col].astype(str).values
    layers_list = sorted(pd.unique(layers_all).tolist())
    layer_to_idx = {name: i for i, name in enumerate(layers_list)}
    return GlobalCtx(device=device, adata_all=adata_all, lr_pairs=lr_pairs, xy_mu=xy_mu, xy_s=xy_s,
                     layers_list=layers_list, layer_to_idx=layer_to_idx, latent_cache={}, layer_col=str(layer_col))

def get_slice(ctx: GlobalCtx, sample_id: str, sample_key: str = "sample") -> sc.AnnData:
    m = ctx.adata_all.obs[sample_key].astype(str) == str(sample_id)
    return ctx.adata_all[m].copy()

def get_layer_idx(ctx: GlobalCtx, ad: sc.AnnData, layer_col: str) -> np.ndarray:
    lab = ad.obs[layer_col].astype(str).to_numpy()
    return np.array([ctx.layer_to_idx.get(s, 0) for s in lab], dtype=np.int64)

def prepare_one_stage(
    ctx, cfg: StageCfg, *, layer_col: str, sample_key: str = "sample",
    x_key: str = "cx_aligned", y_key: str = "cy_aligned",
    GRID_MARGIN: float = 0.02, GRID_HW_MAX: int = 512, GRID_BASE: int = 4,
    DEATH_FRAC: Optional[float] = 0.1, TURNOVER_FRAC: Optional[float] = 0.0,
    SHELL_NEED_CAP: int = 2, name_col: Optional[str] = "orig_name",
    **kwargs
):
    device = ctx.device
    ad_src = get_slice(ctx, f"{cfg.src}", sample_key=sample_key)
    ad_tgt = get_slice(ctx, f"{cfg.tgt}", sample_key=sample_key)
    xysrc_norm = normalize_xy_from_obs(ad_src, x_key=x_key, y_key=y_key, xy_mu=ctx.xy_mu, xy_s=ctx.xy_s)
    xytgt_norm = normalize_xy_from_obs(ad_tgt, x_key=x_key, y_key=y_key, xy_mu=ctx.xy_mu, xy_s=ctx.xy_s)

    layer_idx_src = get_layer_idx(ctx, ad_src, layer_col=layer_col)
    layer_idx_tgt = get_layer_idx(ctx, ad_tgt, layer_col=layer_col)
    n_layers = len(ctx.layers_list)
    base_layers0 = torch.from_numpy(layer_idx_src).to(device=device, dtype=torch.long)

    out_npz = np.load(cfg.out_npz_path, allow_pickle=True)
    coords_frames = [np.asarray(c, dtype=np.float32) for c in out_npz["coords_frames"]]
    mass_frames = [np.asarray(m, dtype=np.float32) for m in out_npz["mass_frames"]]
    Z_frames_in_npz = [np.asarray(z, dtype=np.float32) for z in out_npz["Z_frames"]] if "Z_frames" in out_npz.files else None

    def _pad_or_repeat_Z(Z0: torch.Tensor, n: int) -> torch.Tensor:
        n0 = int(Z0.shape[0]); D = int(Z0.shape[1])
        if n == n0: return Z0.clone()
        if n < n0: return Z0[:n].clone()
        idx = (torch.arange(n - n0, device=Z0.device) % n0).long()
        return torch.cat([Z0, Z0[idx]], dim=0).reshape(n, D)

    coords_seq_torch = [torch.as_tensor(c, device=device, dtype=torch.float32) for c in coords_frames]
    Tp1 = len(coords_frames); T = Tp1 - 1

    target_state = {"coords": torch.from_numpy(xytgt_norm).to(device=device, dtype=torch.float32),
                    "layers": torch.from_numpy(layer_idx_tgt).to(device=device, dtype=torch.long)}

    if name_col is not None and (name_col in ad_src.obs.columns) and (name_col in ad_tgt.obs.columns):
        s0 = set(ad_src.obs[name_col].astype(str).values.tolist())
        st = ad_tgt.obs[name_col].astype(str).values
        is_new = np.array([(x not in s0) for x in st], dtype=np.bool_)
        target_state["is_new"] = torch.from_numpy(is_new).to(device=device)

    if bool(cfg.use_latent):
        Zt, ctx.latent_cache = get_latent_tensor(
            ad_tgt,
            device=device,
            model_dir=cfg.scanvi_dir,
            model_type=cfg.model_type,
            fallback_obsm=cfg.latent_fallback_obsm,
            fallback_layer=cfg.latent_fallback_layer,
            cache=ctx.latent_cache,
            load_ref_adata=getattr(ctx, "adata_all", None),
        )
    else:
        Zt = torch.zeros((ad_tgt.n_obs, 1), device=device)
    target_state["latent"] = Zt

    if bool(cfg.use_latent):
        if (Z_frames_in_npz is not None) and (len(Z_frames_in_npz) == Tp1):
            Z_seq_torch = [torch.as_tensor(z, device=device, dtype=torch.float32) for z in Z_frames_in_npz]
        else:
            Z0_src, ctx.latent_cache = get_latent_tensor(
                ad_src,
                device=device,
                model_dir=cfg.scanvi_dir,
                model_type=cfg.model_type,
                fallback_obsm=cfg.latent_fallback_obsm,
                fallback_layer=cfg.latent_fallback_layer,
                cache=ctx.latent_cache,
                load_ref_adata=getattr(ctx, "adata_all", None),
            )
            Z_seq_torch = [_pad_or_repeat_Z(Z0_src, int(c.shape[0])) for c in coords_seq_torch]
    else:
        Z_seq_torch = [torch.zeros((c.shape[0], 1), device=device) for c in coords_seq_torch]

    shells_norm = load_shells_from_dir(cfg.bound_dir, include_loops=False)
    H, W, _ = auto_choose_grid_size(xysrc_norm, shells_norm[0], pts_per_cell=1.0, base=int(GRID_BASE), max_hw=int(GRID_HW_MAX))
    print(f"[Grid] H={H}, W={W}, H×W={H*W}")

    grid_cache_np = build_shell_grid_cache(shells_norm, H=int(H), W=int(W), margin=float(GRID_MARGIN))
    grid_cache_dev = []
    for (x0, y0, dx, dy, in_shell_np) in grid_cache_np:
        grid_cache_dev.append((torch.tensor(x0, device=device, dtype=torch.float32),
                               torch.tensor(y0, device=device, dtype=torch.float32),
                               torch.tensor(dx, device=device, dtype=torch.float32),
                               torch.tensor(dy, device=device, dtype=torch.float32),
                               torch.from_numpy(in_shell_np).to(device=device)))


    coords_src_t = torch.from_numpy(xysrc_norm).to(device=device, dtype=torch.float32)
    coords_tgt_t = torch.from_numpy(xytgt_norm).to(device=device, dtype=torch.float32)

    cap_auto, cap_info = auto_shell_need_cap_for_stage(
        coords_src_t, coords_tgt_t, grid_cache_dev, H=int(H), W=int(W), T=int(T),
        q=0.90,  
        min_cap=1, max_cap=8
    )
    print("[auto cap]", cap_auto, cap_info)

    SHELL_NEED_CAP = int(cap_auto)  

    Ms = np.array([np.asarray(m, np.float32).reshape(-1).sum() for m in mass_frames], dtype=np.float32)
    N_tilde = compute_N_tilde_from_mass(Ms, int(ad_src.n_obs), int(ad_tgt.n_obs))

    counts0 = np.bincount(layer_idx_src, minlength=n_layers)
    countsT = np.bincount(layer_idx_tgt, minlength=n_layers)
    _, B_step_layer, D_step_layer = build_layer_plan_and_quotas(N_tilde=N_tilde, counts0=counts0, countsT=countsT,
                                                                death_frac=DEATH_FRAC, turnover_frac=TURNOVER_FRAC)


    lr_source = str(cfg.lr_source).lower()
    use_lr = bool(cfg.use_lr) and (lr_source != "none")
    state0 = {"coords": coords_seq_torch[0].clone(),
              "layers": base_layers0.clone(),
              "anchor": torch.arange(coords_seq_torch[0].shape[0], device=device, dtype=torch.long),
              "latent": Z_seq_torch[0].clone(),
              "expr_union": None,
              "lr": torch.zeros(coords_seq_torch[0].shape[0], device=device)}
    recompute_lr_fn = recompute_lr_factory_none(device)

    if use_lr and lr_source == "decoder":
        dp = build_decoder_pack(device=device, stat_json=str(cfg.stat_json), ckpt_path_dec=str(cfg.ckpt_path_dec),
                                lr_pairs_df=ctx.lr_pairs, latent_dim=int(state0["latent"].shape[1]))
        recompute_lr_fn = recompute_lr_factory(dp)
        with torch.no_grad(): state0["lr"] = recompute_lr_fn(state0["latent"], state0["coords"])
    elif use_lr and lr_source == "counts":
        genes_union = genes_union_from_lr_pairs(ctx.lr_pairs, ad_src.var_names)
        if len(genes_union) > 0:
            state0["expr_union"] = torch.from_numpy(dense_from_layer_by_genes(ad_src, str(cfg.counts_layer), genes_union)).to(device=device, dtype=torch.float32)
            target_state["expr_union"] = torch.from_numpy(dense_from_layer_by_genes(ad_tgt, str(cfg.counts_layer), genes_union)).to(device=device, dtype=torch.float32)
            recompute_lr_fn = recompute_lr_factory_counts(device=device, lr_pairs_df=ctx.lr_pairs, genes_union=genes_union, compute_lr_potential_gpu=compute_lr_potential_gpu)
            with torch.no_grad(): state0["lr"] = recompute_lr_fn(state0["expr_union"], state0["coords"])
        else:
            recompute_lr_fn = recompute_lr_factory_none(device)
    else:
        recompute_lr_fn = recompute_lr_factory_none(device)

    return StagePack(cfg=cfg, n_layers=n_layers, T=T, H=int(H), W=int(W), shells_norm=shells_norm,
                     coords_seq_torch=coords_seq_torch, Z_seq_torch=Z_seq_torch,
                     base_layers0=base_layers0, grid_cache_np=grid_cache_np, grid_cache_dev=grid_cache_dev,
                     B_step_layer=B_step_layer, D_step_layer=D_step_layer,
                     state0=state0, target_state=target_state, recompute_lr_fn=recompute_lr_fn,
                     shell_need_cap=int(SHELL_NEED_CAP))


#---------------start function---------------
def run_multi_stages(*, ctx, W_XY_TGT, W_Z_TGT, sample_key, stages: List[StageCfg], best_ckpt_dir: str, train_kwargs: Optional[dict] = None) -> List[Dict[str, Any]]:
    train_kwargs = {} if train_kwargs is None else dict(train_kwargs)
    Path(best_ckpt_dir).mkdir(parents=True, exist_ok=True)
    outs = []
    for cfg in stages:
        lc = (cfg.layer_col if getattr(cfg, "layer_col", None) else getattr(ctx, "layer_col", "annotation"))
        pack = prepare_one_stage(ctx, cfg, sample_key=sample_key, layer_col=lc)
        best_ckpt_path = str(Path(best_ckpt_dir) / f"policy_{cfg.src}_to_{cfg.tgt}.pt")
        out = train_policy_rl(state0=pack.state0, target_state=pack.target_state, shells_norm=pack.shells_norm,
                              coords_seq_torch=pack.coords_seq_torch, Z_seq_torch=pack.Z_seq_torch, base_layers0=pack.base_layers0,
                              grid_cache_np=pack.grid_cache_np, grid_cache_dev=pack.grid_cache_dev, 
                              B_step_layer=pack.B_step_layer, D_step_layer=pack.D_step_layer, T=pack.T,
                              device=ctx.device, n_layers=pack.n_layers, H=pack.H, W=pack.W, recompute_lr_fn=pack.recompute_lr_fn,
                              policy_cls=GrowthPolicyNet, env_cls=SimulationEnv,
                              USE_LR=bool(cfg.use_lr) and (str(cfg.lr_source).lower() != "none"),
                              best_ckpt_path=best_ckpt_path, W_XY_TGT=W_XY_TGT, W_Z_TGT=W_Z_TGT, save_best=True,
                              shell_need_cap=pack.shell_need_cap, **train_kwargs)
        outs.append({"stage": cfg, "best_ckpt_path": best_ckpt_path, "train_out": out})
    return outs


#---------------rollout function--------------- 
def _infer_policy_args_smart(in_features: int, n_layers: int, hidden_dim: int):
    for use_lr in (False, True):
        d = 3 + (1 if use_lr else 0)
        if d == in_features:
            return {"n_layers": n_layers, "use_lr": use_lr, "use_t": False, "hidden_dim": hidden_dim}


    for use_lr in (False, True):
        for use_t in (False, True):
            d = 3 + (1 if use_lr else 0) + (1 if use_t else 0)
            if d == in_features:
                return {"n_layers": n_layers, "use_lr": use_lr, "use_t": use_t, "hidden_dim": hidden_dim}

    raise RuntimeError(f"Unable to infer the GrowthPolicyNet configuration based on the weight dimension {in_features}.")

def _load_policy_net(s2, ckpt_path, *, n_layers, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state", None) or ckpt.get("state_dict", None) or ckpt
    if "net.0.weight" in sd:
        w = sd["net.0.weight"]; in_features = int(w.shape[1]); hidden_dim = int(w.shape[0])
    else:
        keys = list(sd.keys()); w = sd[keys[0]]
        in_features = int(w.shape[1]); hidden_dim = int(w.shape[0])

    try:
        args = _infer_policy_args_smart(in_features, n_layers, hidden_dim)
    except RuntimeError:
        if ckpt.get("policy_args"):
            args = dict(ckpt["policy_args"]); args["n_layers"] = n_layers
        else:
            raise
    PolicyCls = s2.GrowthPolicyNet
    net = PolicyCls(**_filter_kwargs(PolicyCls, args)).to(device)
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net

@torch.no_grad()
def rollout_policy_one_stage(
    s2, ctx, cfg, ckpt_path, *, seed=0, sample_key='sample',
    ADVECT_LATENT=True, TAU_BIRTH=1.0, TAU_DEATH=1.0, LATENT_NOISE_SCALE=0.02,
    AUTO_PRINT: bool = True,
):
    lc = (cfg.layer_col if getattr(cfg, "layer_col", None) else getattr(ctx, "layer_col", "annotation"))
    pack = s2.prepare_one_stage(ctx, cfg, layer_col=lc, sample_key=sample_key, AUTO_PRINT=AUTO_PRINT)
    device = ctx.device
    policy_net = _load_policy_net(s2, ckpt_path, n_layers=pack.n_layers, device=device)

    EnvCls = s2.SimulationEnv
    env_kw = dict(
        state0=pack.state0, target_state=pack.target_state,
        base_coords_seq=pack.coords_seq_torch, base_Z_seq=pack.Z_seq_torch, base_layers0=pack.base_layers0,
        grid_cache_np=pack.grid_cache_np, grid_cache_dev=pack.grid_cache_dev, shells_norm=pack.shells_norm,
        T=pack.T, device=device, n_layers=pack.n_layers, H=pack.H, W=pack.W,
        recompute_lr_fn=pack.recompute_lr_fn, shell_need_cap=getattr(pack, "shell_need_cap", 2),
    )
    env = EnvCls(**_filter_kwargs(EnvCls, env_kw))

    if "uid" not in env.state or env.state.get("uid", None) is None:
        N0 = int(env.state["coords"].shape[0])
        env.state["uid"] = torch.arange(N0, device=device, dtype=torch.long)
        env.state["parent_uid"] = torch.full((N0,), -1, device=device, dtype=torch.long)
        env._next_uid = int(N0)
        
    def _np(x, dt=None):
        if x is None: return None
        y = x.detach().cpu().numpy()
        return y.astype(dt) if dt is not None else y

    def _get_state(e):
        st = e.state
        c, l = st["coords"], st["layers"]
        z, ib = st.get("latent", None), st.get("is_birth", None)
        a, lr = st.get("anchor", None), st.get("lr", None)
        uid, puid = st.get("uid", None), st.get("parent_uid", None)
        return c, l, z, ib, a, lr, uid, puid

    coords_list, layers_list, latent_list, is_birth_list, anchor_list, lr_list = [], [], [], [], [], []
    uid_list, parent_uid_list = [], []

    c0, l0, z0, ib0, a0, lr0, uid0, puid0 = _get_state(env)
    coords_list.append(_np(c0, np.float32)); layers_list.append(_np(l0, np.int64))
    latent_list.append(_np(z0, np.float32))
    is_birth_list.append(np.zeros((coords_list[-1].shape[0],), np.bool_) if ib0 is None else _np(ib0, np.bool_))
    anchor_list.append(_np(a0, np.int64))
    lr_list.append(_np(lr0, np.float32))
    uid_list.append(_np(uid0, np.int64))
    parent_uid_list.append(_np(puid0, np.int64))

    for t in range(pack.T):
        env.step(
            policy_net, birth_quota_layer=pack.B_step_layer[t], death_quota_layer=pack.D_step_layer[t],
            dir_step=+1, advect_latent=ADVECT_LATENT, seed_step=seed * 1000 + t,
            TAU_BIRTH=TAU_BIRTH, TAU_DEATH=TAU_DEATH, LATENT_NOISE_SCALE=LATENT_NOISE_SCALE,
        )
        c, l, z, ib, a, lr, uid, puid = _get_state(env)
        coords_list.append(_np(c, np.float32)); layers_list.append(_np(l, np.int64))
        latent_list.append(_np(z, np.float32))
        is_birth_list.append(np.zeros((coords_list[-1].shape[0],), np.bool_) if ib is None else _np(ib, np.bool_))
        anchor_list.append(_np(a, np.int64))
        lr_list.append(_np(lr, np.float32))
        uid_list.append(_np(uid, np.int64))
        parent_uid_list.append(_np(puid, np.int64))

    t_arr = np.linspace(0.0, 1.0, len(coords_list), dtype=np.float32)
    return dict(
        coords=coords_list, layers=layers_list, latent=latent_list, lr=lr_list, anchor=anchor_list, is_birth=is_birth_list,
        uid=uid_list, parent_uid=parent_uid_list, t=t_arr, T=pack.T, n_layers=pack.n_layers,
    )

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[device]", device)