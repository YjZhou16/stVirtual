import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _parse_stage_key(stage_key: str) -> Tuple[str, str]:
    if "_to_" not in stage_key:
        raise ValueError(f"bad stage_key: {stage_key}")
    a, b = stage_key.split("_to_", 1)
    return str(a), str(b)

def _to_numpy_list(x_list, dtype=np.float32) -> List[np.ndarray]:
    out = []
    for x in x_list:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        out.append(np.asarray(x, dtype=dtype))
    return out

def _align_ann_by_names(adata_sub, ann_key: str, obs_names: Optional[List[str]]) -> Optional[np.ndarray]:
    if obs_names is None:
        return None
    idx = adata_sub.obs_names.get_indexer(obs_names)
    if np.any(idx < 0):
        return None
    return adata_sub.obs.iloc[idx][ann_key].astype(str).to_numpy()

def save_res(
    *,
    res1: Dict[str, Any],
    adata_all,
    out_dir: str,
    slice_key: str = "sample",
    ann_key: str = "His_anno",
    save_prefix: str = "rollout_stage1",
    steps: int = 10,
    n_cache: int = 256,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_num_steps: int = 20000,
    unnormalize: bool = False,
) -> Dict[str, str]:

    from utils.traj_ana import rollout_trace_from_out

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if ann_key not in adata_all.obs.columns:
        raise KeyError(f"ann_key '{ann_key}' not in adata.obs")

    saved: Dict[str, str] = {}

    for stage_key, stage_out in res1.items():
        if not isinstance(stage_key, str) or "_to_" not in stage_key:
            continue

        src, tgt = _parse_stage_key(stage_key)

        trace = rollout_trace_from_out(
            stage_out,
            steps=int(steps),
            n_cache=int(n_cache),
            rtol=float(rtol),
            atol=float(atol),
            max_num_steps=int(max_num_steps),
            unnormalize=bool(unnormalize),
        )

        coords_frames = _to_numpy_list(trace["x"], dtype=np.float32)
        Z_frames      = _to_numpy_list(trace["q"], dtype=np.float32)
        mass_frames   = _to_numpy_list(trace["m"], dtype=np.float32)
        t_eval        = np.asarray(trace["t"], dtype=np.float32)

        N0 = int(coords_frames[0].shape[0])

        adata_src = adata_all[adata_all.obs[slice_key].astype(str) == src]
        adata_tgt = adata_all[adata_all.obs[slice_key].astype(str) == tgt]

        obs_names0 = None
        for k in ["obs_names0", "cell_names0", "obs_names", "cell_names"]:
            if isinstance(stage_out, dict) and (k in stage_out):
                v = stage_out[k]
                if hasattr(v, "tolist"):
                    v = v.tolist()
                if isinstance(v, (list, tuple)) and len(v) == N0:
                    obs_names0 = list(map(str, v))
                    break

        if obs_names0 is None and adata_src.n_obs == N0:
            obs_names0 = list(map(str, adata_src.obs_names.tolist()))

        src_ann = _align_ann_by_names(adata_src, ann_key, obs_names0)
        if src_ann is None:
            src_ann = adata_src.obs[ann_key].astype(str).to_numpy()
            if src_ann.shape[0] != N0:
                src_ann = None

        tgt_ann = adata_tgt.obs[ann_key].astype(str).to_numpy()

        layers_list = sorted(
            np.unique(
                np.concatenate([src_ann if src_ann is not None else np.array([], dtype=object),
                                tgt_ann], axis=0)
            ).astype(str)
        )
        layer_to_idx = {s: i for i, s in enumerate(layers_list)}

        src_layer_idx = None
        if src_ann is not None:
            src_layer_idx = np.array([layer_to_idx.get(s, 0) for s in src_ann], dtype=np.int64)
        tgt_layer_idx = np.array([layer_to_idx.get(s, 0) for s in tgt_ann], dtype=np.int64)

        npz_path = out_dir_p / f"{save_prefix}_{stage_key}.npz"

        payload: Dict[str, Any] = {
            "stage_key": stage_key,
            "src": src,
            "tgt": tgt,
            "t": t_eval,
            "coords_frames": np.array(coords_frames, dtype=object),
            "Z_frames": np.array(Z_frames, dtype=object),
            "mass_frames": np.array(mass_frames, dtype=object),
            "ann_key": str(ann_key),
            "layers_list": np.array(layers_list, dtype=object),
            "tgt_layer_idx": tgt_layer_idx,
        }

        if obs_names0 is not None:
            payload["src_obs_names0"] = np.array(obs_names0, dtype=object)

        if src_ann is not None:
            payload["src_ann0"] = np.array(src_ann, dtype=object)
            payload["src_layer_idx0"] = src_layer_idx

        payload["tgt_obs_names"] = np.array(adata_tgt.obs_names.astype(str).to_numpy(), dtype=object)
        payload["tgt_ann"] = np.array(tgt_ann, dtype=object)

        ns = None
        if isinstance(stage_out, dict):
            ns = stage_out.get("norm_stats", None)
        if isinstance(ns, dict):
            for k in ["xy_mu", "xy_s", "f_mu", "f_std"]:
                if k in ns:
                    v = ns[k]
                    if hasattr(v, "detach"):
                        v = v.detach().cpu().numpy()
                    payload[f"norm_{k}"] = np.asarray(v, dtype=np.float32)

        np.savez_compressed(npz_path, **payload)
        saved[stage_key] = str(npz_path)

    if len(saved) == 0:
        raise RuntimeError(f"no stage saved. keys={list(res1.keys())[:30]}")

    return saved
