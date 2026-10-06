"""Runtime settings for sequential notebook execution."""
import os

def training_epochs(default):
    return int(os.environ.get("STVIRTUAL_TRAIN_EPOCHS", default))

def training_routes(routes):
    return routes[:1] if os.environ.get("STVIRTUAL_SINGLE_WINDOW") == "1" else routes

scanvi_train_kwargs = {"accelerator": "gpu", "devices": 1, "enable_progress_bar": False}
if "STVIRTUAL_SCANVI_EPOCHS" in os.environ:
    scanvi_train_kwargs["max_epochs"] = int(os.environ["STVIRTUAL_SCANVI_EPOCHS"])


def ensure_gpu_idle():
    """Require the selected GPU to be idle before starting cardiac Stage 2."""
    import json
    import subprocess
    import time
    from datetime import datetime
    from pathlib import Path
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Mcardiac Stage 2 requires an available GPU.")
    torch.cuda.synchronize()
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    query = subprocess.check_output([
        "nvidia-smi", "-i", selected,
        "--query-gpu=uuid,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    uuid, free, total, utilization = [value.strip() for value in query]
    time.sleep(2)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    other_processes = []
    for row in processes.splitlines():
        fields = [value.strip() for value in row.split(",")]
        if len(fields) == 3 and fields[0] == uuid and int(fields[1]) != os.getpid():
            other_processes.append({"pid": int(fields[1]), "memory_mib": fields[2]})
    utilization = int(subprocess.check_output([
        "nvidia-smi", "-i", selected, "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip())
    report = {
        "checked_at": datetime.now().isoformat(), "gpu": selected,
        "gpu_uuid": uuid, "free_memory_mib": int(free),
        "total_memory_mib": int(total), "utilization_percent": utilization,
        "other_processes": other_processes,
        "idle": not other_processes and utilization == 0,
    }
    if os.environ.get("STVIRTUAL_SINGLE_WINDOW") == "1":
        path = Path(__file__).resolve().parents[1] / ".work/gpu_tests/mcardiac_gpu_check.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["idle"]:
        raise RuntimeError(f"Mcardiac Stage 2 was not started: GPU {selected} is occupied. {report}")
    print(f"GPU {selected} is idle; starting Mcardiac Stage 2.")


def release_simulation_frame(input_path, output_path):
    """Validate and release temporary frames during a single-window execution."""
    if os.environ.get("STVIRTUAL_SINGLE_WINDOW") != "1":
        return
    import json
    from datetime import datetime
    from pathlib import Path
    import h5py
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    paths = [Path(input_path), Path(output_path)]
    for path in paths:
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_relative_to(root / "experiments"):
            raise ValueError(f"Frame cleanup is outside the tutorial artifacts: {path}")
        parts = resolved.relative_to(root / "experiments").parts
        if "artifacts" not in parts or not ({"simulation", "decoded"} & set(parts)):
            raise ValueError(f"Frame cleanup requires a simulation or decoded artifact: {path}")
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("Input and decoded paths must differ.")
    with h5py.File(paths[0], "r") as source, h5py.File(paths[1], "r") as decoded:
        matrix = decoded["layers/reconstructed_expression"]
        n_cells, n_genes = matrix.shape
        assert source["obsm/X_latent"].shape[0] == n_cells
        for handle in (source, decoded):
            key = handle["obs"].attrs.get("_index", "_index")
            if isinstance(key, bytes):
                key = key.decode()
            assert handle[f"obs/{key}"].shape[0] == n_cells
        source_index = source["obs"][source["obs"].attrs["_index"]][:]
        decoded_index = decoded["obs"][decoded["obs"].attrs["_index"]][:]
        assert np.array_equal(source_index, decoded_index)
        for field in ("uid", "parent_uid"):
            if field in source["obs"]:
                assert np.array_equal(source[f"obs/{field}"][:], decoded[f"obs/{field}"][:])
        library = decoded["obs/predicted_library_size"][:]
        assert np.isfinite(library).all() and (library >= 0).all()
        for start in sorted({0, max(0, n_cells - 256)}):
            stop = min(start + 256, n_cells)
            counts = matrix[start:stop]
            assert np.isfinite(counts).all() and (counts >= 0).all()
            np.testing.assert_allclose(counts.sum(axis=1), library[start:stop], rtol=2e-4, atol=1e-3)
    record = {
        "checked_at": datetime.now().isoformat(),
        "input": str(paths[0].relative_to(root)),
        "decoded": str(paths[1].relative_to(root)),
        "n_cells": n_cells, "n_genes": n_genes,
        "bytes_released": sum(path.stat().st_size for path in paths),
        "validation": "obs identity, latent rows, sampled expected counts and library totals",
    }
    for path in reversed(paths):
        path.unlink()
    with (root / ".work/gpu_tests/frame_cleanup.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def save_run_gallery(experiment_dir):
    """Save a multi-frame overview of the newly generated simulation."""
    import json
    from pathlib import Path
    import yaml
    import matplotlib.pyplot as plt
    from tutorial_utils.frame_gallery import load_gallery, plot_gallery

    directory = Path(experiment_dir).resolve()
    config = yaml.safe_load((directory / "config.yaml").read_text())
    dataset = directory.name
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else directory / path
    if "route_ids" in config:
        src, tgt = config["route_ids"][:2]
    elif "routes" in config:
        src, tgt = config["routes"][0]
    else:
        src, tgt = {
            "human_lung_cancer": ("AAH", "LUAD"),
            "human_gastric_cancer": ("normal", "cancer"),
            "Mcardiac": ("E9.5h", "E11.5h"),
        }[dataset]
    route = f"{src}_to_{tgt}"
    run_root = resolve(config["run_root"])
    frame_dir = run_root / "simulation" / route
    if dataset == "Mcardiac":
        runs = sorted(run_root.glob(f"{route}_*/simulation"), key=lambda path: path.parent.stat().st_mtime)
        if not runs:
            raise FileNotFoundError("No cardiac simulation has been saved.")
        frame_dir = runs[-1]
    checkpoint = resolve(config["stage1_checkpoint"]) / route / "checkpoints/best.pt"
    input_path = resolve(config["prepared_path"]) if "prepared_path" in config else resolve(config["scanvi_dir"]) / "adata.h5ad"
    sample_key = {
        "Membryo": "timepoint", "human_lung_cancer": "status",
        "human_gastric_cancer": "status", "Mcardiac": "stage",
    }.get(dataset, "sample")
    panels = load_gallery(
        input_path, sample_key, config["celltype_source_key"],
        [{"src": src, "tgt": tgt, "frame_dir": frame_dir, "checkpoint": checkpoint}],
        dimension=config["spatial_dimension"],
        interval=5 if dataset in {"human_lung_cancer", "human_gastric_cancer"} else 1,
    )
    palette = json.loads((directory / "colors.json").read_text())
    fig = plot_gallery(
        panels, palette, dimension=config["spatial_dimension"],
        flip_y=dataset in {"Membryo", "human_lung_cancer", "human_gastric_cancer"},
        point_size={"human_breast_cancer": 2.5, "human_lung_cancer": 4, "human_gastric_cancer": 12}.get(dataset),
    )
    output_dir = run_root / "overview"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{route}.png"
    try:
        fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    (output_dir / f"{route}.json").write_text(json.dumps({
        "dataset": dataset, "route": [src, tgt], "panel_count": len(panels),
        "titles": [panel["title"] for panel in panels],
        "frame_directory": str(frame_dir.relative_to(directory)),
        "checkpoint": str(checkpoint.relative_to(directory)),
        "palette": palette,
    }, indent=2) + "\n")
    print(f"Saved {len(panels)} panels: {output}")
    return output
