"""Reproducible seeds, atomic artifacts, and run identity protection."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import numpy as np
import torch


def seed_for(master, *parts):
    message = json.dumps([int(master), *parts], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:4], "big") % (2**31 - 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    # Only load locally generated, trusted checkpoints (optimizer/RNG objects included).
    return torch.load(path, map_location=device, weights_only=False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    packages = {}
    for name in ["torch", "numpy", "scipy", "scikit-learn", "pandas", "igraph", "leidenalg", "PyYAML", "matplotlib", "pytest", "threadpoolctl"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def prepare_run(cfg, data_dir, output, root):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = ["reference.npz", "external.npz", "participants.csv", "rois.csv", "provenance.json"]
    source_files = sorted(Path(root).glob("src/**/*.py")) + [Path(root) / "run.py", Path(root) / "pyproject.toml"]
    identity = {"config": cfg, "inputs": {f: file_digest(Path(data_dir) / f) for f in files},
                "source": {str(p.relative_to(root)): file_digest(p) for p in source_files},
                "packages": environment()["packages"]}
    manifest = output / "run_identity.json"
    if manifest.exists() and read_json(manifest) != jsonable(identity):
        raise ValueError("Run configuration, inputs, code, or environment changed. Choose a new output directory.")
    write_json(manifest, identity)
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "environment.json", environment())
    (output / "DATA_STATUS.txt").write_text("SYNTHETIC: SOFTWARE VERIFICATION ONLY\n" if cfg["synthetic"] else "REAL INPUT DATA; inspect provenance and limitations\n")
