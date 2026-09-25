"""Small, explicit YAML configuration interface; paths resolve from repository root."""
from copy import deepcopy
from pathlib import Path
import yaml


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        cfg = merge(load_config(path.parent / parent), cfg)
    validate_config(cfg)
    return cfg


def validate_config(c):
    if c["n_regions"] != 379:
        raise ValueError("Experiment configurations must use 379 regions; small sizes are for unit tests only")
    if (c["model"]["hidden"], c["model"]["latent"]) != (256, 64):
        raise ValueError("Matched experiment widths must be 256/64")
    if set(c["architectures"]) != {"set", "gat", "gcn"} or len(c["architectures"]) != 3:
        raise ValueError("Specify each of set/gat/gcn once")
    if c["densities"] != [0.1, 0.2, 0.3]:
        raise ValueError("All three prescribed densities are required")
    if any(c["model"]["hidden"] % h for h in c["search"]["heads"]):
        raise ValueError("Each head count must divide the hidden width")
    if min(c["cv"]["outer"], c["cv"]["inner"]) < 2:
        raise ValueError("Invalid CV sizes")
    if type(c["cv"]["candidates"]) is not int or c["cv"]["candidates"] < 1:
        raise ValueError("cv.candidates must be a positive integer")
    if not c["synthetic"] and (c["cv"]["outer"], c["cv"]["inner"]) != (5, 3):
        raise ValueError("Real-data protocol requires 5 outer and 3 inner folds")
