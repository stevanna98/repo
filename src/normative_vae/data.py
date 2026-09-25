"""Strict, identity-bearing input format; no imputation or silent repairs."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from .utils import read_json


@dataclass
class Cohort:
    fc: np.ndarray
    ids: np.ndarray
    sex: np.ndarray
    groups: np.ndarray
    name: str


def validate_fc(fc, n, tolerance=1e-5, psd_tolerance=1e-4):
    if fc.ndim != 3 or fc.shape[1:] != (n, n) or len(fc) == 0:
        raise ValueError(f"Expected nonempty [subjects,{n},{n}] matrices; got {fc.shape}")
    if not np.isfinite(fc).all():
        raise ValueError("Nonfinite FC input")
    if not np.allclose(fc, fc.swapaxes(-1, -2), atol=tolerance, rtol=0):
        raise ValueError("Input FC is not symmetric within tolerance; no repair performed")
    if not np.allclose(np.diagonal(fc, axis1=1, axis2=2), 1, atol=tolerance, rtol=0):
        raise ValueError("Input diagonal is not one")
    if np.max(np.abs(fc)) > 1 + tolerance:
        raise ValueError("Input entries outside correlation range")
    minima = [float(np.linalg.eigvalsh(x.astype(np.float64))[0]) for x in fc]
    if min(minima) < -psd_tolerance:
        raise ValueError("FC is not positive semidefinite within tolerance; no repair performed")
    return {"subjects": len(fc), "regions": n, "minimum_eigenvalue": min(minima), "repairs": 0}


def load_data(directory, cfg):
    directory = Path(directory)
    provenance = read_json(directory / "provenance.json")
    if provenance.get("synthetic") is not cfg["synthetic"]:
        raise ValueError("Data provenance synthetic flag conflicts with configuration")
    meta = pd.read_csv(directory / "participants.csv", dtype=str, keep_default_na=False)
    rois = pd.read_csv(directory / "rois.csv", dtype=str, keep_default_na=False)
    required = {"subject_id", "cohort", "sex", "group"}
    if not required.issubset(meta) or not {"index", "roi_id", "label", "network"}.issubset(rois):
        raise ValueError("Missing metadata or ROI columns; consult data/README.md")
    if meta[list(required)].eq("").any().any() or meta.subject_id.duplicated().any():
        raise ValueError("Missing or duplicate participant identifiers/metadata")
    n = cfg["n_regions"]
    if len(rois) != n or list(rois["index"].astype(int)) != list(range(n)):
        raise ValueError("ROI table must already follow matrix order, indexed 0..N-1")
    if rois.roi_id.duplicated().any() or rois[["roi_id", "label", "network"]].eq("").any().any():
        raise ValueError("Invalid ROI identities/labels")
    if rois.network.nunique() != cfg["clustering"]["clusters"]:
        raise ValueError("ROI mapping does not contain the prescribed 13 groups")
    if not set(meta.sex).issubset({"F", "M"}):
        raise ValueError("Explicit sex encoding required: F or M")
    if not set(meta.cohort).issubset({"reference", "external"}):
        raise ValueError("Unknown cohort")
    cohorts, report = {}, {}
    for name in ("reference", "external"):
        with np.load(directory / f"{name}.npz", allow_pickle=False) as packed:
            fc = packed["fc"]
            ids = packed["subject_ids"].astype(str)
            roi_ids = packed["roi_ids"].astype(str)
        rows = meta.loc[meta.cohort == name]
        if ids.ndim != 1 or len(ids) != len(fc) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid array subject IDs: {name}")
        if not np.array_equal(ids, rows.subject_id.to_numpy()):
            raise ValueError(f"{name} metadata order/IDs differ from array IDs; align explicitly")
        if not np.array_equal(roi_ids, rois.roi_id.to_numpy()):
            raise ValueError(f"{name} ROI axes do not match ordered mapping")
        if not set(rows.group).issubset({"HC"} if name == "reference" else {"HC", "BD", "MDD"}):
            raise ValueError(f"Invalid group labels for {name}")
        report[name] = validate_fc(fc, n, **{ "tolerance": cfg["validation"]["correlation_tolerance"], "psd_tolerance": cfg["validation"]["psd_tolerance"]})
        cohorts[name] = Cohort(fc.astype(np.float32), ids, (rows.sex.to_numpy() == "M").astype(np.float32), rows.group.to_numpy(), name)
    return cohorts["reference"], cohorts["external"], rois, report


def tensors(cohort, adjacency, indices):
    idx = np.asarray(indices, dtype=int)
    return TensorDataset(torch.from_numpy(cohort.fc[idx]), torch.from_numpy(adjacency[idx]))

