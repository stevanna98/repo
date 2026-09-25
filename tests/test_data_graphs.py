from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from normative_vae.config import load_config
from normative_vae.synthetic import generate
from normative_vae.data import load_data
from normative_vae.graphs import adjacency, InsufficientPositiveEdges, normalized_adjacency, additive_mask, safe_attention


def test_positive_threshold_ties_and_isolation():
    x = np.eye(5)
    x[0, 1] = x[1, 0] = -0.99
    for i, j in [(0, 2), (0, 3), (1, 2)]:
        x[i, j] = x[j, i] = .5
    a = adjacency(x, .2)
    assert a.sum() == 4 and not a.diagonal().any()
    assert a[0, 2] and a[0, 3] and not a[1, 2] and not a[0, 1]
    assert np.array_equal(a, a.T)
    t = torch.tensor(a)[None]
    assert torch.isfinite(normalized_adjacency(t)).all()
    assert (normalized_adjacency(t)[0, 4] == 0).all()
    assert additive_mask(t)[0, 0, 2] == 0
    assert torch.isneginf(additive_mask(t)[0, 0, 0])
    with pytest.raises(InsufficientPositiveEdges):
        adjacency(x, .9)


def test_safe_softmax_backward():
    scores = torch.randn(2, 3, 4, 4, requires_grad=True)
    allowed = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    allowed[:, :, 0, 1] = True
    weights = safe_attention(scores, allowed)
    assert weights[:, :, 0, 1].eq(1).all()
    assert weights[:, :, 1:].eq(0).all()
    weights.square().sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_379_synthetic_data_and_strict_alignment(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs/smoke_test.yaml")
    cfg = deepcopy(cfg)
    cfg["generation"].update(reference_n=4, external_hc=2, external_bd=1, external_mdd=1)
    generate(cfg, tmp_path / "dataset")
    ref, ext, rois, report = load_data(tmp_path / "dataset", cfg)
    assert ref.fc.shape == (4, 379, 379)
    assert ext.fc.shape == (4, 379, 379)
    assert report["reference"]["minimum_eigenvalue"] >= -1e-4
    assert rois.network.nunique() == 13
    for density in cfg["densities"]:
        assert adjacency(ref.fc[0], density).sum() == 2 * int(density * 379 * 378 / 2)
    generate(cfg, tmp_path / "dataset")  # Exact specification is idempotent.
    meta = pd.read_csv(tmp_path / "dataset/participants.csv")
    meta.iloc[::-1].to_csv(tmp_path / "dataset/participants.csv", index=False)
    with pytest.raises(ValueError, match="metadata order"):
        load_data(tmp_path / "dataset", cfg)


def test_unspecified_patient_subtype_is_external_only(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs/smoke_test.yaml")
    cfg["generation"].update(reference_n=2, external_hc=1, external_bd=1, external_mdd=1)
    dataset = tmp_path / "dataset"
    generate(cfg, dataset)
    meta = pd.read_csv(dataset / "participants.csv")
    patients = (meta.cohort == "external") & (meta.group != "HC")
    meta.loc[patients, "group"] = "PATIENT"
    meta.to_csv(dataset / "participants.csv", index=False)
    reference, external, _, _ = load_data(dataset, cfg)
    assert reference.groups.tolist() == ["HC", "HC"]
    assert external.groups.tolist() == ["HC", "PATIENT", "PATIENT"]
    assert not np.isin(external.groups, ["BD", "MDD"]).any()

    invalid = meta.copy()
    invalid.loc[invalid.cohort == "reference", "group"] = "PATIENT"
    invalid.to_csv(dataset / "participants.csv", index=False)
    with pytest.raises(ValueError, match="Invalid group labels for reference"):
        load_data(dataset, cfg)

    meta.loc[patients, "group"] = "PZ"
    meta.to_csv(dataset / "participants.csv", index=False)
    with pytest.raises(ValueError, match="Invalid group labels for external"):
        load_data(dataset, cfg)
