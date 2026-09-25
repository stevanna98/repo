from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from normative_vae import cross_validation as cv
from normative_vae.data import Cohort
from normative_vae.training import Stopping, mean_rmse, train_vae
from normative_vae.utils import load_checkpoint


def small_cfg():
    return dict(n_regions=4, model=dict(hidden=8, latent=3, norm_eps=1e-5, gat_slope=.2, bias=True), seed=42, cv=dict(outer=2, inner=2, candidates=1), search=dict(heads=[1], layers=[1], lr=[.001], weight_decay=[.01], dropout=[.1]), architectures=["set", "gat", "gcn"], densities=[.1], training=dict(batch_size=2, max_epochs=3, patience=3, min_delta=1e-4, clip_norm=5., betas=[.9, .999], adam_eps=1e-8, pearson_eps=1e-8))


def test_splits_candidates_weights_and_epochs():
    cfg = small_cfg()
    splits = cv.make_splits(np.resize([0, 1], 24), 5, 3, 42)
    assert sorted(i for s in splits for i in s["test"]) == list(range(24))
    for s in splits:
        assert not set(s["development"]) & set(s["test"])
        assert sorted(i for v in s["inner"] for i in v["validation"]) == sorted(s["development"])
        for inner in s["inner"]:
            assert not set(inner["train"]) & set(inner["validation"])
            assert not set(inner["train"]) & set(s["test"])
    assert cv.candidates(cfg, "gcn", 4) == cv.candidates(cfg, "gcn", 4)
    assert "heads" not in cv.candidates(cfg, "gcn", 4)[0]
    scores = [dict(score=1, validation_n=2, best_epoch=2), dict(score=3, validation_n=1, best_epoch=4), dict(score=1, validation_n=1, best_epoch=3)]
    assert cv.candidate_score(scores) == 1.5
    assert cv.refit_epochs(scores) == 3


def test_best_checkpoint_independent_of_patience_delta():
    stop = Stopping(2, .1)
    assert stop.update(1., 1) == (True, False)
    assert stop.update(.99, 2) == (True, False)
    assert stop.update(.98, 3) == (True, True)
    assert stop.best_epoch == 3 and stop.anchor == 1.


def test_validation_rmse_participant_weighted():
    class Zero(torch.nn.Module):
        def forward(self, x, a):
            return torch.zeros_like(x), None, None
    x = torch.tensor([0., 0., 3.]).reshape(3, 1, 1).expand(-1, 2, 2)
    ds = TensorDataset(x, torch.zeros_like(x, dtype=torch.bool))
    assert mean_rmse(Zero(), ds, 2, "cpu") == 1.


def test_checkpoint_restart_exact_and_loss_state(tmp_path, monkeypatch):
    from normative_vae import training as training_module
    cfg = small_cfg()
    torch.manual_seed(13)
    x = torch.randn(4, 4, 4)
    a = (~torch.eye(4, dtype=torch.bool))[None].expand(4, -1, -1)
    dataset = TensorDataset(x, a)
    candidate = cv.candidates(cfg, "gcn", 1)[0]
    full = train_vae(cfg, "gcn", candidate, dataset, None, tmp_path / "full", 99, "cpu", epochs=3)
    real_save = training_module.save_checkpoint
    def interrupted(path, value):
        real_save(path, value)
        if Path(path).name == "last.pt" and value["epoch"] == 1:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(training_module, "save_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        train_vae(cfg, "gcn", candidate, dataset, None, tmp_path / "resumed", 99, "cpu", epochs=3)
    monkeypatch.setattr(training_module, "save_checkpoint", real_save)
    resumed = train_vae(cfg, "gcn", candidate, dataset, None, tmp_path / "resumed", 99, "cpu", epochs=3)
    first, second = load_checkpoint(full["checkpoint"]), load_checkpoint(resumed["checkpoint"])
    for field in ["model", "objective"]:
        assert all(torch.equal(first[field][k], second[field][k]) for k in first[field])
    assert second["optimizer"]["param_groups"][1]["weight_decay"] == 0
    assert second["objective"]["alpha_logit"] != pytest.approx(np.log(4))


def test_outer_test_never_passed_to_training(tmp_path, monkeypatch):
    cfg = small_cfg()
    x = np.full((12, 4, 4), .5, dtype=np.float32)
    x[:, 0, 0] = np.arange(12)
    cohort = Cohort(x, np.array([f"s{i}" for i in range(12)]), np.resize([0, 1], 12), np.repeat("HC", 12), "reference")
    expected = cv.make_splits(cohort.sex, 2, 2, 42)
    calls = []
    def fake_fit(cfg, architecture, candidate, training, validation, directory, seed, device, epochs=None):
        fold = next(int(p.split("_")[1]) for p in Path(directory).parts if p.startswith("fold_"))
        train_ids = set(training.tensors[0][:, 0, 0].int().tolist())
        assert not train_ids & set(expected[fold]["test"])
        if validation is not None:
            val_ids = set(validation.tensors[0][:, 0, 0].int().tolist())
            assert not val_ids & set(expected[fold]["test"]) and not train_ids & val_ids
        else:
            assert train_ids == set(expected[fold]["development"]) and epochs == 2
        calls.append((architecture, fold, train_ids))
        return dict(checkpoint="unused", best_epoch=2, score=.1, validation_n=len(validation) if validation is not None else 0)
    monkeypatch.setattr(cv, "train_vae", fake_fit)
    monkeypatch.setattr(cv, "load_model", lambda *args: None)
    monkeypatch.setattr(cv, "predict", lambda model, data, batch, device: data.tensors[0].numpy())
    records = cv.run_cv(cfg, cohort, tmp_path, "cpu")
    assert len(records) == 6 and len(calls) == 18
    for architecture in cfg["architectures"]:
        rows = [r for r in records if r["architecture"] == architecture]
        assert [r["split"] for r in rows] == expected


def test_full_training_requires_opt_in():
    import subprocess
    import sys
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, str(root / "run.py"), "all", "--config", "configs/manuscript.yaml"], capture_output=True, text=True)
    assert result.returncode == 2 and "2700 candidate fits" in result.stderr
    assert "--allow-full-run" in result.stderr


def test_run_identity_rejects_changed_config(tmp_path):
    from normative_vae.utils import prepare_run
    root, data, output = tmp_path / "project", tmp_path / "data", tmp_path / "out"
    root.mkdir(); data.mkdir()
    for name in ["run.py", "pyproject.toml"]:
        (root / name).write_text("test fixture")
    for name in ["reference.npz", "external.npz", "participants.csv", "rois.csv", "provenance.json"]:
        (data / name).write_text("test fixture")
    prepare_run({"synthetic": True, "seed": 1}, data, output, root)
    prepare_run({"synthetic": True, "seed": 1}, data, output, root)
    with pytest.raises(ValueError, match="changed"):
        prepare_run({"synthetic": True, "seed": 2}, data, output, root)
