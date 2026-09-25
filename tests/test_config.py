from pathlib import Path

import pytest

from normative_vae.config import load_config, validate_config
from normative_vae.cross_validation import candidates


CONFIGS = Path(__file__).parents[1] / "configs"


def test_real_data_candidate_override_preserves_folds_and_default():
    reduced = load_config(CONFIGS / "hcp_ya_policlinico.yaml")
    manuscript = load_config(CONFIGS / "manuscript.yaml")
    assert reduced["cv"] == {"outer": 5, "inner": 3, "candidates": 2}
    assert manuscript["cv"]["candidates"] == 20
    assert reduced["output_dir"] != manuscript["output_dir"]
    for architecture in reduced["architectures"]:
        sampled = candidates(reduced, architecture, reduced["seed"])
        assert len(sampled) == 2
        assert sampled[0] != sampled[1]


@pytest.mark.parametrize("budget", [0, -1, 2.5, True, "2"])
def test_invalid_candidate_budget_rejected(budget):
    cfg = load_config(CONFIGS / "manuscript.yaml")
    cfg["cv"]["candidates"] = budget
    with pytest.raises(ValueError, match="cv.candidates must be a positive integer"):
        validate_config(cfg)


@pytest.mark.parametrize("outer,inner", [(2, 3), (5, 2)])
def test_real_data_fold_counts_remain_fixed(outer, inner):
    cfg = load_config(CONFIGS / "hcp_ya_policlinico.yaml")
    cfg["cv"].update(outer=outer, inner=inner)
    with pytest.raises(ValueError, match="5 outer and 3 inner folds"):
        validate_config(cfg)
