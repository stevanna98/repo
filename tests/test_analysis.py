import numpy as np
import pytest
from normative_vae.metrics import reconstruction_metrics
from normative_vae.topology import graph_statistics, hub_overlap, discrepancies
from normative_vae.external import bh_adjust, group_test, standardized, matrix_composite
from normative_vae.latent_analysis import classification_metrics
from normative_vae.external import density_selection
from normative_vae.data import Cohort

TOPO = dict(leiden_runs=2, hub_quantile=.9, quantile_method="linear", tolerance=1e-8)


def test_raw_metrics_and_symmetry_not_bounded_below():
    x = np.array([[1., .2], [.2, 1.]])
    scalar, regional = reconstruction_metrics(x, x)
    assert scalar["rmse"] == 0 and scalar["symmetry"] == 1
    assert scalar["pearson"] == pytest.approx(1)
    assert np.allclose(regional["pearson"], 1)
    y = x.copy(); y[0, 1] = 20
    assert reconstruction_metrics(x, y)[0]["symmetry"] < 0
    assert np.isnan(reconstruction_metrics(x, np.ones_like(x))[0]["pearson"])


def test_local_efficiency_uses_neighbor_induced_graph_and_disconnections():
    # Four-cycle: each node's neighbors connect only through other full-graph nodes.
    a = np.zeros((4, 4), bool)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        a[u, v] = a[v, u] = True
    values = graph_statistics(a, TOPO, 42)
    assert values["scalar"]["local_efficiency"] == 0
    assert values["scalar"]["global_efficiency"] == pytest.approx(5 / 6)
    disconnected = np.zeros((4, 4), bool); disconnected[0, 1] = disconnected[1, 0] = True
    result = graph_statistics(disconnected, TOPO, 42)
    assert result["lcc_size"] == 2 and result["scalar"]["path_length"] == 1
    assert result["scalar"]["global_efficiency"] == pytest.approx(1 / 6)
    assert result["regional"]["participation"][3] == 0


def test_hub_identity_not_hub_density():
    a = np.zeros((20, 20), bool)
    a[0, 1:] = a[1:, 0] = True
    a[1, 2:] = a[2:, 1] = True
    perm = np.roll(np.arange(20), 5)
    b = a[np.ix_(perm, perm)]
    original = graph_statistics(a, TOPO, 42)
    reconstructed = graph_statistics(b, TOPO, 42)
    h1, h2 = original["hubs"], reconstructed["hubs"]
    assert h1.sum() == h2.sum() == 2
    assert a[np.ix_(h1, h1)].sum() == b[np.ix_(h2, h2)].sum() == 2
    result = hub_overlap(h1, h2)
    assert result["hub_distance"] == 1 and result["hub_retention"] == 0
    assert (result["status"] == "lost").sum() == 2
    assert hub_overlap(h1, h1)["hub_distance"] == 0
    assert np.isnan(hub_overlap([], [], valid=False)["hub_distance"])


def test_percentage_errors_undefined_denominators():
    absolute, signed, raw = discrepancies([0, -2, 2], [1, -1, 1], 1e-8)
    assert np.isnan(absolute[0]) and raw[0] == 1
    assert np.array_equal(absolute[1:], [50, 50])
    assert np.array_equal(signed[1:], [50, -50])


def test_statistics_family_size_and_effect_direction():
    q = bh_adjust([.01, np.nan, .2])
    assert q[0] == pytest.approx(.03) and np.isnan(q[1])
    cfg = dict(min_group_n=2, permutations=99)
    positive = group_test([4, 5, 6], [1, 2, 3], cfg, 42)
    assert positive["rank_biserial"] == 1 and positive["method"] == "exact_no_ties"
    assert group_test([1, 1], [1, 1], cfg, 42)["p"] == 1
    assert np.isnan(group_test([1, np.nan], [2, 3], cfg, 42)["p"])
    tied = group_test([1, 1, 2], [2, 2, 3], cfg, 42)
    assert tied["method"].startswith("permutation")


def test_standardization_and_strict_composite():
    data = np.tile(np.array([0., 2., 4.])[:, None], (1, 12))
    z, mean, sd = standardized(data, [True, True, False], np.ones(12), 1e-8)
    assert np.allclose(sd, np.sqrt(2))
    assert matrix_composite(z)[2] == pytest.approx(3 / np.sqrt(2))
    z[2, 0] = np.nan
    assert np.isnan(matrix_composite(z)[2])
    assert np.isnan(standardized(np.ones((3, 2)), [True, True, False], [1, 1], 1e-8)[0]).all()
    assert classification_metrics([0, 1], [.5, .5])["recall"] == 1


def test_ensemble_metrics_are_not_mean_of_metrics():
    x = np.eye(3)
    a, b = x + 1, x - 1
    assert reconstruction_metrics(x, (a + b) / 2)[0]["rmse"] == 0
    assert np.mean([reconstruction_metrics(x, y)[0]["rmse"] for y in [a, b]]) == 1


def test_reference_only_density_selection_and_lower_tie(tmp_path):
    matrices = np.repeat(np.eye(2)[None], 4, axis=0)
    reference = Cohort(matrices, np.array(["a", "b", "c", "d"]), np.array([0, 1, 0, 1]), np.repeat("HC", 4), "reference")
    records = []
    for density, error in [(.1, .2), (.2, .1), (.3, .1)]:
        for fold, indices in enumerate([[0, 1], [2, 3]]):
            directory = tmp_path / f"density_{density}_fold_{fold}"
            directory.mkdir()
            np.savez(directory / "test_predictions.npz", indices=indices, subject_ids=reference.ids[indices], reconstruction=matrices[indices] + error)
            records.append(dict(architecture="set", density=density, fold=fold, directory=str(directory)))
    density, _ = density_selection({"densities": [.1, .2, .3]}, reference, records)
    assert density == .2
    with pytest.raises(ValueError, match="exactly one"):
        density_selection({"densities": [.1, .2, .3]}, reference, records + [records[0]])


def test_incomplete_hc_reference_not_silently_omitted():
    x = np.array([[0., 1.], [np.nan, 3.], [4., 5.]])
    z, _, _ = standardized(x, [True, True, False], [1, 1], 1e-8)
    assert np.isnan(z[:, 0]).all() and np.isfinite(z[:, 1]).all()


def test_topology_cache_separates_configurations(tmp_path):
    from normative_vae.topology import TopologyEvaluator
    a = np.ones((5, 5), dtype=bool); np.fill_diagonal(a, False)
    TopologyEvaluator(TOPO, 42, tmp_path).graph(a)
    TopologyEvaluator(dict(TOPO, leiden_runs=3), 42, tmp_path).graph(a)
    assert len(list(tmp_path.glob("*.json"))) == 2
