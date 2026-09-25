"""Frozen external ensembles and exploratory, explicitly scoped clinical analyses."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, PermutationMethod
from .data import tensors
from .graphs import adjacency_batch
from .metrics import RECON, REG_RECON
from .topology import TOPO_ERRORS, REGIONAL, TopologyEvaluator
from .training import load_model, predict
from .reporting import evaluate_predictions
from .utils import seed_for, write_json, read_json

MATRIX = RECON + TOPO_ERRORS
REGION = REG_RECON + [f"{k}_ape" for k in REGIONAL]


def bh_adjust(pvalues):
    """Keep full prespecified family; unavailable p-values count as 1 but stay NA."""
    p = np.asarray(pvalues, float)
    valid = np.isfinite(p)
    work = np.where(valid, p, 1.)
    order = np.argsort(work, kind="stable")
    adjusted = np.minimum.accumulate((work[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1.)
    result[~valid] = np.nan
    return result


def group_test(patient, control, cfg, seed):
    patient, control = np.asarray(patient, float), np.asarray(control, float)
    result = dict(patient_total=len(patient), hc_total=len(control))
    patient, control = patient[np.isfinite(patient)], control[np.isfinite(control)]
    result.update(patient_valid=len(patient), hc_valid=len(control))
    for label, values in [("patient", patient), ("hc", control)]:
        result[f"{label}_median"] = float(np.median(values)) if len(values) else np.nan
        result[f"{label}_q1"] = float(np.quantile(values, .25)) if len(values) else np.nan
        result[f"{label}_q3"] = float(np.quantile(values, .75)) if len(values) else np.nan
    if min(len(patient), len(control)) < cfg["min_group_n"]:
        return dict(result, p=np.nan, rank_biserial=np.nan, method="untestable", reason="insufficient_finite_observations")
    pooled = np.r_[patient, control]
    ties = len(np.unique(pooled)) < len(pooled)
    if np.ptp(pooled) == 0:
        return dict(result, p=1., rank_biserial=0., method="constant_null", reason="all_values_equal")
    if min(len(patient), len(control)) <= 8 and not ties:
        method, name = "exact", "exact_no_ties"
    elif min(len(patient), len(control)) < 10 and ties:
        method = PermutationMethod(n_resamples=cfg["permutations"], batch=256, random_state=seed)
        name = f"permutation_{cfg['permutations']}"
    else:
        method, name = "asymptotic", "asymptotic_tie_corrected_continuity"
    statistic = mannwhitneyu(patient, control, alternative="two-sided", method=method, use_continuity=True)
    return dict(result, p=float(statistic.pvalue), rank_biserial=float(2 * statistic.statistic / (len(patient) * len(control)) - 1), method=name, reason="", seed=seed)


def standardized(values, hc, orientation, tolerance):
    """Strict complete-HC reference per component; no changing reference membership."""
    values = np.asarray(values, float)
    reference = values[np.asarray(hc, bool)]
    shape = values.shape[1:]
    if len(reference) < 2:
        return np.full(values.shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    mean, sd = reference.mean(0), reference.std(0, ddof=1)
    valid = np.isfinite(reference).all(0) & np.isfinite(sd) & (sd > tolerance)
    scores = np.divide(values - mean, sd, out=np.full(values.shape, np.nan), where=valid)
    return scores * np.asarray(orientation), mean, sd


def matrix_composite(scores):
    values = .5 * (scores[:, :5].mean(1) + scores[:, 5:].mean(1))
    return np.where(np.isfinite(scores).all(1), values, np.nan)


def density_selection(cfg, reference, index):
    results = []
    for density in cfg["densities"]:
        rmses, seen = [], []
        for record in index:
            if record["architecture"] != "set" or record["density"] != density:
                continue
            with np.load(Path(record["directory"]) / "test_predictions.npz") as bundle:
                ids = bundle["indices"]
                if not np.array_equal(reference.ids[ids], bundle["subject_ids"]):
                    raise ValueError("Misaligned OOF predictions")
                difference = reference.fc[ids].astype(float) - bundle["reconstruction"]
                rmses.extend(np.sqrt(np.mean(difference**2, axis=(1, 2))).tolist())
                seen.extend(ids.tolist())
        if sorted(seen) != list(range(len(reference.fc))):
            raise ValueError("Density selection requires exactly one OOF prediction per reference participant")
        results.append(dict(density=density, participant_mean_rmse=float(np.mean(rmses)), n=len(rmses)))
    selected = min(results, key=lambda r: (r["participant_mean_rmse"], r["density"]))["density"]
    return selected, results


def clinical_analysis(cfg, cohort, rois, matrix, regional, directory, architecture):
    directory = Path(directory)
    hc = cohort.groups == "HC"
    patient = ~hc
    tests = []
    for metric in MATRIX:
        result = group_test(matrix.loc[patient, metric], matrix.loc[hc, metric], cfg["statistics"], seed_for(cfg["seed"], "matrix_test", architecture, metric))
        tests.append(dict(metric=metric, **result))
    tests = pd.DataFrame(tests)
    tests["q"] = bh_adjust(tests.p)
    tests["reject"] = np.isfinite(tests.q) & (tests.q < cfg["statistics"]["fdr"])
    tests["synthetic"] = cfg["synthetic"]
    tests.to_csv(directory / "matrix_group_tests.csv", index=False)
    array = np.stack([regional.loc[regional.subject_id == identifier, REGION].to_numpy(float) for identifier in cohort.ids])
    if array.shape != (len(cohort.ids), len(rois), len(REGION)):
        raise ValueError("Regional metric shape mismatch")
    groups = {"all_patients": patient, "BD": cohort.groups == "BD", "MDD": cohort.groups == "MDD"}
    rows = []
    for name, selected in groups.items():
        for m, metric in enumerate(REGION):
            family = []
            for node in range(len(rois)):
                result = group_test(array[selected, node, m], array[hc, node, m], cfg["statistics"], seed_for(cfg["seed"], "regional_test", architecture, name, metric, node))
                family.append(dict(comparison=name, metric=metric, roi_id=rois.roi_id.iloc[node], **result))
            adjusted = bh_adjust([r["p"] for r in family])
            for row, q in zip(family, adjusted):
                rows.append(dict(row, q=q, reject=bool(np.isfinite(q) and q < cfg["statistics"]["fdr"]), synthetic=cfg["synthetic"]))
    pd.DataFrame(rows).to_csv(directory / "regional_group_tests.csv", index=False)
    orientation = np.array([-1 if m in ["pearson", "spearman", "symmetry"] else 1 for m in MATRIX])
    scores, mean, sd = standardized(matrix[MATRIX].to_numpy(float), hc, orientation, cfg["statistics"]["variance_tolerance"])
    composite = matrix_composite(scores)
    composite[hc] = np.nan  # Patient descriptive outcome only.
    score_table = pd.DataFrame(scores, columns=[f"{m}_z" for m in MATRIX])
    score_table["subject_id"], score_table["group"], score_table["composite"], score_table["synthetic"] = cohort.ids, cohort.groups, composite, cfg["synthetic"]
    score_table["valid_components"] = np.isfinite(scores).sum(1)
    score_table.to_csv(directory / "matrix_standardized_scores.csv", index=False)
    pd.DataFrame(dict(metric=MATRIX, hc_mean=mean, hc_sample_sd=sd, complete_hc_reference=np.isfinite(matrix.loc[hc, MATRIX].to_numpy(float)).all(0))).to_csv(directory / "matrix_reference_statistics.csv", index=False)
    regional_z, regional_mean, regional_sd = standardized(array, hc, np.array([-1, -1, 1, 1, 1, 1, 1, 1]), cfg["statistics"]["variance_tolerance"])
    subject_scores = np.abs(regional_z).mean(2)  # NA propagates, all eight required.
    np.savez_compressed(directory / "regional_standardized_scores.npz", subject_ids=cohort.ids, roi_ids=rois.roi_id.to_numpy(dtype=str), scores=regional_z, absolute_metric_mean=subject_scores, hc_mean=regional_mean, hc_sample_sd=regional_sd)
    spatial, networks = [], []
    for name, selected in groups.items():
        values = subject_scores[selected].mean(0) if selected.any() else np.full(len(rois), np.nan)
        complete_brain = np.isfinite(values).all()
        threshold = values.mean() + values.std(ddof=1) if complete_brain else np.nan
        for node, value in enumerate(values):
            spatial.append(dict(comparison=name, roi_id=rois.roi_id.iloc[node], network=rois.network.iloc[node], score=value, patients_total=int(selected.sum()), patients_valid=int(np.isfinite(subject_scores[selected, node]).sum()), whole_brain_threshold=threshold, synthetic=cfg["synthetic"]))
        for network in rois.network.unique():
            mask = rois.network.to_numpy() == network
            subset = values[mask]
            complete = np.isfinite(subset).all()
            networks.append(dict(comparison=name, network=network, mean=float(subset.mean()) if complete else np.nan, sample_sd=float(subset.std(ddof=1)) if complete and len(subset) > 1 else np.nan, fraction_above_threshold=float(np.mean(subset > threshold)) if complete and complete_brain else np.nan, rois_total=len(subset), rois_valid=int(np.isfinite(subset).sum()), synthetic=cfg["synthetic"]))
    pd.DataFrame(spatial).to_csv(directory / "regional_group_scores.csv", index=False)
    pd.DataFrame(networks).to_csv(directory / "network_summaries.csv", index=False)
    hub_summary = regional.groupby(["group", "roi_id", "hub_status"]).size().rename("count").reset_index()
    hub_summary["synthetic"] = cfg["synthetic"]
    hub_summary.to_csv(directory / "hub_status_counts.csv", index=False)
    write_json(directory / "score_validity.json", {"patient_composite_valid": int(np.isfinite(composite[patient]).sum()), "patient_total": int(patient.sum()), "regional_patient_scores_valid": int(np.isfinite(subject_scores[patient]).sum()), "missing_policy": "all HC per component, all eight metrics per regional subject, all patients per regional group; no reweighting", "synthetic": cfg["synthetic"]})


def run_external(cfg, reference, external, rois, output, device):
    output = Path(output)
    index = read_json(output / "cv_index.json")
    density, scores = density_selection(cfg, reference, index)
    write_json(output / "deployment_density.json", {"density": density, "reference_oof_scores": scores, "selection_uses_external": False, "synthetic": cfg["synthetic"]})
    a = adjacency_batch(external.fc, density)
    dataset = tensors(external, a, range(len(external.fc)))
    evaluator = TopologyEvaluator(cfg["topology"], cfg["seed"], output / "graph_cache")
    for architecture in cfg["architectures"]:
        directory = output / "external" / architecture
        if (directory / "complete.json").exists():
            continue
        print(f"External: {architecture}, reference-selected density={density}", flush=True)
        records = sorted([r for r in index if r["architecture"] == architecture and r["density"] == density], key=lambda r: r["fold"])
        if [r["fold"] for r in records] != list(range(cfg["cv"]["outer"])):
            raise ValueError("Missing or duplicate ensemble members")
        total = np.zeros(external.fc.shape, dtype=np.float64)
        for record in records:
            model = load_model(record["final"]["checkpoint"], device)
            total += predict(model, dataset, cfg["training"]["batch_size"], device)
        ensemble = total / len(records)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "ensemble_predictions.npz", subject_ids=external.ids, reconstruction=ensemble)
        matrix, regional = evaluate_predictions(cfg, external, rois, range(len(external.fc)), ensemble, density, evaluator, directory, dict(architecture=architecture, density=density))
        clinical_analysis(cfg, external, rois, matrix, regional, directory, architecture)
        write_json(directory / "complete.json", {"members": [r["final"]["checkpoint"] for r in records], "density": density, "synthetic": cfg["synthetic"]})
