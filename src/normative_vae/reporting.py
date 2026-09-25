"""Auditable evaluation tables and descriptive figures; no architecture p-values."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import reconstruction_metrics
from .topology import TopologyEvaluator
from .graphs import adjacency_batch
from .latent_analysis import analyze_latent
from .utils import write_json, read_json


def evaluate_predictions(cfg, cohort, rois, indices, predictions, density, evaluator, directory, tag):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, region_rows, audits = [], [], []
    for index, prediction in zip(indices, predictions):
        scalar, regional = reconstruction_metrics(cohort.fc[index], prediction)
        top, nodal, audit = evaluator.compare(cohort.fc[index], prediction, density)
        scalar.update(top)
        regional.update(nodal)
        row = dict(subject_id=cohort.ids[index], group=cohort.groups[index], synthetic=cfg["synthetic"], **tag, **scalar)
        for metric, value in scalar.items():
            row[f"{metric}_valid"] = bool(np.isfinite(value))
        row["graph_valid"] = audit["graph_valid"]
        row["graph_reason"] = audit["reason"]
        rows.append(row)
        part = rois.copy()
        for key, value in regional.items():
            part[key] = value
            part[f"{key}_valid"] = np.isfinite(value)
        part["subject_id"], part["group"], part["hub_status"], part["synthetic"] = cohort.ids[index], cohort.groups[index], audit["status"], cfg["synthetic"]
        for key, value in tag.items():
            part[key] = value
        region_rows.append(part)
        audits.append(dict(subject_id=cohort.ids[index], undefined_scalar=[k for k, v in scalar.items() if not np.isfinite(v)], undefined_regional={k: int(np.count_nonzero(~np.isfinite(v))) for k, v in regional.items()}, **audit))
    pd.DataFrame(rows).to_csv(directory / "matrix_metrics.csv", index=False)
    pd.concat(region_rows, ignore_index=True).to_csv(directory / "regional_metrics.csv", index=False)
    write_json(directory / "graph_audit.json", audits)
    return pd.DataFrame(rows), pd.concat(region_rows, ignore_index=True)


def run_reference_analysis(cfg, cohort, rois, output, device):
    output = Path(output)
    index = read_json(output / "cv_index.json")
    evaluator = TopologyEvaluator(cfg["topology"], cfg["seed"], output / "graph_cache")
    adjacency_cache = {}
    latent_rows = []
    for record in index:
        directory = Path(record["directory"])
        print(f"Analysis: {record['architecture']} density={record['density']} fold={record['fold']}", flush=True)
        evaluation = directory / "evaluation"
        if not (evaluation / "complete.json").exists():
            with np.load(directory / "test_predictions.npz") as bundle:
                if not np.array_equal(cohort.ids[bundle["indices"]], bundle["subject_ids"]):
                    raise ValueError("Prediction IDs do not match source participants")
                evaluate_predictions(cfg, cohort, rois, bundle["indices"], bundle["reconstruction"], record["density"], evaluator, evaluation, {k: record[k] for k in ["architecture", "density", "fold"]})
            write_json(evaluation / "complete.json", {"synthetic": cfg["synthetic"]})
        if record["density"] not in adjacency_cache:
            adjacency_cache[record["density"]] = adjacency_batch(cohort.fc, record["density"])
        latent = analyze_latent(cfg, cohort, rois, adjacency_cache[record["density"]], record, device)
        for partition, metrics in latent["probe"].items():
            latent_rows.append(dict(architecture=record["architecture"], density=record["density"], fold=record["fold"], analysis=f"probe_{partition}", synthetic=cfg["synthetic"], **metrics))
        latent_rows.append(dict(architecture=record["architecture"], density=record["density"], fold=record["fold"], analysis="clustering", synthetic=cfg["synthetic"], **latent["clustering"]))
    pd.DataFrame(latent_rows).to_csv(output / "latent_fold_metrics.csv", index=False)


def descriptive_tables(table, metrics, group_columns):
    """Fold means first; SD across available fold means (not a confidence interval)."""
    pieces = []
    for metric in metrics:
        fold = table.groupby(group_columns + ["fold"], dropna=False)[metric].agg(["mean", "count"]).reset_index()
        summary = fold.groupby(group_columns, dropna=False).agg(mean=("mean", "mean"), fold_sd=("mean", "std"), valid_folds=("mean", "count"), valid_observations=("count", "sum")).reset_index()
        summary["metric"] = metric
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True)


def make_reports(cfg, output):
    output = Path(output)
    index = read_json(output / "cv_index.json")
    tables = [pd.read_csv(Path(r["directory"]) / "evaluation" / "matrix_metrics.csv") for r in index]
    pd.DataFrame([dict(architecture=r["architecture"], density=r["density"], fold=r["fold"], parameter_count=r["final"]["parameter_count"], epochs=r["final"]["epochs_run"], synthetic=cfg["synthetic"]) for r in index]).to_csv(output / "model_sizes_and_durations.csv", index=False)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(output / "reference_participant_metrics.csv", index=False)
    metrics = [c[:-6] for c in table.columns if c.endswith("_valid") and c != "graph_valid"]
    validity = []
    for (architecture, density), group in table.groupby(["architecture", "density"]):
        for metric in metrics:
            validity.append(dict(architecture=architecture, density=density, metric=metric, total=len(group), valid=int(np.isfinite(group[metric]).sum()), undefined=int((~np.isfinite(group[metric])).sum()), synthetic=cfg["synthetic"]))
    pd.DataFrame(validity).to_csv(output / "reference_validity_counts.csv", index=False)
    summary = descriptive_tables(table, metrics, ["architecture", "density"])
    summary["synthetic"] = cfg["synthetic"]
    summary.to_csv(output / "reference_fold_summary.csv", index=False)
    all_pairs = []
    keys = ["subject_id", "density", "fold"]
    base = table.loc[table.architecture == "set", keys + ["rmse"]]
    for architecture in ["gat", "gcn"]:
        paired = base.merge(table.loc[table.architecture == architecture, keys + ["rmse"]], on=keys, validate="one_to_one", suffixes=("_set", "_baseline"))
        paired["difference"] = paired.rmse_set - paired.rmse_baseline
        paired["baseline"], paired["synthetic"] = architecture, cfg["synthetic"]
        all_pairs.append(paired)
    pairs = pd.concat(all_pairs, ignore_index=True)
    pairs.to_csv(output / "paired_rmse_differences.csv", index=False)
    paired_summary = descriptive_tables(pairs, ["difference"], ["baseline", "density"])
    paired_summary["synthetic"] = cfg["synthetic"]
    paired_summary.to_csv(output / "paired_rmse_summary.csv", index=False)
    latent = pd.read_csv(output / "latent_fold_metrics.csv")
    latent_metrics = [c for c in latent.select_dtypes(include="number").columns if c not in ["density", "fold", "synthetic"]]
    ls = descriptive_tables(latent, latent_metrics, ["architecture", "density", "analysis"])
    ls["synthetic"] = cfg["synthetic"]
    ls.to_csv(output / "latent_summary.csv", index=False)
    plots = output / "figures"
    plots.mkdir(exist_ok=True)
    label = "SYNTHETIC SOFTWARE TEST — " if cfg["synthetic"] else ""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for architecture in cfg["architectures"]:
        s = summary[(summary.architecture == architecture) & (summary.metric == "rmse")].sort_values("density")
        axes[0].errorbar(s.density, s["mean"], yerr=s.fold_sd, marker="o", label=architecture.upper())
        h = summary[(summary.architecture == architecture) & (summary.metric == "hub_distance")].sort_values("density")
        axes[1].plot(h.density, h["mean"], marker="o", label=architecture.upper())
    axes[0].set_ylabel("RMSE (fold mean ± fold SD)")
    axes[1].set_ylabel("Hub Jaccard distance")
    for ax in axes:
        ax.set_xlabel("Edge density")
        ax.legend()
    fig.suptitle(label + "Descriptive reconstruction and hub fidelity")
    fig.savefig(plots / "reconstruction_hubs.png", dpi=180)
    plt.close(fig)
    for record in index:
        directory = Path(record["directory"])
        history = pd.read_csv(directory / "refit" / "history.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
        axes[0].plot(history.epoch, history.alpha_end, label="alpha")
        axes[0].plot(history.epoch, history.beta_end, label="beta")
        for metric in ["mse", "weighted_pearson", "weighted_kl"]:
            axes[1].plot(history.epoch, history[metric], label=metric)
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend()
        fig.suptitle(label + f"{record['architecture']} κ={record['density']} fold={record['fold']}")
        fig.savefig(directory / "loss_weights.png", dpi=140)
        plt.close(fig)
        clusters = pd.read_csv(directory / "latent" / "clusters.csv")
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.scatter(clusters.tsne_1, clusters.tsne_2, c=clusters.cluster, cmap="tab20", s=12)
        heading = "SYNTHETIC SOFTWARE TEST\n" if cfg["synthetic"] else ""
        ax.set_title(heading + "Regional t-SNE\nColors: latent K-means", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(directory / "latent" / "tsne.png", dpi=140)
        plt.close(fig)
    report = f"# {'SYNTHETIC SOFTWARE VERIFICATION' if cfg['synthetic'] else 'Research analysis'}\n\nDescriptive architecture comparisons only; fold SD is not a confidence interval.\n\n{len(index)} outer refits evaluated. Missing values and valid counts are retained.\n\nNo empirical or clinical superiority claim is made. See CSV artifacts and graph_audit.json for undefined outcomes.\n"
    (output / "REPORT.md").write_text(report)
    write_json(output / "report_complete.json", {"synthetic": cfg["synthetic"], "outer_refits": len(index)})
