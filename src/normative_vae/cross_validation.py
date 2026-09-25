"""Shared subject splits, independent random search, and leakage-free outer predictions."""
import itertools
import math
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .data import tensors
from .graphs import adjacency_batch
from .training import train_vae, load_model, predict
from .utils import seed_for, write_json, read_json


def make_splits(labels, outer, inner, seed):
    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2 or np.bincount(labels.astype(int)).min() < outer:
        raise ValueError("Both sex classes need enough participants for outer stratification")
    result = []
    for k, (development, test) in enumerate(StratifiedKFold(outer, shuffle=True, random_state=seed_for(seed, "outer")).split(np.zeros(len(labels)), labels)):
        if np.bincount(labels[development].astype(int)).min() < inner:
            raise ValueError("Insufficient classes for inner folds")
        folds = []
        for tr, va in StratifiedKFold(inner, shuffle=True, random_state=seed_for(seed, "inner", k)).split(development, labels[development]):
            folds.append(dict(train=development[tr].tolist(), validation=development[va].tolist()))
        result.append(dict(development=development.tolist(), test=test.tolist(), inner=folds))
    return result


def candidates(cfg, architecture, seed):
    keys = ["lr", "weight_decay", "dropout"]
    if architecture in {"set", "gat"}:
        keys += ["heads"]
    if architecture in {"gat", "gcn"}:
        keys += ["layers"]
    pool = [dict(zip(keys, values)) for values in itertools.product(*(cfg["search"][k] for k in keys))]
    count = cfg["cv"]["candidates"]
    if count > len(pool):
        raise ValueError("Search budget exceeds unique candidate space")
    return [pool[i] for i in np.random.default_rng(seed).choice(len(pool), count, replace=False)]


def candidate_score(results):
    return float(np.average([r["score"] for r in results], weights=[r["validation_n"] for r in results]))


def refit_epochs(results):
    return int(math.ceil(np.median([r["best_epoch"] for r in results])))


def run_cv(cfg, cohort, output, device):
    output = Path(output)
    if cohort.name != "reference" or not np.all(cohort.groups == "HC"):
        raise ValueError("VAE training only accepts reference HC")
    splits = make_splits(cohort.sex, cfg["cv"]["outer"], cfg["cv"]["inner"], cfg["seed"])
    write_json(output / "splits.json", {"subject_ids": cohort.ids, "folds": splits, "seeds": {"outer": seed_for(cfg["seed"], "outer"), "inner": [seed_for(cfg["seed"], "inner", k) for k in range(len(splits))]}})
    index = []
    for density in cfg["densities"]:
        a = adjacency_batch(cohort.fc, density)
        for architecture in cfg["architectures"]:
            for k, split in enumerate(splits):
                directory = output / architecture / f"density_{density:.1f}" / f"fold_{k}"
                directory.mkdir(parents=True, exist_ok=True)
                if (directory / "fold.json").exists():
                    index.append(read_json(directory / "fold.json"))
                    continue
                print(f"CV: {architecture}, density={density}, outer={k + 1}/{len(splits)}", flush=True)
                search_seed = seed_for(cfg["seed"], "search", architecture, density, k)
                configs = candidates(cfg, architecture, search_seed)
                write_json(directory / "candidates.json", {"seed": search_seed, "configurations": configs})
                results = []
                for trial, config in enumerate(configs):
                    fits = []
                    for j, inner in enumerate(split["inner"]):
                        fits.append(train_vae(cfg, architecture, config, tensors(cohort, a, inner["train"]), tensors(cohort, a, inner["validation"]), directory / "search" / f"trial_{trial}" / f"inner_{j}", seed_for(cfg["seed"], "fit", architecture, density, k, trial, j), device))
                    results.append(dict(candidate=config, inner=fits, score=candidate_score(fits)))
                    write_json(directory / "search_results.json", results)
                chosen_index = min(range(len(results)), key=lambda i: (results[i]["score"], i))
                selected = results[chosen_index]
                final = train_vae(cfg, architecture, selected["candidate"], tensors(cohort, a, split["development"]), None, directory / "refit", seed_for(cfg["seed"], "refit", architecture, density, k), device, refit_epochs(selected["inner"]))
                model = load_model(final["checkpoint"], device)
                predictions = predict(model, tensors(cohort, a, split["test"]), cfg["training"]["batch_size"], device)
                np.savez_compressed(directory / "test_predictions.npz", indices=split["test"], subject_ids=cohort.ids[split["test"]], reconstruction=predictions)
                record = dict(architecture=architecture, density=density, fold=k, directory=str(directory), split=split, selected_candidate=selected["candidate"], selected_index=chosen_index, inner_fits=selected["inner"], final=final)
                write_json(directory / "fold.json", record)
                index.append(record)
                write_json(output / "cv_index.partial.json", index)
    write_json(output / "cv_index.json", index)
    return index
