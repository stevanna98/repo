"""Frozen-encoder probes and development-reference network clustering."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, roc_auc_score
from .data import tensors
from .models import initialize
from .training import Stopping, loader, load_model, predict
from .utils import set_seed, seed_for, save_checkpoint, load_checkpoint, write_json, read_json, rng_state, restore_rng


class Probe(nn.Module):
    def __init__(self, latent=64, width=16, dropout=.4, norm_eps=1e-5):
        super().__init__()
        self.nodes = nn.Sequential(nn.Linear(latent, width), nn.LayerNorm(width, eps=norm_eps), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.score = nn.Linear(width, 1, bias=False)
        self.classifier = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))
        self.apply(initialize)

    def forward(self, z):
        h = self.nodes(z)
        weights = self.score(h).softmax(dim=1)
        return self.classifier((weights * h).sum(1)).squeeze(-1)


def probe_dataset(z, labels):
    return TensorDataset(torch.from_numpy(z.astype(np.float32)), torch.as_tensor(labels, dtype=torch.float32))


@torch.no_grad()
def probe_evaluate(model, data, batch, device):
    model.eval()
    scores, labels, loss_sum = [], [], 0.
    for z, y in loader(data, batch):
        logits = model(z.to(device))
        loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, y.to(device), reduction="sum").item()
        scores.extend(logits.sigmoid().cpu().tolist())
        labels.extend(y.tolist())
    return loss_sum / len(data), np.asarray(scores), np.asarray(labels)


def classification_metrics(labels, scores):
    labels, predicted = np.asarray(labels, bool), np.asarray(scores) >= .5
    tp, fp, fn = np.sum(labels & predicted), np.sum(~labels & predicted), np.sum(labels & ~predicted)
    return dict(accuracy=float(np.mean(labels == predicted)), precision=float(tp / (tp + fp)) if tp + fp else np.nan, recall=float(tp / (tp + fn)) if tp + fn else np.nan, f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else np.nan, auc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else np.nan)


def fit_probe(cfg, train, validation, directory, seed, device, epochs=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = dict(latent=cfg["model"]["latent"], width=cfg["probe"]["width"], dropout=cfg["probe"]["dropout"], norm_eps=cfg["model"]["norm_eps"])
    set_seed(seed)
    model = Probe(**spec).to(device)
    p = cfg["probe"]
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"], betas=tuple(cfg["training"]["betas"]), eps=cfg["training"]["adam_eps"])
    stop = Stopping(p["patience"], p["min_delta"])
    history, start, limit = [], 1, epochs or p["max_epochs"]
    complete = directory / "complete.json"
    if complete.exists():
        result = read_json(complete)
        model.load_state_dict(load_checkpoint(result["checkpoint"])["model"])
        return model.eval(), result
    if (directory / "last.pt").exists():
        state = load_checkpoint(directory / "last.pt", device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        stop = Stopping(**state["stopping"])
        history, start = state["history"], state["epoch"] + 1
        restore_rng(state["rng"])
    for epoch in range(start, limit + 1):
        if validation is not None and stop.bad >= stop.patience:
            break
        model.train()
        total = 0.
        for z, y in loader(train, p["batch_size"], seed_for(seed, "batch", epoch), True):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(z.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probe loss")
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise FloatingPointError("Nonfinite probe gradients")
            opt.step()
            total += float(loss.detach()) * len(z)
        row = dict(epoch=epoch, train_bce=total / len(train))
        improved, stopped = False, False
        if validation is not None:
            row["validation_bce"] = probe_evaluate(model, validation, p["batch_size"], device)[0]
            improved, stopped = stop.update(row["validation_bce"], epoch)
        history.append(row)
        state = dict(model=model.state_dict(), model_spec=spec, optimizer=opt.state_dict(), epoch=epoch, seed=seed, history=history, stopping=vars(stop), rng=rng_state())
        if improved:
            save_checkpoint(directory / "best.pt", state)
        save_checkpoint(directory / "last.pt", state)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        if stopped:
            break
    path = directory / ("best.pt" if validation is not None else "last.pt")
    result = dict(checkpoint=str(path), best_epoch=stop.best_epoch if validation is not None else history[-1]["epoch"], seed=seed, epochs_run=history[-1]["epoch"])
    write_json(complete, result)
    model.load_state_dict(load_checkpoint(path)["model"])
    return model.eval(), result


def analyze_latent(cfg, cohort, rois, a, record, device):
    directory = Path(record["directory"]) / "latent"
    directory.mkdir(exist_ok=True)
    if (directory / "complete.json").exists():
        return read_json(directory / "complete.json")
    split = record["split"]
    batch = cfg["training"]["batch_size"]
    key = (record["architecture"], record["density"], record["fold"])
    durations = []
    for j, (inner, fit) in enumerate(zip(split["inner"], record["inner_fits"])):
        encoder = load_model(fit["checkpoint"], device)
        encoder.requires_grad_(False)
        train_z = predict(encoder, tensors(cohort, a, inner["train"]), batch, device, latent=True)
        val_z = predict(encoder, tensors(cohort, a, inner["validation"]), batch, device, latent=True)
        _, result = fit_probe(cfg, probe_dataset(train_z, cohort.sex[inner["train"]]), probe_dataset(val_z, cohort.sex[inner["validation"]]), directory / f"inner_probe_{j}", seed_for(cfg["seed"], "probe", *key, j), device)
        write_json(directory / f"inner_probe_{j}" / "participants.json", {"encoder_checkpoint": fit["checkpoint"], "train_ids": cohort.ids[inner["train"]], "validation_ids": cohort.ids[inner["validation"]]})
        durations.append(result["best_epoch"])
    encoder = load_model(record["final"]["checkpoint"], device)
    encoder.requires_grad_(False)
    train_z = predict(encoder, tensors(cohort, a, split["development"]), batch, device, latent=True)
    test_z = predict(encoder, tensors(cohort, a, split["test"]), batch, device, latent=True)
    training = probe_dataset(train_z, cohort.sex[split["development"]])
    test = probe_dataset(test_z, cohort.sex[split["test"]])
    model, probe_result = fit_probe(cfg, training, None, directory / "final_probe", seed_for(cfg["seed"], "probe_refit", *key), device, int(math.ceil(np.median(durations))))
    probe_scores = {}
    for partition, dataset, indices in [("development", training, split["development"]), ("test", test, split["test"])]:
        _, probabilities, labels = probe_evaluate(model, dataset, cfg["probe"]["batch_size"], device)
        pd.DataFrame(dict(subject_id=cohort.ids[indices], label=labels, probability=probabilities, predicted=probabilities >= .5, synthetic=cfg["synthetic"])).to_csv(directory / f"probe_{partition}_predictions.csv", index=False)
        probe_scores[partition] = classification_metrics(labels, probabilities)
    probe_scores["development_minus_test"] = {k: probe_scores["development"][k] - probe_scores["test"][k] for k in probe_scores["test"]}
    c = cfg["clustering"]
    reference = train_z.mean(0)
    labels = KMeans(n_clusters=c["clusters"], init="k-means++", n_init=c["n_init"], max_iter=c["max_iter"], tol=c["tol"], random_state=c["seed"]).fit_predict(reference)
    truth = rois.network.to_numpy()
    scores = dict(ari=adjusted_rand_score(truth, labels), nmi=normalized_mutual_info_score(truth, labels, average_method="arithmetic"), ami=adjusted_mutual_info_score(truth, labels, average_method="arithmetic"))
    projection = TSNE(n_components=2, perplexity=c["tsne_perplexity"], init=c["tsne_init"], learning_rate=c["tsne_learning_rate"], max_iter=c["tsne_iterations"], random_state=c["tsne_seed"]).fit_transform(reference)
    table = rois.copy()
    table["cluster"], table["tsne_1"], table["tsne_2"], table["synthetic"] = labels, projection[:, 0], projection[:, 1], cfg["synthetic"]
    table.to_csv(directory / "clusters.csv", index=False)
    np.savez_compressed(directory / "reference_latent.npz", mean=reference, roi_ids=rois.roi_id.to_numpy(dtype=str), development_ids=cohort.ids[split["development"]])
    result = dict(probe=probe_scores, clustering=scores, probe_refit=probe_result, development_ids=cohort.ids[split["development"]], synthetic=cfg["synthetic"])
    write_json(directory / "complete.json", result)
    return result
