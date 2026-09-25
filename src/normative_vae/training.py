"""Single-run optimization, exact best checkpoints, and deterministic evaluation."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .losses import Objective
from .models import VAE, make_model
from .utils import set_seed, seed_for, save_checkpoint, load_checkpoint, rng_state, restore_rng, write_json, read_json


@dataclass
class Stopping:
    patience: int
    delta: float
    best: float = float("inf")
    anchor: float = float("inf")
    bad: int = 0
    best_epoch: int = 0

    def update(self, score, epoch):
        improved = score < self.best
        if improved:
            self.best, self.best_epoch = score, epoch
        if score < self.anchor - self.delta:
            self.anchor, self.bad = score, 0
        else:
            self.bad += 1
        return improved, self.bad >= self.patience


def loader(dataset, batch_size, seed=0, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def mean_rmse(model, dataset, batch_size, device):
    model.eval()
    total, count = 0., 0
    for x, a in loader(dataset, batch_size):
        x, a = x.to(device), a.to(device)
        prediction = model(x, a)[0]
        values = (x - prediction).square().mean((1, 2)).sqrt()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite validation predictions")
        total += values.double().sum().item()
        count += len(x)
    return total / count


@torch.no_grad()
def predict(model, dataset, batch_size, device, latent=False):
    model.eval()
    values = []
    for x, a in loader(dataset, batch_size):
        x, a = x.to(device), a.to(device)
        value = model.encode(x, a)[0] if latent else model(x, a)[0]
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite evaluation output")
        values.append(value.cpu().numpy())
    return np.concatenate(values)


def load_model(path, device):
    checkpoint = load_checkpoint(path)
    model = VAE(**checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def train_vae(cfg, architecture, candidate, training, validation, directory, seed, device, epochs=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        return read_json(directory / "complete.json")
    set_seed(seed)
    model = make_model(cfg, architecture, candidate).to(device)
    settings = cfg["training"]
    objective = Objective(settings["pearson_eps"]).to(device)
    optimizer = torch.optim.AdamW([{ "params": model.parameters(), "weight_decay": candidate["weight_decay"]}, {"params": objective.parameters(), "weight_decay": 0.}], lr=candidate["lr"], betas=tuple(settings["betas"]), eps=settings["adam_eps"])
    stop = Stopping(settings["patience"], settings["min_delta"])
    history, start = [], 1
    limit = epochs if epochs is not None else settings["max_epochs"]
    if limit < 1 or (epochs is not None and validation is not None):
        raise ValueError("Fixed-duration refits must have positive epochs and no validation partition")
    if (directory / "last.pt").exists():
        saved = load_checkpoint(directory / "last.pt", device)
        if saved["seed"] != seed or saved["model_spec"] != model.spec or saved["epoch_limit"] != limit:
            raise ValueError("Checkpoint does not match requested fit")
        model.load_state_dict(saved["model"])
        objective.load_state_dict(saved["objective"])
        optimizer.load_state_dict(saved["optimizer"])
        stop = Stopping(**saved["stopping"])
        history, start = saved["history"], saved["epoch"] + 1
        restore_rng(saved["rng"])
    for epoch in range(start, limit + 1):
        if validation is not None and stop.bad >= stop.patience:
            break
        model.train()
        totals, count = {}, 0
        for x, a in loader(training, settings["batch_size"], seed_for(seed, "batch", epoch), True):
            x, a = x.to(device), a.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(x, a)
            loss, terms = objective(x, reconstruction, mu, logvar)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {directory}, epoch {epoch}; no clipping/repair applied")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(objective.parameters()), settings["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            for k, v in terms.items():
                totals[k] = totals.get(k, 0.) + float(v.detach()) * len(x)
            count += len(x)
        row = {"epoch": epoch, **{k: v / count for k, v in totals.items()}, "alpha_end": float(objective.alpha_logit.sigmoid().detach()), "beta_end": float(objective.beta_logit.sigmoid().detach())}
        best, stopped = False, False
        if validation is not None:
            row["validation_rmse"] = mean_rmse(model, validation, settings["batch_size"], device)
            best, stopped = stop.update(row["validation_rmse"], epoch)
        history.append(row)
        state = dict(model_spec=model.spec, model=model.state_dict(), objective=objective.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, epoch_limit=limit, seed=seed, stopping=vars(stop), history=history, rng=rng_state())
        if best:
            save_checkpoint(directory / "best.pt", state)
        save_checkpoint(directory / "last.pt", state)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        if stopped:
            break
    result = dict(checkpoint=str(directory / ("best.pt" if validation is not None else "last.pt")), best_epoch=stop.best_epoch if validation is not None else history[-1]["epoch"], score=stop.best if validation is not None else None, epochs_run=history[-1]["epoch"], hit_epoch_cap=history[-1]["epoch"] == limit, validation_n=len(validation) if validation is not None else 0, parameter_count=sum(p.numel() for p in model.parameters()), seed=seed)
    write_json(directory / "complete.json", result)
    return result

