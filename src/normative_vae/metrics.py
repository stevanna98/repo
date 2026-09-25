"""Metrics on untouched decoder outputs, using explicit undefined values."""
import numpy as np
from scipy.stats import rankdata

RECON = ["pearson", "spearman", "rmse", "mae", "symmetry"]
REG_RECON = ["pearson", "spearman", "rmse", "mae"]


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a * b).sum(-1), denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def reconstruction_metrics(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Metric inputs must be matching matrices")
    a, b = x.ravel(), y.ravel()
    difference = x - y
    scalar = dict(pearson=float(correlation(a, b)), spearman=float(correlation(rankdata(a), rankdata(b))), rmse=float(np.sqrt(np.mean(difference**2))), mae=float(np.mean(np.abs(difference))), frobenius=float(np.linalg.norm(difference) / np.linalg.norm(x)) if np.linalg.norm(x) > 0 else np.nan, symmetry=float(1 - np.mean(np.abs(y - y.T))))
    regional = dict(pearson=correlation(x, y), spearman=correlation(rankdata(x, axis=1), rankdata(y, axis=1)), rmse=np.sqrt(np.mean(difference**2, axis=1)), mae=np.mean(np.abs(difference), axis=1))
    return scalar, regional

