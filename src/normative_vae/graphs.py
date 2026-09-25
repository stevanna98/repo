"""Global, loop-free, positive-edge proportional thresholding."""
import numpy as np
import torch


class InsufficientPositiveEdges(ValueError):
    pass


def adjacency(matrix, density):
    x = np.asarray(matrix)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected a finite square matrix")
    if not 0 < density <= 1:
        raise ValueError("Density must be in (0,1]")
    if not np.allclose(x, x.T, atol=1e-5, rtol=0):
        raise ValueError("Thresholding expects symmetric input")
    n = len(x)
    i, j = np.triu_indices(n, 1)  # Lexicographic order.
    w = x[i, j]
    m = int(np.floor(density * n * (n - 1) / 2))
    if np.count_nonzero(w > 0) < m:
        raise InsufficientPositiveEdges(f"Need {m} positive pairs, found {np.count_nonzero(w > 0)}")
    selected = np.argsort(-w, kind="stable")[:m]
    a = np.zeros((n, n), dtype=bool)
    a[i[selected], j[selected]] = True
    a[j[selected], i[selected]] = True
    return a


def adjacency_batch(fc, density):
    values = []
    for i, x in enumerate(fc):
        try:
            values.append(adjacency(x, density))
        except InsufficientPositiveEdges as exc:
            raise InsufficientPositiveEdges(f"Subject index {i}, density {density}: {exc}") from exc
    return np.stack(values)


def normalized_adjacency(a):
    a = a.float()
    degrees = a.sum(-1)
    inv = degrees.clamp_min(1).rsqrt() * (degrees > 0)
    return inv.unsqueeze(-1) * a * inv.unsqueeze(-2)


def additive_mask(a):
    return torch.zeros_like(a, dtype=torch.float32).masked_fill(~a.bool(), float("-inf"))


def safe_attention(scores, allowed):
    """Masked row-softmax; empty rows have identically zero weights and gradients."""
    if allowed is None:
        return torch.softmax(scores, dim=-1)
    allowed = allowed.bool()
    nonempty = allowed.any(-1, keepdim=True)
    masked = scores.masked_fill(~allowed, float("-inf"))
    safe = torch.where(nonempty, masked, torch.zeros_like(masked))
    return torch.softmax(safe, -1).masked_fill(~allowed, 0)

