"""Binary graph fidelity and ROI-identity-aware hub preservation."""
import hashlib
import json
from pathlib import Path
import igraph as ig
import leidenalg
import numpy as np
from .graphs import adjacency, InsufficientPositiveEdges
from .utils import seed_for, read_json, write_json

SCALARS = ["clustering", "path_length", "global_efficiency", "local_efficiency", "modularity", "participation"]
REGIONAL = ["clustering", "local_efficiency", "nodal_efficiency", "participation"]
TOPO_ERRORS = [f"{k}_ape" for k in SCALARS] + ["hub_distance"]


def canonical_membership(membership):
    mapping = {}
    return tuple(mapping.setdefault(int(v), len(mapping)) for v in membership)


def inverse_distances(graph):
    distances = np.asarray(graph.distances(), dtype=float)
    return np.divide(1., distances, out=np.zeros_like(distances), where=(distances > 0) & np.isfinite(distances))


def graph_statistics(a, cfg, seed):
    a = np.asarray(a, dtype=bool)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.array_equal(a, a.T) or np.any(np.diag(a)):
        raise ValueError("Topology requires binary undirected loop-free adjacency")
    n = len(a)
    graph = ig.Graph(n=n, edges=list(zip(*np.where(np.triu(a, 1)))), directed=False)
    degree = a.sum(1)
    clustering = np.asarray(graph.transitivity_local_undirected(mode="zero"))
    inv = inverse_distances(graph)
    nodal_efficiency = inv.sum(1) / (n - 1) if n > 1 else np.zeros(n)
    local = np.zeros(n)
    for i in range(n):
        neighbors = np.flatnonzero(a[i])
        if len(neighbors) >= 2:
            sub = graph.induced_subgraph(neighbors.tolist())
            local[i] = inverse_distances(sub).sum() / (len(neighbors) * (len(neighbors) - 1))
    components = sorted((sorted(v) for v in graph.connected_components()), key=lambda v: (-len(v), v))
    largest = graph.induced_subgraph(components[0])
    path = float(np.asarray(largest.distances())[np.triu_indices(largest.vcount(), 1)].mean()) if largest.vcount() > 1 else np.nan
    best_q, membership = -np.inf, tuple(range(n))
    seeds = [seed_for(seed, "leiden", i) for i in range(cfg["leiden_runs"])]
    if graph.ecount():
        for current in seeds:
            part = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, n_iterations=-1, seed=current)
            labels = canonical_membership(part.membership)
            q = float(part.quality())
            if q > best_q or (q == best_q and labels < membership):
                best_q, membership = q, labels
    else:
        best_q = np.nan
    membership = np.asarray(membership)
    participation = np.zeros(n)
    if np.any(degree):
        fractions = np.stack([a[:, membership == label].sum(1) / np.maximum(degree, 1) for label in np.unique(membership)], axis=1)
        participation = np.where(degree > 0, 1 - (fractions**2).sum(1), 0.)
    threshold = float(np.quantile(degree, cfg["hub_quantile"], method=cfg["quantile_method"]))
    hubs_valid = bool(graph.ecount())
    hubs = degree >= threshold if hubs_valid else np.zeros(n, dtype=bool)
    return dict(scalar=dict(clustering=float(clustering.mean()), path_length=path, global_efficiency=float(nodal_efficiency.mean()), local_efficiency=float(local.mean()), modularity=best_q, participation=float(participation.mean())), regional=dict(clustering=clustering, local_efficiency=local, nodal_efficiency=nodal_efficiency, participation=participation), membership=membership, degree=degree, hubs=hubs, hubs_valid=hubs_valid, hub_threshold=threshold, lcc_size=largest.vcount(), leiden_seeds=seeds)


def hub_overlap(original, reconstructed, valid=True):
    original, reconstructed = np.asarray(original, bool), np.asarray(reconstructed, bool)
    inter, union = np.sum(original & reconstructed), np.sum(original | reconstructed)
    jaccard = inter / union if union and valid else np.nan
    retention = inter / original.sum() if original.any() and valid else np.nan
    status = np.full(original.shape, "nonhub", dtype="<U12")
    status[original & reconstructed] = "retained"
    status[original & ~reconstructed] = "lost"
    status[~original & reconstructed] = "new"
    if not valid:
        status[:] = "undefined"
    return dict(hub_distance=1 - jaccard, hub_jaccard=jaccard, hub_retention=retention, original_hubs=int(original.sum()), reconstructed_hubs=int(reconstructed.sum()), status=status)


def discrepancies(original, reconstructed, tolerance):
    a, b = np.asarray(original, float), np.asarray(reconstructed, float)
    signed = np.divide(100 * (b - a), np.abs(a), out=np.full(a.shape, np.nan), where=np.isfinite(a) & np.isfinite(b) & (np.abs(a) > tolerance))
    return np.abs(signed), signed, np.abs(b - a)


class TopologyEvaluator:
    def __init__(self, cfg, seed, cache):
        self.cfg, self.seed, self.cache = cfg, seed, Path(cache)

    def graph(self, a):
        identity = json.dumps({"settings": self.cfg, "seed": self.seed}, sort_keys=True).encode()
        key = hashlib.sha256(identity + a.tobytes()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            value = read_json(path)
            value["scalar"] = {k: np.nan if v is None else v for k, v in value["scalar"].items()}
            value["regional"] = {k: np.asarray(v, float) for k, v in value["regional"].items()}
            value["hubs"] = np.asarray(value["hubs"], bool)
            return value
        value = graph_statistics(a, self.cfg, seed_for(self.seed, key))
        write_json(path, value)
        return value

    def compare(self, x, reconstruction, density):
        n = len(x)
        try:
            original = self.graph(adjacency(x, density))
            reconstructed = self.graph(adjacency((reconstruction + reconstruction.T) / 2, density))
        except InsufficientPositiveEdges as exc:
            scalar = {k: np.nan for k in TOPO_ERRORS + ["hub_jaccard", "hub_retention", "original_hubs", "reconstructed_hubs"]}
            regional = {f"{k}_ape": np.full(n, np.nan) for k in REGIONAL}
            return scalar, regional, {"reason": str(exc), "status": np.full(n, "undefined"), "graph_valid": False}
        scalar, regional = {}, {}
        for k in SCALARS:
            av, bv = original["scalar"][k], reconstructed["scalar"][k]
            absolute, signed, raw = discrepancies(av, bv, self.cfg["tolerance"])
            scalar.update({f"{k}_ape": float(absolute), f"{k}_signed_pct": float(signed), f"{k}_absolute_difference": float(raw), f"{k}_original": av, f"{k}_reconstructed": bv})
        for k in REGIONAL:
            regional[f"{k}_ape"] = discrepancies(original["regional"][k], reconstructed["regional"][k], self.cfg["tolerance"])[0]
            regional[f"{k}_original"] = original["regional"][k]
            regional[f"{k}_reconstructed"] = reconstructed["regional"][k]
        hub = hub_overlap(original["hubs"], reconstructed["hubs"], original["hubs_valid"] and reconstructed["hubs_valid"])
        status = hub.pop("status")
        scalar.update(hub)
        scalar.update(original_lcc=original["lcc_size"], reconstructed_lcc=reconstructed["lcc_size"])
        return scalar, regional, dict(status=status, graph_valid=True, reason="", original_membership=original["membership"], reconstructed_membership=reconstructed["membership"], original_degree=original["degree"], reconstructed_degree=reconstructed["degree"], original_hub_threshold=original["hub_threshold"], reconstructed_hub_threshold=reconstructed["hub_threshold"])
