"""Factor-generated correlation fixtures, not a disease or biological-sex model."""
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import seed_for, write_json, read_json


def generate(cfg, directory):
    directory = Path(directory)
    spec = dict(generation=cfg["generation"], seed=seed_for(cfg["seed"], "synthetic"), n_regions=cfg["n_regions"], synthetic=True, generator_version=1)
    if directory.exists() and any(directory.iterdir()):
        if (directory / "provenance.json").exists() and read_json(directory / "provenance.json") == spec:
            if all((directory / f).exists() for f in ["reference.npz", "external.npz", "participants.csv", "rois.csv"]):
                return
        raise FileExistsError("Synthetic destination is nonempty or incomplete; choose a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec["seed"])
    n = cfg["n_regions"]
    networks = np.r_[np.repeat(np.arange(12), 30), np.repeat(12, 19)]
    if n != 379:
        raise ValueError("Synthetic experiment generator requires N=379")
    roi_ids = np.array([f"synthetic_roi_{i:03d}" for i in range(n)])
    pd.DataFrame(dict(index=np.arange(n), roi_id=roi_ids, label=roi_ids, network=[f"synthetic_network_{v:02d}" for v in networks])).to_csv(directory / "rois.csv", index=False)
    settings = cfg["generation"]
    rows = []
    for cohort, groups in [("reference", ["HC"] * settings["reference_n"]), ("external", ["HC"] * settings["external_hc"] + ["BD"] * settings["external_bd"] + ["MDD"] * settings["external_mdd"])]:
        sexes = np.resize(np.array(["F", "M"]), len(groups))
        rng.shuffle(sexes)  # Null association with FC; not a sex classifier benchmark.
        matrices, identifiers = [], []
        for i, group in enumerate(groups):
            t = settings["timepoints"]
            shared = rng.normal(size=(t, 1))
            factors = rng.normal(size=(t, 13))
            loads = rng.lognormal(0, 0.12, n)
            if group != "HC":
                loads[networks == (0 if group == "BD" else 1)] *= 1 + settings["perturbation"]
            signals = 0.35 * shared + factors[:, networks] * loads + rng.normal(size=(t, n)) * 0.8
            matrix = np.corrcoef(signals, rowvar=False).astype(np.float32)
            identifier = f"SYN_{cohort}_{i:04d}"
            matrices.append(matrix)
            identifiers.append(identifier)
            rows.append(dict(subject_id=identifier, cohort=cohort, sex=sexes[i], group=group, synthetic=True))
        np.savez_compressed(directory / f"{cohort}.npz", fc=np.stack(matrices), subject_ids=np.array(identifiers), roi_ids=roi_ids)
    pd.DataFrame(rows).to_csv(directory / "participants.csv", index=False)
    write_json(directory / "provenance.json", spec)

