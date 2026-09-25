# Normative FC VAE research pipeline

Native PyTorch implementation of the approved healthy-reference FC protocol:
SET-VAE, **original** GAT-VAE, and GCN-VAE. The input is precomputed Pearson FC;
MRI preprocessing is deliberately excluded. This is an implementation of a
prospective analysis, not a reproduction of empirical manuscript findings.

## Quick start

Python 3.10+ is required. From this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
python run.py all --config configs/smoke_test.yaml --output outputs/my-smoke
```

The editable installation is optional when using `run.py`, which locates the local
package. Its dependencies must still be installed. For an NVIDIA system, install
the appropriate PyTorch CUDA build using the official selector:
https://pytorch.org/get-started/locally/ . Verify with
`python -c "import torch; print(torch.cuda.is_available())"`.

`requirements-tested.txt` records the directly tested CPU package versions; it is
not a complete cross-platform lockfile. The delivered local `.venv` reuses the
host's installed scientific packages and adds the graph dependencies locally.
Fresh machines should follow the clean-environment instructions above.

`device: auto` selects CUDA when available, otherwise CPU. An unavailable explicit
`--device cuda` request fails rather than silently falling back. MPS is not the
target backend. All initial optimization uses float32; no mixed precision.

The smoke configuration uses 12 synthetic reference participants, 8 external
participants, **379 regions, d=256, d_z=64**, two outer and two inner folds, one
candidate per condition, and two maximum epochs for VAEs/probes. It exercises all
three architectures and densities. These are SOFTWARE CHECKS, not study results.
It can consume roughly 1 GB because optimizer/RNG states and selected inner
models are preserved. Runtime depends on hardware, topology, and permutation
tests. The full research configuration is much more expensive.

## Stage-wise execution and restart

```sh
python run.py generate --config configs/smoke_test.yaml
python run.py validate --config configs/smoke_test.yaml --output outputs/my-smoke
python run.py train --config configs/smoke_test.yaml --output outputs/my-smoke
python run.py analyze --config configs/smoke_test.yaml --output outputs/my-smoke
python run.py external --config configs/smoke_test.yaml --output outputs/my-smoke
python run.py report --config configs/smoke_test.yaml --output outputs/my-smoke
```

Use the same configuration and output directory to restart. Completed fits and
stages are reused; interrupted fits restore optimizer, coefficient, RNG,
history, and stopping states from epoch-boundary checkpoints. Runs reject changed
inputs/configuration/source/package versions. Use a **new output directory** when
those change. Do not load untrusted `.pt` files: these local research checkpoints
include serialized Python/NumPy RNG states.

Generation refuses a nonempty incompatible destination. Do not point the
generator at real data. Source matrices are never repaired or overwritten.

## Running real data later

1. Export data following [data/README.md](data/README.md), with explicit subject
   and ROI identities. No metadata encodings are guessed.
2. Copy `configs/manuscript.yaml`; replace `data_dir` and choose a new output path.
   Retain `synthetic: false`. Paths resolve relative to the repository unless absolute.
3. Run `validate` first and address any reported alignment or input-QC issues.
4. Deliberately launch the research run:

```sh
python run.py all --config configs/my-study.yaml --allow-full-run
```

Do not execute this example until data and compute resources are ready. Full CV
requires **2,700 candidate fits and 45 outer refits**, plus 135 inner and 45 final
probe fits. `--allow-full-run` is required for real-data training or more than 100
candidate fits. No distributed scheduler or automatic cloud execution is used.

## Outputs

Each output directory contains a frozen run identity, resolved settings,
environment/package versions, input validation, and subject-level splits.
Architecture/density/fold directories contain search candidates and scores,
checkpoints, histories, out-of-fold reconstructions, raw reconstruction metrics,
topological comparisons, graph audit records, and latent/probe outputs.

Key aggregate files:

- `reference_participant_metrics.csv`: one test observation per participant and condition.
- `reference_fold_summary.csv`: mean and sample SD of fold means, with valid counts.
- `paired_rmse_differences.csv` / `paired_rmse_summary.csv`: descriptive SET-minus-baseline differences.
- `latent_fold_metrics.csv` / `latent_summary.csv`: probes, train–test gaps and clustering.
- `deployment_density.json`: reference-only SET out-of-fold density selection.
- `external/<architecture>/`: ensemble reconstructions, 12-outcome matrix tests,
  eight separate regional families, standardized patient summaries and hub maps.
- `figures/`, per-fold `loss_weights.png` and `latent/tsne.png`: descriptive figures.
- `REPORT.md`, `DATA_STATUS.txt`: result status and interpretation restrictions.

`*_valid` columns, valid sample counts, graph audit reasons and JSON `null` values
represent undefined outcomes; they are not automatically converted to zero.
CSV missing entries and NPZ NaNs have the same meaning. Percentage errors have
units of percent; hub Jaccard distance is dimensionless and is not a percentage.

## Scientific interpretation

- Reconstruction metrics use all N² raw output entries; only topology uses symmetrization.
- SET decoding receives the subject's input-derived adjacency, whereas graph
  decoders do not. This compares complete architectures, not isolated encoders.
- Matched dimensions do not equalize parameter counts.
- Architecture comparisons are descriptive, without p-values or superiority claims.
- The deployment density is selected from SET reference outcomes; it is not
  necessarily optimal for the baselines and does not provide an unbiased estimate
  of an additional selection step.
- Clinical group tests are exploratory; their separate FDR families do not
  control error across the entire study. Different significance across models is
  not evidence of a between-model difference.
- Synthetic factors, networks, sexes and diagnoses are artificial fixtures.
- No site harmonization or age/motion adjustment is implemented. Scores are not
  calibrated pathological probabilities and cannot establish clinical superiority.
- No claim is made that reference subjects are unrelated. Splits implement the
  currently approved subject-level design.

See [docs/protocol.md](docs/protocol.md) for exact conventions, source mapping,
remaining manuscript updates and numerical policies.
