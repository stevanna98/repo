# Verification record — 24 September 2026

## Scope

Software verification used synthetic data only. No HCP/Policlinico observations
were accessed, no full 2,700-candidate research search was launched, and no
empirical reconstruction or clinical superiority claim is supported by these runs.
The manuscript archive was read but not changed.

## Environment

- macOS arm64; Python 3.10.0.
- PyTorch 2.6.0; NumPy 1.26.4; SciPy 1.14.1; scikit-learn 1.5.2.
- pandas 2.2.3; matplotlib 3.9.2; PyYAML 6.0.2; threadpoolctl 3.5.0.
- igraph 0.11.9; leidenalg 0.10.2; pytest 8.3.3.
- **CPU execution. CUDA is unavailable and has not been tested.**
- Project-local environment reused installed scientific dependencies and added
  graph dependencies locally. The old local pip was upgraded to support PEP 660
  editable installation. `pip install --no-deps -e .` succeeded; the installed
  package imported successfully from outside the repository.

These are tested versions, not a claim of fresh-install verification on Linux
or of bitwise reproducibility across platforms.

## Automated tests

Command: `.venv/bin/python -m pytest -q`

**26 passed**, most recent run approximately 2.6 seconds. Tests include:

- 379-region model forward/backward and deterministic evaluation for all models.
- Original GAT hand calculation and loop-free GCN.
- Full-matrix MSE, latent-summed KL, learned-weight gradients and initialization.
- Safe all-masked attention rows, including nonzero output biases.
- Positive-edge ranking, ties, exact counts and isolated nodes.
- PSD synthetic correlation matrices and strict participant/ROI alignment.
- Disjoint, shared nested partitions and spies excluding outer test data from fits.
- Participant-weighted validation RMSE and distinct best-checkpoint/patience logic.
- Exact CPU equality of model and objective states after an interrupted fit resumes.
- Guard against changed run settings and accidental full research execution.
- Known-graph local/global efficiency and hub identity changes despite equal hub density.
- Fixed BH family sizes, effect directions, test ties and missing outcomes.
- Strict HC standardization, complete composite scores, and ensemble metric order.
- Reference-only deployment density selection and lower-density tie handling.

`python -m compileall -q src run.py tests` also succeeded.

## End-to-end integration

Command:

```sh
.venv/bin/python run.py all --config configs/smoke_test.yaml --output outputs/verified-smoke
```

Completed all stages with 12 synthetic reference participants and 8 external
participants (4 HC, 2 BD, 2 MDD). All inputs were **379 × 379**; model widths were
256 and 64. The deliberately reduced design used two outer/two inner folds,
one candidate per architecture/density/fold, and at most two VAE/probe epochs.
All three architectures and all three densities were exercised.

The run includes:

- 36 candidate VAE fits and 18 final outer refits.
- 36 inner probes and 18 final probes.
- 18 development-reference clustering/t-SNE analyses.
- Three external ensembles with two members each (not the research five).
- 108 reference participant–condition records, with no duplicates.
- 72 SET-minus-baseline paired RMSE differences.
- 12 matrix-level tests per external architecture.
- 9,096 regional tests per architecture: 379 ROIs × eight metrics × three contrasts.
- External reconstruction arrays of shape [8,379,379].

Programmatic audits confirmed source hashes matched the final executed code,
each reference participant had exactly one held-out prediction per condition,
inner probe participant identities excluded the corresponding outer test set,
and all clinical correction families had their intended sizes. Repeated seeded
CPU runs produced exactly equal saved reference reconstructions.

The initial integration was cold-cache. The last run reused only deterministic,
content/settings-keyed graph calculations from the previous verified run; VAE
and probe fits were run again. Restart of an already-completed pipeline was
also exercised. Example reconstruction/hub and t-SNE figures were inspected;
a clipped t-SNE title was corrected before the final run.

## Limitations and interpretation

- Real-data ingestion has been tested against the defined schema using generated
  fixtures, not against the user's actual exports or atlas mapping.
- CUDA, real-cohort performance, full-budget runtime, and multi-machine restart
  have not been validated. Checkpoint paths are runtime absolute paths; move the
  source to the intended GPU machine and create a fresh run there.
- Missing reference variances or zero topological denominators intentionally
  produce undefined scores and valid-count reports rather than fabricated values.
- Test coverage is substantial but does not prove scientific or clinical validity.
- `requirements-tested.txt` lists direct tested versions, not every transitive
  dependency. Per-run environment and identity artifacts capture the execution.
- Earlier diagnostic folders `outputs/verification-v1` and
  `outputs/verification-final` are retained separately. `outputs/verified-smoke`
  corresponds to the final source. Generated outputs/environments are not included
  in the source-only delivery archive.
