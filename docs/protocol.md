# Implementation specification and manuscript alignment

## Authority and implementation plan

Source: `graph_vae_normative_modeling_paper_revised_2026-09-23-v3.zip`, Methods,
Introduction, related work and REVISION_NOTES; superseded where specified by the
user's implementation request. The original archive and manuscript are unchanged.
Read-only source archive SHA-256:
`6334305dac7fbcb30ef4da1e67f5a6cd845fffd35eab260f275834fa01a418d4`.
This repository implements a prospective protocol, not completed HCP/Policlinico
experiments. Source provenance/QC, diagnostic ascertainment and ethics remain
researcher responsibilities and cannot be inferred from FC matrices.

Implementation milestones: (1) data/graphs/models/loss, (2) nested fitting and
identity-bearing predictions, (3) probes/clustering/topology/clinical analysis,
(4) analytical tests, (5) 379-region end-to-end synthetic verification. Actual
execution status and test evidence belong in `docs/verification.md`, not here.

## Traceability

| Requirement | Implementation | Main verification |
|---|---|---|
| Fixed FC/ROI identities, no fitted preprocessing | data.py | data alignment and PSD tests |
| Strongest positive edges, global density, no loops | graphs.py | edge counts/ties/isolation |
| Standard then masked SET blocks, post-LayerNorm | attention.py, models.py | 379-region forward/backward |
| Original GAT and D^-1/2 A D^-1/2 GCN | models.py | hand-computed messages/no loops |
| Shared VAE and learned-weight loss | losses.py | exact KL/MSE/scalar gradients |
| Five outer/three inner search and refit | cross_validation.py, training.py | split spies, stopping, exact restart |
| Raw full-matrix fidelity | metrics.py | analytical metric tests |
| Frozen independent probes and regional clustering | latent_analysis.py | synthetic integration/partition provenance |
| Independent partitions, induced-neighbor efficiency | topology.py | known graph tests |
| Hub identity preservation | topology.py | identical densities/different hub identity test |
| Frozen external ensembles and exploratory scores | external.py | ensemble-order, FDR and missing-value tests |
| Descriptive architectural summaries | reporting.py | participant pairs and fold counts |

## Dimensions, architecture, and numerical conventions

All experiment configurations retain N=379, d=256 and d_z=64. Model constructors
allow smaller dimensions for analytical unit tests. Models are native PyTorch,
using dense batched matrices; no PyG, Lightning, pretrained models, or implicit
graph self-loops. Dense computation is deliberate at this graph size.

Affine biases are enabled except GAT's feature projection, whose bias is applied
after aggregation, and the probe's scalar pooling-score projection. Ordinary
linear weights use Xavier uniform, biases zero, LayerNorm scale one and offset
zero, epsilon 1e-5. Original GAT source/target score vectors use Xavier uniform;
LeakyReLU slope is 0.2. GCN also applies any affine bias after aggregation. An
isolated node has zero neighborhood *message*; later affine/normalization
operations are not claimed to leave its representation zero.

SET dropout uses the candidate probability on attention weights, projected
attention contributions, the FFN hidden Mish output, and the FFN contribution
before its residual addition. The two SET output-head linear layers have Mish
between them and no extra dropout. Graph encoders apply LayerNorm–Mish–dropout
after aggregation; original GAT attention coefficients have **no additional
dropout**. Graph decoder hidden layers use Linear–LayerNorm–Mish–dropout.
Posterior and final reconstruction outputs are unconstrained affine layers.

For fully masked rows, scores are replaced by safe zero logits *before* softmax,
and excluded probabilities are explicitly zero. The final masked contribution,
including output bias, is then zeroed before residual addition. No softmax of an
all-negative-infinity row and no NaN replacement are used. Standard attention
can include diagonal interactions; loop exclusion applies to A and masked blocks.

The posterior uses node-wise diagonal Gaussians. Training samples Z; all
validation/testing/inference uses deterministic decoding at mu. The latter is
not the posterior expectation of a nonlinear decoder. SET receives A in both
encoder and decoder; graph decoders receive Z only. Parameter counts are saved,
not equated by matching d and d_z.

## Loss and fitting

MSE averages all B*N² entries. Pearson loss is 1 minus mean participant
correlation, using a denominator product lower bound of 1e-8 **only during
optimization**. Constant-vector evaluation correlations remain undefined.
KL sums the latent coordinates and averages B*N; there is no division by d_z.
No log-variance clipping, KL annealing, coefficient floor or hidden regularizer.
Nonfinite loss/gradients fail explicitly rather than silently skipping subjects.

Effective initial coefficients 0.8 and 0.1 correspond to logits log(4) and
log(1/9). AdamW uses betas (0.9,0.999), epsilon 1e-8, and the selected learning
rate for both groups. Weight decay applies to **all model parameters**, including
biases and normalization, but not coefficient logits. Global L2 clipping at 5
includes both groups. Logs contain batch-participant-weighted epoch loss terms
and the end-of-epoch alpha/beta. Declining coefficients are inspected, not
interpreted automatically as balanced or collapsed learning.

Fixed-duration refits have no validation monitoring. Their histories therefore
do not contain outer-test or validation RMSE. Checkpoint selection uses the exact
minimum validation score; patience uses a distinct anchor improved by >1e-4.
The first best checkpoint wins exact epoch-score ties. Candidate-score ties use
the sampled candidate-list order. Median refit epochs are integers for 3 folds;
for the 2-fold smoke design, fractional medians round upward.

Run restart is at an epoch boundary with optimizer, RNG, coefficient and patience
states. The dedicated data-loader generator is derived from fit seed and epoch.
Completed search results are reused, not recomputed with new random seeds.

## Splits, search, and seeds

Master seed 42. Purpose-specific seeds use the first four SHA-256 bytes of a
JSON-encoded [master,purpose,context...] key modulo 2^31-1. No process-dependent
hash is used. Split and search seeds, checkpoints' initialization seeds,
cluster seeds, permutation seeds, and per-graph Leiden seeds are saved.

Sex-stratified subject-level outer folds are shared across all conditions. Each
outer development set has its own shared stratified inner folds. Only HC in the
reference cohort can enter VAE training. This implementation accepts one matrix
per unique subject and makes no claim about familial independence.

Random search draws 20 unique points uniformly without replacement from the
finite Cartesian product of applicable search settings, separately per
architecture/density/outer fold. Unsupported parameters are not sampled. Every
candidate uses all three inner folds; failures raise rather than allowing
partial-fold candidate rankings. Validation weights participants equally.

All nine reference conditions are reported. Only each participant's own outer
refit produces their reference prediction. External deployment selects the
lowest participant-averaged SET out-of-fold RMSE, lower density resolving ties.
Selection does not access clinical outcomes. Ensembles average reconstructed
matrices in float64, then compute metrics; they do not average metrics or latent
coordinates. The selected reference score is not an unbiased estimate of the
additional deployment-selection step.

## Probe and clustering

Probe node MLP: Linear(64,16)–LayerNorm–ReLU–Dropout(0.4)–Linear(16,16).
Pooling weights are softmax over regions of w^T h_i, a bias-free scalar affine
score; pooled vector is their weighted mean. Classifier is
Linear(16,16)–ReLU–Dropout(0.4)–Linear(16,1). Adam (not AdamW) uses betas
(0.9,0.999), epsilon 1e-8 and coupled weight decay 0.01. No probe gradient clipping,
class weighting, feature scaling, scheduler or tuning is added. Nonfinite probe
gradients fail. Undefined precision/F1 denominators and single-class AUC yield
NA; probability >=0.5 predicts M. Fixed probe stopping uses mean subject BCE.

Selected inner encoders are frozen in eval mode and used only in their own
coordinate system. Inner probes select duration only. A fresh outer probe is
trained on final-encoder development representations and tested on its own
outer test set. No cross-encoder latent pooling is performed.

K-means is on the development-averaged N*d_z matrix without scaling, not t-SNE.
K=13, K-means++, 50 starts, 500 iterations, tolerance 1e-4, seed 0. NMI/AMI use
arithmetic averaging. t-SNE: two dimensions, perplexity 30, PCA initialization,
automatic learning rate, seed 0, 1000 iterations (300 in smoke). It is visualization
only, with coordinates never aligned/averaged across folds.

## Topology and revised hub endpoint

Only topological analysis symmetrizes reconstruction. Original/reconstructed
graphs use the same global positive-edge rule independently. Inadequate positive
edges make the affected comparison undefined, with a recorded reason. Input
graph failure stops model training; it never silently excludes participants.

Global/local efficiency use zero contributions for disconnected pairs, divide
by all possible ordered pairs, and use binary graph distances. Local efficiency
uses the graph induced by neighbors, never paths through non-neighbors. Clustering
and local efficiency are zero for degree <2. Whole-graph means include isolated
nodes. Nodal efficiency averages full-graph inverse distances over N-1 nodes.

Largest-component ties use lexicographic sorted ROI-index lists; characteristic
path length is the mean over unordered connected pairs, equivalent to the
ordered-pair mean. It is NA if the largest component has fewer than two nodes.
The component size is always reported.

Leiden optimizes ordinary modularity, starting at singleton partitions, until
no further improvement (`n_iterations=-1`), for 20 purpose-seeded runs. The highest
observed Q wins. Exact Q ties use the lexicographically smallest canonical
membership vector (labels assigned by first ROI occurrence). Original and
reconstructed partitions are independent. Identical adjacency matrices reuse
deterministic cached graph results. Participation is zero for isolated nodes;
the no-edge graph has undefined modularity and hub identity. The seed derivation
uses the graph content to avoid order-dependent community optimization.

Six scalar graph metrics use absolute percentage error
100*abs(reconstructed-original)/abs(original). Separate signed percentages use
the same absolute denominator; separate absolute raw differences are also saved.
An original magnitude <=1e-8 gives undefined percentages, never a replacement
within that outcome's statistical test.

**Updated endpoint:** hubs have degree >= NumPy's linearly interpolated 90th
percentile, including ties. Hub counts can exceed 10% and are reported. A graph
with no edges has an undefined hub set. Compare aligned ROI sets: Jaccard
similarity, Jaccard distance (1-similarity), retention of original hubs, and
retained/lost/new/consistently nonhub status. Empty union/undefined sets produce
NA, not perfect agreement. Jaccard distance is the primary matrix-level
hub-preservation discrepancy; retention/status are descriptive companions.
This does not claim rich-club enrichment or compare only hub-to-hub density.

## Clinical inference and standardized scores

Architectural comparisons remain **descriptive**. The external clinical tests
are separate: two-sided Mann–Whitney, effect 2*U_patient/(n_patient*n_HC)-1.
For each outcome, only finite observations enter the test and original/valid
counts are reported. At least two per group are required. No ties and a group
size <=8: exact U distribution. Ties and a group size <10: seeded permutation U
test (9999 resamples, batch 256; 199 in smoke), exact enumeration if the number
of distinct partitions is smaller. Otherwise use asymptotic tie correction and
continuity correction. Identical pooled values yield p=1 and effect=0.

BH includes the full prespecified family size. Untestable outcomes are treated
as p=1 for the calculation but retain NA adjusted values and are never rejected.
Matrix family: five reconstruction metrics plus six scalar topological errors
and **hub Jaccard distance**, total 12, within each architecture. Region families:
379 tests within each of eight metrics, architecture and HC-vs-group comparison.
No study-wide FDR claim, extra hub-map test, or composite-score test is added.

HC reference statistics require >=2 controls and **all** controls finite for
that component. Sample SD <=1e-8 makes standardization undefined. Matrix
orientation is negative for Pearson/Spearman/symmetry and positive for errors.
Composite: half the mean of five reconstruction z-scores plus half the mean of
seven topology/hub z-scores; all 12 required. Composite is reported for patients
only. Tests use unstandardized outcomes, not HC scores standardized against
themselves.

Regional absolute z-scores are averaged over all eight metrics per patient;
then over all patients in each diagnostic group. Missing entries propagate
rather than changing the contributing patients/metrics. Network means/SDs
require every ROI score in that network. The whole-brain mean+sample-SD threshold
requires all ROI scores; otherwise exceeding fractions are undefined. Valid
counts remain available even when strict aggregate scores are undefined.

## Required manuscript edits (not applied to source)

1. Replace raw rich-club coefficient and its percentage error with ROI hub
   Jaccard distance; add retention and regional hub-status descriptions.
2. Replace 'seven topological percentage errors' with six percentage errors plus
   the dimensionless hub-overlap error, including the 12-outcome clinical family
   and the 50:50 composite definitions.
3. Explicitly retain descriptive architectural comparisons and remove the
   unresolved architectural hypothesis-test annotation.
4. Incorporate the numerical conventions above where needed for reproducibility.
5. Retain input provenance/ethics and figure update tasks. Do not portray code
   choices or synthetic checks as completed empirical experiments.

## Sources for implemented algorithms

- Original GAT: https://arxiv.org/abs/1710.10903
- GCN: https://arxiv.org/abs/1609.02907
- Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
- Leiden API: https://leidenalg.readthedocs.io/en/stable/reference.html
- SciPy U-test methods/ties: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
- scikit-learn nested CV: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

Software versions and CUDA availability are recorded per run. Deterministic
seeds do not imply bitwise equivalence across different hardware/library builds.
