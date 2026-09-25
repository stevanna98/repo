# Identity-bearing data format

Supply these five files in the configured `data_dir`:

1. `reference.npz`: arrays `fc` [S,379,379], `subject_ids` [S] Unicode strings,
   and `roi_ids` [379] Unicode strings.
2. `external.npz`: the same fields, with the identical ROI order.
3. `participants.csv`: `subject_id,cohort,sex,group`. Cohort must be `reference`
   or `external`; sex `F` or `M`; reference group `HC`; external group `HC`, `BD`,
   `MDD` or `PATIENT`. Use `PATIENT` when patient status is known but diagnostic
   subtype is unavailable. These participants enter pooled patient-versus-HC
   analyses, but not BD/MDD subgroup analyses; subgroup tests with insufficient
   observations are reported as untestable. Optional extra columns are retained
   in the source but not modeled.
4. `rois.csv`: `index,roi_id,label,network`. Indices must be 0..378 in matrix-axis
   order. ROI IDs must be unique. The provided reference mapping must contain
   13 groups; no real atlas mapping is invented by this implementation.
5. `provenance.json`: at least `{"synthetic": false}`, plus acquisition/export,
   participant-selection, ROI mapping and QC provenance supplied by the researcher.

Each cohort's participant rows, filtered from `participants.csv` without sorting,
must already match that NPZ's `subject_ids`. Both matrix axes match `roi_ids`.
IDs are explicit, not inferred from filenames or numerical position. Reference
and external IDs cannot overlap. One matrix per subject is currently supported;
combine repeated runs upstream according to your documented protocol rather
than treating them as independent subjects.

The loader rejects duplicates, missing fields, unknown labels, wrong order,
nonfinite matrices, asymmetry, non-unit diagonals, out-of-range correlations,
and negative eigenvalues exceeding tolerance. It does not drop, impute, reorder,
symmetrize, clip or otherwise repair real data. Accepted inputs are converted to
float32 for PyTorch; analysis metrics use float64. Validation tolerances are
recorded in the configuration. Graph construction separately verifies enough
positive edges at each density; it never fills with negative/artificial edges.

Synthetic generation uses the same schema, labels every identity as `SYN_...`,
and writes a provenance record. Its 12 groups of 30 regions plus one group of
19 are *simulated assignments*, not anatomical atlas annotations. Its balanced
random sex labels have no designed association with connectivity. Diagnostic
perturbations modify network loadings in a factor model and are not clinical
models of bipolar disorder or depression.

