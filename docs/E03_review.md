# E-03 — Methodological correction after PR #67 review

The earlier E-03 outputs used known-only false-alarm calibration to select
OC-SVM gamma. That objective did not measure anomaly separation. **Those
outputs are superseded and must not support H3 or a representation comparison.**
This revision changes shared `models/baselines.py`, not a private E-03 detector.
`docs/decisions.md` is preserved; this document records implementation choices
and deviations explicitly rather than rewriting pre-registration retrospectively.

## Nested known-family selection

Each outer frozen P2 case excludes Fast_Scattering and its rotated unknown
family from training, validation, preprocessing and selection. Within the
selected outer training IDs, each remaining known family becomes a surrogate
anomaly in turn. Rows of every other family are deterministically split 80/20
(rounded up for calibration, minimum one) into inner fit and known calibration.
The surrogate family and calibration rows **never fit the inner RobustScaler**.
The seven-feature transform and L0 embedding are then recomputed with that
inner scaler. These exact inner splits/preprocessing are shared across models.
The outer validation/test rows do not enter gamma selection. The legacy
`cv_folds` constructor/config field is retained for interface compatibility;
novelty LOCO uses one fold per known family, not that K-fold count.

For each representation and each inner fit:

1. Compute the median of all off-diagonal squared Euclidean training distances
   (unique unordered pairs; diagonal excluded). Reject a nonpositive median.
2. Evaluate all nine `c = logspace(-2, 2, 9)`; gamma is `c / median`.
3. For that gamma, choose nu from `[0.001, 0.005, 0.01, 0.05, 0.1, 0.2]`
   by smallest absolute difference between native known-calibration FPR and
   **0.05**, ties favouring smaller nu. This choice never uses surrogate scores.
4. Evaluate AUROC on the known calibration rows (negative) and the entire
   surrogate family (positive). Select c by the mean AUROC across all surrogate
   families, with equal family weights. Exact ties favour smaller c/gamma.
5. Recompute the distance median on the full selected outer training
   representation and transfer the selected dimensionless c to its final gamma.
   Fit each candidate nu on this training set and calibrate it against the
   external **known-only outer validation**. No final refit includes validation.

The labelled OC-SVM fitting path constructs this data-dependent grid directly;
the legacy static `_param_grid` does not participate in labelled gamma selection.

**Material calibration deviation:** none of the 160 final OC-SVM fits reaches
5% native validation FPR in the production nu grid. Selected native FPR ranges
are **16.65–67.14% projected** and **10.40–65.82% RBF**. Therefore this is not
reported as successful nu-only 5% calibration. The operating point uses an
explicit independently calibrated intercept/score threshold instead.

A validation-only feasibility diagnostic on the primary RBF seed-0 case
extends nu from 1e-5 to 0.9 and tightens solver tolerance from 0.001 to 1e-6.
The lowest observed native FPR remains 32.18%. This single-case check does not
prove global impossibility; it shows that simply extending this grid or solver
precision did not repair that case. No test rows enter this diagnostic and
none of its fits change the production scores or selected parameters.
See `experiments/e03_nu_check.py` and `docs/E03_nu_check.json`.

Reason for the deviation: nu controls a training-set support/outlier trade-off;
it is not a validation FPR parameter, and its zero-score decision boundary can
reject many unseen known examples at separation-optimal gamma. Calibrating the
score intercept on known validation retains surrogate-AUROC selection and an
honest 5% operating point without using true anomalies. This extension must be
reviewed as a deviation for primary H3, not portrayed as exact pre-registration
compliance or a formal confirmation of H3.

The finite nu grid leaves a substantial native FPR residual in this run. Both the grid and residual
are audited; there is no claim that nu itself equals a false-alarm probability.
A separate strict `score > quantile(scores_known, .95, method='higher')`
operating threshold ensures empirical validation FPR at most 5%, including ties.
The raw continuous anomaly scores determine AUROC/AP; shifting a threshold does
not change their ranking. No score sign is chosen using test AUROC.

Isolation Forest and LOF use the same shared surrogate-AUROC objective and
independently fitted inner feature transformations. Their contamination is
fixed to 0.05: varying this offset creates identical rankings, not new anomaly
representations. Their final 5% operating thresholds are known-validation based.

## Explicit deviations and limitations

- Section 4 specifies equal searches of 60 configurations. This revision uses a
  common **ceiling** of 60, **not exactly 60 distinct configurations per model**:
  OC-SVM evaluates all 9×6 = **54** gamma/nu pairs (plus six final calibration
  fits), Isolation Forest selects **60** distinct configurations from its
  nonredundant grid, and LOF exhausts **15** neighbor/metric configurations.
  Padding to 60 by repeating offset-equivalent fits would not add search
  information. Counts, fold AUROC, calibration FPR and all candidates are saved.
  This is a disclosed deviation requiring reviewer assessment; the identical
  complete OC-SVM search for R1/R3 addresses H3's representation comparison.
- The nu grid, 80/20 inner calibration allocation, median convention and
  equal-family AUROC aggregation above resolve details not specified in §4.
- Budgets below 54 are explicitly exploratory/synthetic smoke configurations:
  fixed nu=0.05, a deterministic subset of the nine c values. They cannot be
  represented as the preregistered full run. The delivered real-data YAML uses 60.
- `fit(X)` remains available to existing feature-only callers, but is now an
  **untuned fixed heuristic** with null CV score and explicit audit mode.
  Family labels are required for LOCO. For custom pipelines use
  `fit(X, family_labels, validation=(X_val, labels_val), inner_representations=...)`.
  Precomputed inner representations must come from the shared splitter and
  exclude the surrogate/calibration rows when fitting preprocessing. If external
  validation is omitted, final calibration reserves known training rows and
  the final estimator does not refit on those rows; production E-03 supplies
  external validation. Feature-only/pretransformed callers cannot retroactively
  undo a scaler fitted outside their inner fold.
- The existing 2048/2048/8192 caps and fixed-test sampling remain explicit
  resource limits. The original capped experiment is not upgraded to a claim
  about fitting every curated row.
- H3 is primary, not exploratory. Its final inference must be reconciled with
  the planned seed/rotation pairing and Holm family H1–H4. Conditional bootstrap
  intervals alone neither validate that familywise inference nor establish a
  general quantum advantage. Previous test outcomes have already been observed;
  this correction is reported transparently as a post-review reanalysis.

## The 1% diagnostic is not the 5% tuning target

E-03 retains its separate 99th-percentile known-validation threshold for
`family_errors.csv`, Fast_Scattering missed-event neighbours and the two
`*_validation_threshold` metrics. This is a 1% diagnostic operating point.
`TPR@1%FPR` is separately the largest attained test ROC TPR at FPR≤1%, without
interpolating an unattained low-FPR point. The model's own 5% selection/calibration
threshold is saved in `audit.json` and does not use test labels.

## E-04: hardware scaler compatibility

E-03 fits one scaler on each selected outer training fold, and an additional
scaler on each inner fit subset. Hardware campaign features instead use the
scaler fitted on the **full frozen P2 seed-0 training IDs**, as implemented in
`scripts/qpu_campaign.py:gravityspy_features`. These scalers are not identical,
even for E-03 seed 0 because E-03 caps its training set at 2048.

For E-04, transform every matched classical, ideal L0 and hardware point with
that same saved/reproduced hardware training scaler, with identical feature
order, clipping and embedding configuration. Recompute ideal/classical kernels
and any associated train/validation representations under that scaler. Do not
compare existing fold-specific E-03 Bloch vectors directly with hardware
vectors or interpret their difference as device noise. Record the scaler
training-ID hash, center, scale and point IDs for both sides.

## M-03 follow-up status

Kernel density, kernel-space kNN and kernel-PCA reconstruction detectors are
not implemented by this correction. `docs/M03_followup.md` contains a concrete
issue body and acceptance criteria. Automatic creation was attempted but the
GitHub connector returned 404 for this private repository: **the follow-up
issue is not yet open**. Create it and reference its number in PR #67 to finish
this review item; do not mark it resolved merely because this draft exists.

## Warning correction

`experiments/e03_parallel.py` concatenates only populated family diagnostic
frames. If every frame is empty, it preserves the explicit empty schema. This
removes the pandas empty-column FutureWarning without suppressing warnings.
