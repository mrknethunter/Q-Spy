# E-03 implementation audit

**Audit date:** 2026-11-25
**Repository:** `/home/michio/Documents/Q_SPY/Q-Spy`
**Branch at preflight:** `E-03-task`
**Initial Git state:** clean; no staged or unstaged changes.

## Scope and sources reviewed

The audit reviewed the E-03 task card and the supplied implementation brief; the
Q-SPY proposal; the complete available dependency task cards for L-04, M-03,
M-04, M-06, D-04, L-02, I-03, I-04, Q-01, Q-02 and Q-03; the L-02 shared
contract; M-03/M-04 implementation notes; E-03 source, tests, configuration,
documentation and current output files; and the D-04 split implementation and
frozen manifest. Relevant passages of the local Glanzer et al. and Pranjic et
al. papers were checked for the physical and projected-kernel claims.

The official proposal defines the projected Gaussian kernel as
`exp(-gamma/n * sum_q ||B_iq-B_jq||^2)`, puts bandwidth scaling in the shared
embedding, and explicitly rejects a quantum-advantage claim. D-04 stores P2
unknown-class membership in each manifest entry; rotated folds contain
`Fast_Scattering` plus the selected family. The shared feature transform uses
the seven contractual Omicron columns, log10, training-only RobustScaler and
clipping. M-04 supplies the four requested novelty detectors and a common
budgeted search/scoring implementation. The M-06 task card describes seed
bootstrap summaries and paired tests, but this checkout contains no compatible
`stats.py` module.

## E-03-owned components

| Component | Responsibility and existing interface | Correct behavior | Findings before this change |
| --- | --- | --- | --- |
| `experiments/e03_novelty.py` | I-03 data loader, full-partition checks, deterministic subset selection, L0 embedding, four-model run, metrics and case outputs | Uses the frozen D-04 P2 manifest for unknown-family membership; checks full partitions before preprocessing; keeps test IDs fixed across seeds; sends identical outer IDs to every model; uses the shared feature transform, embedding and M-04 detector classes; applies the contracted projected-kernel normalization; calibrates a strict `>` threshold on known-only validation; writes seed-paired results and provenance. | **Inner-CV preprocessing leakage:** the outer-training scaler is fit before M-04's internal CV, so each inner validation fold contributes to its scaler statistics. **No event-level bootstrap:** only seed-resampling intervals are produced. Output audit records only the final outer scaler, not each inner-fold scaler. Source fingerprints omit some E-03 runner/config/report inputs. Selected test-family support and exact cross-model IDs are implicit in the run loop rather than all reasserted after selection. |
| `experiments/e03_parallel.py` | Independent held-out-family workers and aggregation | Uses spawn workers, per-family stage directories, checks child settings/provenance and refuses a nonempty destination; a worker failure leaves a failed manifest and partial staging data rather than declaring success. | Does not limit BLAS/OpenMP threads inside workers; aggregation does not check each child's output hashes or exact per-family seed coverage before reading it. It has no resume path; a controlled restart must use a new output directory. |
| `experiments/e03_report.py` | Recalculates saved metrics and regenerates the human-readable report | Recomputes AUROC/AP/attained-point TPR from float64 saved scores; checks selected IDs against frozen split membership and catalog labels; checks known-only threshold/budget and recomputes seed bootstrap tables. | Cannot validate the current output tree because the required prediction files and selected-ID archive are absent. It does not validate the complete source-fingerprint set or calculate event-bootstrap intervals. |
| `experiments/rq4/e03_gravityspy.yaml` | I-03 experiment declaration | Specifies P2, eight rotations, seeds 0–9, four workers, 8 candidates/model, 3 folds, 8-qubit/2-layer L0 embedding, `s=0.1778`, and explicit resource caps. | Correct as a resource-bounded configuration. It must not be described as an uncapped full-partition experiment. |
| `tests/test_e03.py` | E-03 protocol, metric, kernel, integration, report and parallel tests | Includes checks for partition exclusion, sampling, score orientation, ties, conservative thresholding, projected-kernel equivalence, batching, paired seed bootstrap, a synthetic integration run and parallel aggregation. | Does not test fold-specific scaler fitting or event-level bootstrap. No separate E-03 metrics/parallel suites exist. The logged earlier test receipt is historical and was not re-executed during this audit. |
| `docs/E03.md`, `docs/E03_results.md`, `docs/GUIDA_E03_IT.md`, `docs/E03_validation.json` | Method, result interpretation, reproduction instructions and prior validation receipt | Method documentation describes P2 scopes, score conventions, resource caps, no-hardware L0, and seed-conditional uncertainty. The physics note distinguishes nearest-feature neighbors from physical classification. | Documentation claims a successful independent verification and complete result set. In this checkout, `results/e03/manifest.json` lists eight prediction archives and `selected_ids.json.gz`, but none of these nine files is present. The prior receipt reports a pass from 2026-11-24; it is not evidence that the current on-disk tree is complete. The report also lacks the task-required finite-test-sample intervals. |
| `results/e03/**` | Existing E-03 outputs | Metric tables, plots, audit, summary and manifest are present as local artifacts. | The manifest says `status=complete` and hashes missing predictions/selected IDs. The result tables cannot be independently recalculated from the current checkout. These pre-existing outputs were inspected and left untouched. |

## Shared dependency findings (read-only)

- **D-04 / `data/splits.py`:** The frozen P2 bundle contains ten seeds and
  eight folds. Its manifest explicitly lists unknown classes per fold, and the
  `catalog.npz` contains IDs, labels, detector and run, but not raw features.
  No P2 split changes are needed or authorized.
- **D-02 data input:** the checked-out manifest names the expected Gravity Spy
  files, but neither `data/raw/` nor `data/curated/curated.csv` is present.
  Therefore the catalog alone cannot reproduce features or model scores.
- **M-04 / `models/baselines.py`:** RBF OC-SVM, Isolation Forest and LOF are
  available through `build_baseline`; LOF sets `novelty=True`; shared
  `score_samples` is oriented higher-is-more-anomalous. Its CV accepts one
  already transformed matrix and has no fold-local preprocessing hook. An
  E-03-local wrapper can preserve its candidate generation, estimator
  construction and known-only calibration objective while fitting the shared
  transform separately inside each fold.
- **M-03 / `models/novelty.py`:** provides the same higher-is-more-anomalous
  convention for the projected OCSVM contract; E-03 uses the M-04 projected
  representation with the shared RBF OCSVM estimator.
- **M-06:** the complete task card exists, but the promised shared statistics
  module is absent. E-03-local metric/bootstrap helpers are allowed by the
  task brief and are needed unless a compatible shared API is later supplied.
- **L-02, I-03/I-04, Q-01/Q-02/Q-03:** shared contracts and implementations
  are present for the seven-feature transformation, YAML runner, Bloch store,
  feature map, Pauli observables and exact L0 backend. No shared file needs to
  change. The installed `.venv` is Python 3.13 and lacks Qiskit and several
  pinned runtime dependencies, so it is not the documented project runtime.

## Scientific and data risks

The reported nearest-known-family counts are measured in transformed
seven-feature Euclidean space, not by a classifier that predicts a physical
family. The local Glanzer et al. paper supports the limited statements that
Fast Scattering is typically below about 60 Hz, Tomte power is mostly below
about 64 Hz, their O3 rates differ strongly by detector, and Tomte's physical
coupling remains unidentified. Those are context, not a causal explanation of
this model's errors. A local copy of the additional Soni paper cited by the
existing E-03 report was not found; revised physics text should rely on the
available primary source unless that paper is separately supplied.

The existing experiment is capped at 2,048 train, 2,048 validation and 8,192
test observations per case. Its ten-seed intervals condition on the fixed
selected test IDs and do not quantify event-sample uncertainty. H1/L1 rows are
not clustered by coincident physical-event identity in the D-04 contract.

## Required E-03 corrections

1. Add an E-03-local estimator adapter that fits `make_features()` on each
   inner training fold, reuses that scaler only for its fold's validation data,
   and fits the final scaler on the full selected outer training partition.
   The quantum representation must be embedded after that fold-local transform.
2. Preserve M-04 candidate selection and known-only scoring, equal candidate
   counts, and identical outer IDs across all four models.
3. Add paired, event-level bootstrap intervals from saved scores, separate from
   seed-variability intervals, with the resampling unit and estimand documented.
4. Strengthen E-03 tests for fold-local scaler membership, unknown-family
   membership after subset selection, event-bootstrap pairing and provenance.
5. Record all E-03 execution source/config fingerprints and verify worker output
   hashes before aggregation. Limit numerical-library threads in workers.
6. Correct E-03 documentation to distinguish prior reported metrics from the
   incomplete artifact tree currently available. Do not overwrite these
   pre-existing result files or claim full real-data validation without the
   curated input and missing outputs.

## Blocking conditions for real-data completion

The main experiment cannot be rerun from this checkout: the D-02 raw or curated
Gravity Spy input is absent, the available split catalog has no features, and
the active local Python environment lacks the Qiskit exact-simulation stack.
The current saved outputs also lack the prediction and selected-ID artifacts
needed by the independent verifier. These conditions block real-data rerun and
acceptance validation; they do not prevent implementing and exercising the
E-03-local logic on synthetic inputs when the required runtime is available.

## Post-audit implementation and current checks

Implemented within E-03-owned paths:

- Added `experiments/e03_preprocessing.py` to wrap the shared M-04 estimators.
  It binds the same deterministic KFold training-ID membership used by M-04,
  fits the shared transform inside each training fold, and fits the final
  transform only on the full selected outer training IDs. The projected model
  embeds each fold-specific transformed representation at exact L0.
- Added `experiments/e03_metrics.py` with the existing score convention and
  metric definitions, plus a stratified paired trigger bootstrap. Per-seed
  event intervals and paired differences are distinct artifacts from the
  seed-resampling summaries.
- Updated the runner and parallel aggregator to check selected test support,
  exact model partition-ID identity, scaler membership, validation scores,
  worker output hashes, source/config provenance and family/seed coverage.
  Numerical worker threads are limited to one.
- Updated the independent verifier to check source and artifact hashes, full
  partition exclusion, selected IDs and labels, validation thresholds, fold
  and outer scaler membership, metric values, and both seed and event interval
  tables. The report links to the per-case event interval artifacts.
- Updated the E-03 method/Italian handoff docs. The prior numerical report is
  explicitly historical. The old validation receipt is retained as a nested
  historical record in `E03_validation.json`; the current receipt records
  checks actually run in this checkout.

Executed checks on 2026-11-25:

- `tests/test_e03_metrics.py` and `tests/test_e03_preprocessing.py`: **8
  passed**.
- Python AST parsing for all eight changed E-03 Python files: **passed**.
- `tests/test_e03.py`: collection **blocked** by
  `ModuleNotFoundError: No module named 'qiskit'` in the active Python 3.13
  environment. Dependencies were not installed.
- Full L0 Gravity Spy run and independent validation: **not run**. The required
  curated D-02 CSV and Qiskit exact-simulation runtime are unavailable. No
  result placeholders were created and the prior `results/e03/**` outputs were
  not altered.

Accordingly, this checkout is **implemented — not validated**. Synthetic helper
tests establish only the tested adapter/bootstrap behavior; they are not
evidence of real-data performance, all-model L0 integration, or scientific
acceptance.
