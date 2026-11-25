> **Esito successivo della review:** 80/80 casi completati e verificati, 3.200 metriche ricalcolate. La validazione tecnica corrente è descritta in [E03_results_loco.md](E03_results_loco.md) e `E03_validation.json`; l’accettazione formale H3 resta distinta per il budget dichiarato di nove configurazioni.

> **Revisione successiva autorizzata:** la review fornita dall’utente richiede la correzione condivisa del tuning. L’ambiente pinned è ora disponibile e i test scientifici sono stati eseguiti. Questo rapporto conserva la fase precedente; lo stato aggiornato è in [E03_REVIEW_RESPONSE.md](E03_REVIEW_RESPONSE.md) e `E03_validation.json`.

# E-03 delivery review — 2026-11-25

**Status: IMPLEMENTED — NOT VALIDATED.** The repaired code has not been
executed with sklearn/Aer. No new Gravity Spy comparison, confidence interval
or scientific conclusion is claimed.

## A. Initial audit

The ZIP contained repository `Q_SPY/Q-Spy`, branch `E-03-task`, seven modified
tracked E-03 files and five untracked E-03 files. All were preserved as input
work; `results/e03/review_20261125/initial_state.json` records their hashes and
Git state. The user subsequently authorized a new branch, `E-03-task-review`.

Reviewed every E-03 source/test/config/document/output file, complete original
task cards E-03/L-04/M-04/M-06/D-04/M-03/L-02/I-03/I-04/Q-01/Q-02/Q-03,
the proposal, contracts, actual shared implementations and local primary
papers relevant to kernel and physics interpretation. See
[the implementation audit](E03_IMPLEMENTATION_AUDIT.md) for component-level
responsibilities, dependencies, correct behavior, defects and corrections.

Two E-03 runner files were truncated and syntactically invalid. The existing
fold-local preprocessing/event-bootstrap work was present but not executable.
The prior results lack nine manifest-listed artifacts, and their historical
source hashes differ from this checkout. No raw/curated feature input exists.
M-06's shared statistics module is absent; I-03 lacks its proposed MLflow
integration. These shared components were inspected and left unchanged.

## B. Implementation

Created in this review:

- `experiments/e03_integrity.py`: E-03-only streaming hashes, complete artifact
  and case checks, restart identity, atomic manifests and conservative threshold.
- `tests/test_e03_integrity.py`: 18 executable stdlib tests.
- `tests/test_e03_review.py`: five scientific regression functions covering
  restart/failure orchestration, threshold recalculation, preprocessing guard,
  event pairing order and consistent bootstrap FPR limits.
- This report and evidence under `results/e03/review_20261125/`.

Modified existing E-03-owned sources only:

- `e03_novelty.py`: completed output/failure handling; retained fold-local
  preprocessing; moved supported Bloch-store writes under E-03 outputs;
  added explicit known/unknown counts, known validation score outputs and
  per-event missed-Fast IDs/neighbors; fingerprinted all execution sources.
- `e03_parallel.py`: completed source, preserved thread limiting, added strict
  worker checks and controlled resume with data/split/source/software/settings
  identity. Complete matching workers are reused; partial attempts are retained;
  corrupt complete workers and changed requests are rejected.
- `e03_preprocessing.py`: reject unregistered scaler fitting before the transform.
- `e03_metrics.py`: correct nondefault FPR point estimates and explicit pairing
  reference independent of dictionary order.
- `e03_report.py`: complete source/output coverage; duplicate detection;
  threshold/FPR recalculation from saved known validation scores; scaler,
  budget, common-ID and support checks; dynamic detector/run diagnostics,
  candidate-family neighbor counts and locally supported physics context.

Maintained `E03.md`, `E03_IMPLEMENTATION_AUDIT.md`, `E03_results.md` and
`E03_validation.json`. Previous results and the prior receipt are preserved.
The configuration and original tests retain their pre-existing modifications;
they were not rewritten by this review.

The local adapter preserves shared M-04 candidate generation and scoring. No
generic/shared module, dependency declaration, build or CI file was changed.
The extra integrity module isolates checks that can execute without Qiskit.

## C. Scientific methodology

- P2: frozen D-04 membership, Fast_Scattering always unknown; rotated family
  also unknown. Train/validation are known-only. Evaluate `target_family` and
  `protocol_unknown` separately, using the manifest rather than `is_held_out`.
- Same selected outer IDs and deterministic seeds for all four detectors.
  Defaults remain eight rotations, seeds 0–9, eight candidates, three inner folds,
  and resource caps 2048/2048/8192. This is not an uncapped experiment.
- Shared seven-feature log10/median-IQR/clipping pipeline. Each inner scaler
  uses its own training fold; final scaler uses the selected outer training
  partition. Quantum embedding follows the corresponding scaler.
- Shared exact L0 embedding. The annotated `hw-freeze` tag and current YAML
  agree on 8 qubits, 2 layers, `s=0.1778`. Bandwidth is applied once by embed.
  RBF on `B_flat/sqrt(n)` implements the contractual projected kernel.
- M-04 owns models, candidate generation and known-only calibration scoring;
  no held-out test label or score changes sign, hyperparameters or thresholds.
  LOF retains `novelty=True`. Equal candidate count is not equal compute cost.
- Threshold: higher 99th percentile of independent known validation scores,
  decision `score > threshold`; explicit empirical FPR bound includes ties.
- AUROC, Average Precision and greatest attained TPR with test FPR <= .01;
  validation-calibrated TPR/FPR remain separate from descriptive ROC statistics.
- Separate seed-paired percentile bootstrap and stratified event-paired
  percentile bootstrap. Events are trigger rows; class counts/prevalence are
  fixed in event resampling. Rotations overlap and are not independent datasets.
  One seed has no estimated across-seed CI.
- Physics text distinguishes measured feature proximity, literature and causal
  hypotheses. Glanzer's local primary paper supports the limited scattering,
  frequency and detector context; no unsupported Soni citation or synthetic
  detector count is introduced. New measurements still require a real rerun.

## D. Testing

| Suite/check | Written | Executed | Passed | Failed | Skipped | Evidence/limit |
| --- | --- | --- | --- | --- | --- | --- |
| New stdlib integrity suite | 18 test methods | 18 | 18 | 0 | 0 | Bundled Python 3.12; includes E-03 source/test AST parsing, corruption/omission/duplicates, restart identity, atomic output and threshold edge cases. |
| Existing E-03 scientific tests plus five new regression functions | Present | 0 | 0 | 0 | 0 | Pytest invocation exits 1 **before collection**: no pytest. These are unexecuted, not passing or skipped tests. |
| Frozen D-04 full partition check | Audit script | 80 cases | 80 | 0 | 0 | NumPy 2.3.5/pandas 3.0.1 in bundled Python 3.12; membership only, outside the project's pinned environment. |
| Full E-03 model/quantum integration and experiment | Existing integration plus new regressions | 0 | 0 | 0 | 0 | Pinned Windows runtime and real features unavailable. |

The bundled runtime lacks sklearn/Qiskit/Aer/pytest. The archived Linux venv is
not Windows compatible, differs from pinned NumPy/pandas/sklearn and lacks
Qiskit. No packages were installed or dependency files edited. Historical
249-test/14-test receipts are not current evidence. Logs, exact commands and
versions are in `results/e03/review_20261125/test_execution.json`.

## E. Experimental results

No replacement Fast_Scattering or rotation results were generated. Nine
existing manifest-listed files match their hashes; eight predictions and
`selected_ids.json.gz` are missing. Prior AUROC/AP/TPR tables remain historical,
qualified as such in `E03_results.md`; they are not acceptance evidence for the
corrected code. No synthetic output is presented as Gravity Spy evidence.

## F. Validation

Read-only verification of all 80 FULL frozen P2 cases succeeded: ID uniqueness,
partition disjointness, known-only train/validation, unknown membership from the
manifest, all unknown families represented in test, both populations in test,
and identical test IDs across seeds within each rotation. Catalog: 318,801 rows.

Equal-budget, fold-scaler and kernel/model runtime assertions are implemented
but have not been exercised here. Independent saved-score metric/threshold/CI
recalculation cannot run on the incomplete historical artifact set. There is no
new complete experiment manifest. The current review receipt records partial
verification and blockers rather than asserting full validation.

## G. Remaining issues

1. Provide the existing project-pinned runtime and read-only D-02 curated
   features. No download to protected shared data paths was authorized/executed.
2. Execute all scientific suites, including real Aer batching/kernel checks,
   fold-specific preprocessing, complete synthetic integration and resume tests.
3. Run all 80 configured real-data cases into a fresh E-03 output directory,
   then independently recalculate metrics, thresholds, budgets, hashes and CIs.
4. Review the resulting physics diagnostics, detector/run imbalance, sample
   caps and descriptive uncertainty with the project lead. No quantum advantage
   or statistical superiority is claimed.
5. M-06 and full I-03/MLflow work remain with their owners. No request to change
   shared interfaces is made. The task's explicit M-04 procedure takes precedence
   over a different procedure in the draft decision document.

## H. Git and modification audit

Initial branch: `E-03-task`; final branch: `E-03-task-review`, authorized by the
user's later instruction. HEAD remains `b9800bd566c83cb451c725f518cae3edd89071cf`.
Initial/staged/changed file inventories and final hash comparisons are recorded
under `results/e03/review_20261125/`; see `initial_state.json` and
`final_safety_audit.json`. The latter distinguishes edits from this review
from modifications already contained in the ZIP.

All implementation/documentation edits belong exclusively to E-03. Shared
components, prior result artifacts and external project documentation are
checked against their initial/archive bytes. No commit, staging, push, merge,
reset, stash or Git cleanup occurred. The sole Git mutation was the explicitly
authorized creation/switch to the new task branch.

## I. Final status

**IMPLEMENTED — NOT VALIDATED.** Passing integrity checks and frozen partition
checks do not constitute model validation or scientific acceptance.
