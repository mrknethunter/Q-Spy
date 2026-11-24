# Q-Spy interface contracts (L-02)

These are the only signatures modules may rely on across task boundaries.
Changing one requires the lead's approval and an update of this file and of `tests/contracts/`.
Stubs for not-yet-implemented modules raise `NotImplementedError`; their contract tests are skipped
until the stub is implemented, then they are enforced automatically.

Synthetic data that satisfies every contract lives in `tests/fixtures/synthetic.py`:
develop and test against it without waiting for other tasks.

## Shared constants

| Name | Value |
|---|---|
| Known classes | `Blip, Extremely_Loud, Koi_Fish, Low_Frequency_Burst, Low_Frequency_Lines, Scattered_Light, Tomte` (D-02 `KNOWN_CLASSES`) |
| Held-out class | `Fast_Scattering` (present in O3a/O3b only) |
| Feature columns | `snr, peak_frequency, central_freq, bandwidth, duration, q_value, amplitude` |

Column names are the Gravity Spy names used by D-01/D-02 (`central_freq`, not `central_frequency`).
The order above is the order of the columns of `X_tilde`, and therefore the feature-to-qubit assignment
(qubit `q` receives feature `q mod 7`).

## Data (D-02, D-03, D-04)

```python
data.curate.load_curated() -> pd.DataFrame
```
Columns: `id: str` (unique), `gps: float`, `ifo: "H1"|"L1"`, `label: str`, `confidence: float`,
the 7 feature columns (raw, physical units), `run: "O1"|"O2"|"O3a"|"O3b"`, `is_held_out: bool`.

```python
features.transforms.make_features(df, scaler=None) -> tuple[np.ndarray, scaler]
```
Returns `X_tilde` of shape `(len(df), 7)`, `float64`, every value in `[-1, 1]`, columns in the order above.
`scaler=None` fits on `df`: callers pass training rows only (or the unlabelled pool, never test rows).

```python
data.splits.load_split(protocol, seed, k=None, held_out=None) -> Split
Split(train_ids: tuple[str, ...], val_ids: tuple[str, ...], test_ids: tuple[str, ...])
```
`protocol` is `"P1"` (few-shot, `k` in {5, 10, 20, 50}), `"P2"` (leave-one-class-out, `held_out` set)
or `"P3"` (O3a -> O3b). The three id sets are disjoint. In P2 the held-out class appears in `test_ids` only.

## Quantum embedding (Q-01..Q-05)

```python
quantum.embed.EmbedConfig(n_qubits: int, layers: int, s: float, entangle: bool = True)
quantum.embed.embed(X_tilde, cfg, backend="L0", *, precision=0.03, seed=None, submit=False, device="ibm_kingston") -> EmbedResult
quantum.embed.estimate_usage(n_circuits, shots, n_subjobs=1) -> float   # QPU seconds, 30% margin included
quantum.embed.EmbedResult(B: np.ndarray, meta: dict)
```
- Angles are `theta = s * pi * X_tilde`; the bandwidth `s` is applied here, nowhere else.
- `backend` is the tier: `"L0"` exact, `"L1"` shots, `"L2"` fake Heron, `"HW"` IBM hardware.
- `B` has shape `(N, n_qubits, 3)` with `B[i, q] = (<X_q>, <Y_q>, <Z_q>)`, `float32` or `float64`,
  `||B[i, q]|| <= 1` up to shot noise.
- `meta` contains at least `tier` and `embed_config` (a dict of `EmbedConfig`); a submitted `HW` run adds `job_ids`, `usage_s` and
  `calibration` (last update, T1/T2/readout error of the used qubits, two-qubit gate errors between them).
- `"HW"` defaults to a dry run (no network, `B is None`; `meta` has `n_circuits`, `shots`, `executions`, `estimated_qpu_s`);
  it submits only with `submit=True`.
- `"L2"` and `"HW"` add `backend_name`, `layout` (physical qubits) and `two_qubit_depth` to `meta`; `precision` is the target standard error per expectation value.

## Bloch-vector store (I-04)

```python
store.io.write_bloch(run_id: str, ids, B, meta, root=None) -> pathlib.Path
store.io.read_bloch(run_id: str, root=None) -> tuple[list[str], np.ndarray, dict]
```
Round trip is exact (ids order, values, meta). Writing an existing `run_id` whose `meta["tier"] == "HW"`
raises `FileExistsError`. `root=None` means `store/runs/`. G-03 `compare_runs` consumes this tuple.
On disk a run is `store/runs/<run_id>/` with `B.npy`, `ids.json` and `meta.json`; `write_bloch` adds `created_at` and
`git_commit` to `meta`. Runs whose id starts with `hw-` are committed to git, all others are ignored.

## Kernels (M-01)

```python
kernels.projected.projected_kernel(B1, B2, gamma) -> np.ndarray   # (N1, N2)
kernels.classical.rbf_kernel(X1, X2, gamma) -> np.ndarray          # (N1, N2)
```
`projected_kernel` = `exp(-gamma / n * sum_q ||B1[i,q] - B2[j,q]||^2)`, i.e. an RBF kernel on
`B.reshape(N, -1)` with `gamma / n`. The factor 1/2 relating Bloch and Frobenius distances is absorbed in `gamma`.

## Models (M-02..M-04)

Every model follows `models.baselines.BaseBudgetedModel` (M-04):

```python
model = SomeModel(budget=..., seed=...)
model.fit(Z_train, y_train)          # y=None for novelty models
model.predict(Z)                     # labels (few-shot) or inlier/outlier flags (novelty)
model.score_samples(Z)               # few-shot: class scores; novelty: HIGHER = MORE ANOMALOUS
model.get_budget_report()            # must contain n_configs_evaluated <= budget_limit
```
`Z` is the representation, not the kernel: `X_tilde` for classical rungs, `B.reshape(N, -1)` for projected rungs.
Because the projected kernel is an RBF kernel on flattened Bloch vectors, the same model class serves both, which
makes the comparison fair by construction. Novelty scores use the opposite sign of scikit-learn's `score_samples`:
negate it when wrapping a scikit-learn detector. `tests/contracts/model_contract.py` checks all of this.

Representation names used in results: `R0` linear on `X_tilde`, `R1` RBF/Laplacian on `X_tilde`,
`R2` projected without entanglement, `R3` projected with entanglement, `R4` linear on Bloch vectors, `R5` fidelity kernel.

## Results table (M-02, M-03 write; M-06, figures read)

`results/<experiment>.parquet`, one row per metric value (long format):

| Column | Type | Example |
|---|---|---|
| `experiment` | str | `E-02` |
| `protocol`, `seed`, `k`, `held_out` | str, int, int or null, str or null | `P1`, `3`, `5`, null |
| `representation`, `model`, `tuning` | str | `R3`, `RBFSVMBaseline`, `loo` |
| `embed_run_id` | str or null | store run used, null for classical |
| `budget`, `n_configs_evaluated` | int | `60`, `60` |
| `best_params` | str (JSON) | `{"C": 1.0, "gamma": 0.5}` |
| `metric`, `value` | str, float | `macro_f1`, `0.83` |
