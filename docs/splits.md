# D-04 — Frozen protocol splits

`data/splits.py` implements the interface supplied with L-02:

```python
load_split(protocol: str, seed: int, k: int | None, held_out: str | None) -> Split
# Split(train_ids, val_ids, test_ids)
```

The return fields are immutable, sorted tuples of event-ID strings, not row
numbers in the caller's DataFrame. Optional defaults for `k` and `held_out` and
a keyword-only `split_dir` are added without changing four-argument calls.
The actual shared L-02 implementation/tests were not supplied; its `Split`
type and any additional requirements must be checked when integrating.

## Fixed inputs and source binding

The distributed bundle is generated from the default D-02 curation:
**318,801 rows**, seven known labels, plus reserved `Fast_Scattering`.
The D-02 source release, confidence threshold, coincident-trigger policy and
provisional seven-feature vector are documented in the D-02 data card.

| Input property | Frozen value |
| --- | --- |
| Dataset SHA-256 | `c8bb3d84efa5b583ee41d671ae195b73bb5cdf901e3af03e351e1e3eac0260f1` |
| Experiment seeds | Integers 0 through 9 |
| Few-shot training sizes | k = 5, 10, 20, 50 per known class |
| Fixed test-pool seed | 2026 |
| Version directory | `data/splits/v1/` |
| Known labels | Blip, Extremely_Loud, Koi_Fish, Low_Frequency_Burst, Low_Frequency_Lines, Scattered_Light, Tomte |
| Reserved novelty label | Fast_Scattering |

Rows are canonicalized by ID. The dataset digest includes all IDs, detector,
run, target labels, GPS, confidence and the seven raw feature values. Float
values are hashed as float64 binary values. Changes to input features invalidate
the binding even when all IDs remain the same. Extra DataFrame columns are not
used. The D-02 `is_held_out` column, when present, must agree with the reserved label.

`verify_dataset(df)` checks this binding before a model uses the partitions.
Read a persisted D-02 CSV with `float_precision='round_trip'`; ordinary decimal
parsing may change the last bit of some floats and legitimately fail verification.

## Protocol definitions

The issue specifies the main protocol requirements but not validation budgets,
fractional allocations or every held-out interaction. The following choices are
explicit implementation defaults for review; they are recorded in the manifest.
Changing a choice requires a new version directory.

| Protocol | Train | Validation | Test | Unused |
| --- | --- | --- | --- | --- |
| P1 | Exactly k rows per known class | 20 rows per known class | One fixed pool of 100 rows per known class | Other rows, including all Fast_Scattering |
| P2 | About 60% of each remaining known class/detector stratum | About 20% of each remaining known class/detector stratum; known families only | About 20% normal examples plus every held-out example | None |
| P3 | About 80% of known O3a rows within class/detector strata | Remaining about 20% of known O3a rows | Every O3b row, including reserved novelty examples | O1/O2 and O3a Fast_Scattering |

### P1: few-shot

The fixed test pool is selected once, before experimental seeds or k are used,
and is disjoint from every P1 training/validation set. It is balanced by known
class and approximately proportional to detector frequency within each class,
with at least one example from each detector.

For each of ten seeds, a class-balanced validation pool of 20 examples per class
is selected from the non-test rows. The validation pool stays the same for all
four k values for that seed. Training is selected from the remaining rows,
without replacement and stratified by detector within class. Training examples
are nested across k for this two-detector allocation: the k=5 set is a subset
of k=10, which is a subset of k=20 and k=50.

The k budget counts **training examples only**. Validation uses a separately
declared budget of 140 labelled examples in this seven-class dataset; this must
be considered when interpreting few-shot comparisons. `Fast_Scattering` does
not occur in any P1 partition. Changing the seed changes training/validation
selection but cannot change the fixed test pool.

### P2: leave one known family out, with reserved Fast_Scattering

Default `held_out=None` means the `Fast_Scattering` fold: all seven known classes
supply normal train/validation/test rows, and all Fast_Scattering rows are novel
test examples.

For each rotation, `held_out='Blip'`, for example, additionally excludes that
known family from training and validation. Test then contains all Blip and all
Fast_Scattering examples, plus normal test examples from the other six families.
Thus Fast_Scattering is **always reserved**, consistent with D-02; a rotated
fold has two novel labels. This is the stated interpretation of the issue's
“Fast_Scattering held out + rotation of other held-out classes”, and is not the
alternative strict single-unknown protocol that would train on Fast_Scattering
when another class is rotated out. The group should review this distinction.

The known-family test subset is fixed independently of the experimental seed
within each `(label, ifo)` stratum. Validation is then selected by seed from the
remaining known-family rows. Both validation and test receive
`max(1, floor(stratum_size * 0.2))` rows; training receives the remainder.
Strata too small to support all three partitions fail explicitly.

Test includes known normal examples so that novelty metrics requiring both
normal and novel targets can be evaluated. The split API returns IDs; novelty
targets must be derived from the manifest's `unknown_classes` for that fold.
Neither training nor validation ever contains a held-out label.

Eight folds (Fast_Scattering and each of seven known labels), each with ten
seeds, produce 80 P2 splits. All retained rows are assigned exactly once in
every P2 split. Test is unchanged across the ten seeds within a fold.

### P3: O3a to O3b

For each known `(label, ifo)` stratum, validation receives
`max(1, floor(O3a_stratum_size * 0.2))` examples selected by seed. Remaining O3a
examples train the model. There must be at least one training and one validation
example per stratum. No O3b row participates in training, validation or fitting
of preprocessing.

All O3b examples are in test, so test is fixed across seeds. O3a
Fast_Scattering is unused to preserve the D-02 novelty reservation. O3b
Fast_Scattering remains in test and is identifiable by label. A supervised
closed-set metric must treat this reserved class explicitly rather than
silently considering it a trainable known class. O1/O2 rows are unused.

## Versioned on-disk representation

Only five files are required, keeping the committed split indices compact:

| File | Contents |
| --- | --- |
| `catalog.npz` | One canonical ID array and aligned label/detector/run arrays |
| `P1.npz` | 40 uint8 membership arrays, one for every seed/k |
| `P2.npz` | 80 uint8 membership arrays, one for every seed/held-out rotation |
| `P3.npz` | 10 uint8 membership arrays, one per seed |
| `manifest.json` | Dataset digest, settings, exact file hashes/sizes, selectors and per-partition class/detector/run counts |

Position i in a membership array refers to position i in the catalog. Codes are
0=unused, 1=train, 2=validation, 3=test. `flatnonzero(membership == code)` gives
the partition indices; `catalog['id'][indices]` gives their IDs. Membership
codes give an exclusive assignment by construction, and the tests also assert
disjointness through the public ID API.

NPZ is NumPy's compressed archive format; no pickle/object arrays are used.
Each archive has fixed entry order, timestamps and compression settings.
File hashes and sizes are verified before loading; partition shape, codes,
ID uniqueness and counts are also checked. The archive is a committed input
artifact, not an intermediate cache to ignore in Git.

## Reproducibility and immutable versions

SHA-256 ranking of namespaced IDs defines every randomized ordering; Python's
process-randomized `hash()` and NumPy's mutable global RNG state are not used.
Canonical ordering also makes input row ordering irrelevant.

Generating the same dataset/settings again leaves existing files untouched.
A changed dataset, configuration or asset causes an error when targeting an
existing version. Write to `data/splits/v2/` for an intentional new version and
use `split_dir` to select it. Builds are staged before the version directory
is installed, so an incomplete generation is not presented as a complete bundle.

Byte-for-byte reproducibility is verified for the delivered toolchain. Future
NumPy or compression-library versions could encode an equivalent archive
differently; keep committed files as the experiment input and record the
toolchain when regenerating. The ID-selection ranking is deterministic.

## Usage

From the project root with D-02 integrated:

```python
from data.curate import load_curated
from data.splits import load_split, verify_dataset

df = load_curated()
verify_dataset(df)
s = load_split('P1', 0, 5, None)
indexed = df.set_index('id', verify_integrity=True)
train = indexed.loc[list(s.train_ids)]
validation = indexed.loc[list(s.val_ids)]
test = indexed.loc[list(s.test_ids)]

novelty = load_split('P2', 0, None, 'Fast_Scattering')
rotated = load_split('P2', 0, None, 'Blip')
temporal = load_split('P3', 0, None, None)
```

Select by ID, not by the current row number or DataFrame index. Fit scalers,
feature transforms and model/hyperparameter choices using only the allowed
training/validation data. Do not resample the fixed test pool per model.

Read the committed splits without the original CSVs:

```sh
python -c "from data.splits import load_split; s=load_split('P1',0,5,None); print(len(s.train_ids),len(s.val_ids),len(s.test_ids))"
```

Regenerate from default D-02 curation or from its persisted CSV:

```sh
python -m data.splits
python -m data.splits --curated-csv data/curated/curated.csv
```

Overrides include `--raw-dir`, `--output-dir`, `--test-per-class`,
`--val-per-class`, `--test-pool-seed`, `--p2-test-fraction`,
`--p2-val-fraction` and `--p3-val-fraction`. Use a fresh output directory
when changing settings. Seeds are fixed to 0–9 in the CLI; the Python
`SplitConfig` also supports a custom seed tuple for explicit alternative versions.

Run the tests with `python -m unittest discover -s tests -v`.

## Boundaries

Splits guarantee ID exclusivity and the stated class/run constraints. D-02
already deduplicates triggers within each detector. D-04 does not infer a
shared physical-event identity or cluster nearby timestamps across H1 and L1;
such a grouped analysis would require another documented protocol version.
The provisional seven-feature choice in D-02 and the actual L-02 `Split` type
remain group integration decisions.
