# D-04 validation

Validated on 2026-11-21 with Python 3.12.14, pandas 2.2.3, NumPy 2.3.5
on Linux. Native Windows execution was not performed; Windows commands are in
`GUIDA_D04.md`.

## Executed verification

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` on D-02 plus D-04 | PASS: 60 tests, including 20 new D-04 tests and the existing 40 tests |
| Default generation on the real curated D-02 CSV | PASS: 130 splits for all 318,801 rows; P1=40, P2=80, P3=10 |
| Independent catalog comparison | PASS: catalog IDs, labels, detectors and observing runs exactly match the real curated table in canonical ID order |
| Independent verification of every real split | PASS: all 130 membership arrays, ID disjointness, allowed codes, partition sizes and per-class/detector counts verified |
| P1 exact training sizes | PASS: k examples per known class for k=5,10,20,50 across ten seeds |
| P1 fixed balanced test | PASS: the same 700 IDs in every P1 split, 100 per known class |
| P1 validation | PASS: 20 per known class, 140 total; test and validation are disjoint from training |
| P2 novelty reservation and rotations | PASS: all Fast_Scattering and each additionally rotated class are test-only; validation contains known labels only; test also includes normal examples |
| P2 complete assignment | PASS: every real retained ID belongs to exactly one partition in every P2 fold |
| P3 temporal separation | PASS: train/validation only known O3a rows; test exactly all O3b rows; reserved Fast_Scattering never appears in training/validation |
| Frozen asset hashes and sizes | PASS: all four NPZ files match the committed manifest |
| Public loader on real files | PASS: P1, default P2, rotated P2 and P3 return disjoint ID tuples |
| Dataset fingerprint verification | PASS: actual curated table matches the full data binding |
| Second full real-data generation from reversed row order | PASS: all five files byte-identical to the first generation |
| Repeat generation on an existing identical version | PASS in tests: no file rewrites or timestamp changes |
| Different data/settings targeting an existing version | Rejected; existing files remain unchanged |
| Corrupt assets, invalid selectors and missing versions | Rejected explicitly; loader does not silently sample replacement splits |

Generation command used the verified D-02 CSV from the earlier curation run:

```sh
python -m data.splits --curated-csv /path/to/verified/D02/curated.csv
```

The source was read using round-trip float parsing. Output was the default
`data/splits/v1/`; all protocol settings used defaults. The second generation
used the same values with row order reversed and a different temporary output
directory. No absolute input path enters the versioned files.

Dataset SHA-256:
`c8bb3d84efa5b583ee41d671ae195b73bb5cdf901e3af03e351e1e3eac0260f1`.

## Real default partition sizes

Counts below are the same for all ten seeds. Membership in the seeded
training/validation partitions changes; the stated fixed test pools do not.

### P1

| k per known class | Train | Validation | Test | Unused |
| --- | --- | --- | --- | --- |
| 5 | 35 | 140 | 700 | 317,926 |
| 10 | 70 | 140 | 700 | 317,891 |
| 20 | 140 | 140 | 700 | 317,821 |
| 50 | 350 | 140 | 700 | 317,611 |

### P2

Fast_Scattering remains unknown in every fold. Rotating another class adds it
to the unknown test set, rather than promoting Fast_Scattering to training.
Known normal examples are also in each test set.

| Selected held_out | Train | Validation | Test | Unused |
| --- | --- | --- | --- | --- |
| Fast_Scattering | 157,386 | 52,453 | 108,962 | 0 |
| Blip | 144,134 | 48,036 | 126,631 | 0 |
| Extremely_Loud | 145,267 | 48,415 | 125,119 | 0 |
| Koi_Fish | 145,936 | 48,639 | 124,226 | 0 |
| Low_Frequency_Burst | 140,662 | 46,879 | 131,260 | 0 |
| Low_Frequency_Lines | 145,778 | 48,584 | 124,439 | 0 |
| Scattered_Light | 89,771 | 29,916 | 199,114 | 0 |
| Tomte | 132,768 | 44,249 | 141,784 | 0 |

### P3

| Train O3a known | Validation O3a known | Test O3b | Unused |
| --- | --- | --- | --- |
| 60,889 | 15,215 | 176,199 | 66,498 |

All O3b rows are tested, including 34,605 reserved Fast_Scattering examples.
The unused rows comprise the earlier observing runs plus O3a Fast_Scattering.
The exact per-class/detector/run counts for each partition are in
`data/splits/v1/manifest.json`.

## Offline tests

The tests generate small independent curated fixtures with 1,920 rows, never
substituting their counts for the real statistics. They cover the complete
seed/protocol grid, disjoint IDs, exact training k, fixed balanced P1 test,
nested k subsets, detector-proportional sampling, P2 rotations and known-only
validation, P3 run restrictions, deterministic input-order independence,
byte-for-byte file identity, immutable versions, full feature-value binding,
integrity failures, invalid inputs/selectors, insufficient class/run support,
CSV CLI generation and the four-argument loader contract.

## Acceptance status and review choices

| Requirement | Status |
| --- | --- |
| `data/splits.py -> load_split(...)` | Implemented; local interface tests pass |
| Versioned split indices under `data/splits/` | Delivered in `v1/`, with all 130 real splits |
| P1 four k values × ten seeds and fixed balanced test pool | Verified on real and synthetic data |
| P2 reserved Fast_Scattering plus other rotations; validation known-only | Verified for the documented permanently reserved-Fast interpretation |
| P3 train O3a, test O3b | Verified; validation is a subset of known O3a and reserved O3a examples are unused |
| No ID in two partitions of a split | Verified for every split |
| Same seed/data/settings produce identical files | Verified byte-for-byte on all real assets |
| Shared L-02 contract suite/type | Not supplied; local tests use the issue's signature. Check the shared `Split` type and additional conventions after integration |
| Unspecified policy parameters | Defaults and their implications documented in `docs/splits.md` for group review |
| Feature list | Uses the provisional seven raw features carried over from D-02; changing inputs requires a new split version |
| Remote upload or merge | Not performed by this package; follow `GUIDA_D04.md` |

P1 validation uses a separate labelled budget beyond k. Rotated P2 folds have
two unknown classes because Fast_Scattering stays reserved. P3 also preserves
that reservation. These are explicit choices, not additional requirements
claimed to have been numerically specified in the issue.
