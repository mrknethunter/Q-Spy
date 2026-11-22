# D-05 — Hardware subsets and reference points

## Current lifecycle state

**L-05 is not yet completed**, as confirmed by the project contributor.
The delivered `data/splits/hardware_subsets.json` is therefore explicitly
`draft_pending_l05`, with `l05_freeze: null`. Its event IDs are real and fixed;
it is a review proposal, not an approved hardware configuration.

The issue requires a main set of 120 balanced points including the held-out
family, 40 control points, three concentration groups of 30, and five drift
reference points. It does not specify the exact candidate pool, overlap rules
between those roles or detector allocation. The choices below are explicit
implementation decisions for review with L-05.

## Source and deterministic selection

Candidates are drawn from the frozen D-04 **P2 seed-0 Fast_Scattering-fold test**,
restricted to O3b and excluding every ID used in training or validation by
**any of the 130 committed D-04 splits**. The available pool is 62,581 events.
The selected unique IDs are also in the test partition of every current P2
and P3 split. P1 training/validation IDs are excluded, but these extra hardware
IDs are not claimed to be the fixed P1 test pool.

The D-04 manifest and all four NPZ assets are integrity-checked. Selection is
bound to its full curated-dataset SHA-256 and the exact split-manifest bytes.
`verify_dataset` additionally checks all D-02 feature values and provenance.
The provisional D-02 feature list is inherited without changing or fitting it.

SHA-256 ranking of IDs, namespaced by role, class, detector and selection seed
2026, selects the events. Neither raw feature values, model performance nor
quantum/hardware outputs determine which candidates are preferred. Reordering
the input table cannot change the result. Class and detector labels are used
only for the declared balance quotas.

| Binding | Value |
| --- | --- |
| D-02 dataset SHA-256 | `c8bb3d84efa5b583ee41d671ae195b73bb5cdf901e3af03e351e1e3eac0260f1` |
| D-04 source manifest | `data/splits/v1/manifest.json` |
| Experiment source | P2 / seed 0 / Fast_Scattering fold / test |
| Additional source filters | O3b only; exclude all committed D-04 train/validation IDs |
| Candidate pool | 62,581 |
| Selection seed | 2026 |
| Selection SHA-256 | `7fee3e50508c951f1161db9e3cd593ea66e609e572d3e38dc7bdbbbc58a0c7c7` |

## Sets and permitted overlap

| Set | Size | Class allocation | H1 / L1 | Relationship |
| --- | --- | --- | --- | --- |
| Main | 120 | 15 from each of 8 families | 60 / 60 | Disjoint from control |
| Control | 40 | 5 from each family | 20 / 20 | Disjoint from main and therefore its subgroups |
| Concentration 1 | 30 | 3 or 4 per family | 15 / 15 | Subset of main |
| Concentration 2 | 30 | 3 or 4 per family | 15 / 15 | Subset of main; disjoint from groups 1 and 3 |
| Concentration 3 | 30 | 3 or 4 per family | 15 / 15 | Subset of main; disjoint from groups 1 and 2 |
| Drift references | 5 | 5 distinct families, including Fast_Scattering | 3 / 2 | Subset of main; intentional reuse, may overlap concentration |

The eight families are Blip, Extremely_Loud, Fast_Scattering, Koi_Fish,
Low_Frequency_Burst, Low_Frequency_Lines, Scattered_Light and Tomte.
Main therefore contains 15 held-out events and control contains 5. All selected
events are O3b, consistent with the source dataset's O3-only Fast_Scattering.

Class quotas for groups of 30 distribute the remainder cyclically, avoiding
always assigning the extra slots to the same family. Each group's detector
quotas are balanced globally and differ by at most one within a family.
The three concentration groups use 90 distinct main points, leaving 30 other
main points outside those groups. This is a fixed selection for repeated
concentration measurements, not groups chosen after observing kernel values.

Four known reference families are selected by a separate deterministic
class ranking, plus Fast_Scattering. One main-set point from each is retained.
The reference families in this proposal are Extremely_Loud, Fast_Scattering,
Koi_Fish, Low_Frequency_Lines and Scattered_Light. Reference measurements reuse
the same IDs over time; this controlled overlap is intentional. In the delivered
selection, the five reference points overlap concentration groups in counts
1, 2 and 1 respectively; one reference is among the other 30 main points.

There are **160 unique events overall**, not 255 distinct events: the 90
concentration and five reference IDs reuse main points. An eventual hardware
experiment may still measure reused points repeatedly or under different
settings; this file does not schedule jobs or assume a circuit/QPU cost.

## JSON contract and validation

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema` | `q-spy-hardware-subsets-v1` |
| `status` | `draft_pending_l05` or `bound_to_hw_freeze` |
| `selection_seed` | Fixed data-selection seed |
| `source` | Dataset digest, manifest digest/path and candidate-pool definition |
| `subsets.main`, `subsets.control` | Ordered IDs, sizes, class/detector/run counts |
| `subsets.concentration` | Three named groups with the same per-group metadata |
| `subsets.drift_references` | Five ordered reference IDs and counts |
| `overlap_policy` | Explicit source, disjointness and intentional-reuse policy |
| `selection_sha256` | Digest of the selection and source; excludes lifecycle/freeze metadata |
| `l05_freeze` | Null while pending; actual Git tag/commit and tagged-file hashes after binding |

Every ID is unique within its own set and sorted. The validator recomputes
all counts against the integrity-checked D-04 catalog, verifies source membership,
cardinality, class/detector balance, required disjointness and reference-family
coverage. A changed source manifest or curated feature table is rejected.

Review the current pending file explicitly:

```python
from data.hardware_subsets import load_hardware_subsets
proposal = load_hardware_subsets(allow_draft=True)
```

Default `load_hardware_subsets()` rejects the current proposal because L-05
is pending. The `allow_draft` flag is for inspection and tests, not a scientific
approval or an instruction to run hardware jobs.

## Link to L-05 after the team completes it

L-05 belongs to the team lead and requires group approval of n, L, s, gamma and
the models before hardware results are observed. This D-05 package does not
select those values, create `configs/hardware_frozen.yaml`, write an invented
approval decision, or create/move/push a `hw-freeze` tag.

Once the actual approved L-05 tag is available in a local Git checkout, bind it:

```sh
python -m data.hardware_subsets --curated-csv data/curated/curated.csv --bind-freeze-from .
```

The binding routine only reads Git. It resolves `refs/tags/hw-freeze^{commit}`
and reads these blobs **from that commit**, not from uncommitted working files:

- `configs/hardware_frozen.yaml`
- `docs/decisions.md`
- `data/splits/v1/manifest.json`

All must be nonempty, and the tagged split manifest must exactly match this
proposal's source manifest. The JSON records the actual resolved commit and
SHA-256/size of each tagged file. It becomes `bound_to_hw_freeze`; the selection
digest and every selected ID stay unchanged. A draft can be promoted in place
only when that exact selection is preserved. A different or already-bound
selection requires a new output version.

The Git binding proves which tagged bytes were referenced. It does not inspect
GitHub review approvals, validate the scientific merits/schema of an unknown
L-05 YAML format, or prove that all members approved. Those are L-05 workflow
responsibilities; review its real decision record and confirm that the selected
models/protocols accept this P2/O3b hardware evaluation pool. The tag alone is
not represented as an automatically verified group approval.

After binding, a consumer can verify the live local tag has not moved:

```python
subset = load_hardware_subsets(freeze_repo='.')
```

The recorded commit remains pinned even if a tag later changes. Explicit live
verification rejects a moved tag or mismatched tagged files.

## Reproduce and version changes

```sh
python -m data.hardware_subsets
# Or use an already generated D-02 CSV with round-trip float parsing:
python -m data.hardware_subsets --curated-csv data/curated/curated.csv
```

Options include `--split-dir`, `--seed`, `--output` and `--raw-dir`.
Generation from the default curated loader verifies the raw D-02 originals.
An identical output is reused without rewriting. Changing the seed/data/selection
does not silently overwrite an existing selection: choose a new `--output`
and review the resulting hardware experiment version. No hardware results have
been produced or used by this delivery.
