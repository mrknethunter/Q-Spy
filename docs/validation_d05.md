# D-05 validation and pending L-05 dependency

Validated on 2026-11-21 with Python 3.12.14, pandas 2.2.3 and NumPy 2.3.5
on Linux. Windows instructions are supplied; native Windows execution was not
performed. L-05 is still pending, as explicitly confirmed by the contributor.

## Delivered real selection

| Property | Result |
| --- | --- |
| Curated source | Default verified D-02 table of 318,801 rows |
| Candidate definition | D-04 P2 seed-0 Fast_Scattering test, O3b only, excluding every committed D-04 training/validation ID |
| Candidate count | 62,581 real events |
| Main | 120; 15 per family; H1=60, L1=60 |
| Control | 40; 5 per family; H1=20, L1=20 |
| Concentration | 3 × 30, each H1=15/L1=15, each family count 3 or 4 |
| Drift references | 5 main points across five distinct families including Fast_Scattering; H1=3/L1=2 |
| Unique event count overall | 160 |
| Lifecycle | `draft_pending_l05`; `l05_freeze` is null |
| Selection SHA-256 | `7fee3e50508c951f1161db9e3cd593ea66e609e572d3e38dc7bdbbbc58a0c7c7` |

## Executed checks

| Check | Result |
| --- | --- |
| Complete D-02/D-04/D-05 unit suite | PASS: 76 tests, including 16 new D-05 tests |
| Generation against actual D-02 values and D-04 version | PASS; source fingerprint, catalog and all protocol assets verified |
| Independent real-ID membership | PASS: all 160 unique selected IDs resolve to actual O3b catalog events |
| Independent main/control balance | PASS: exactly 15/5 per each of eight families, exact 60/60 and 20/20 detector counts |
| Main/control disjointness | PASS |
| Concentration | PASS: each group has 30 unique IDs, all within main, union has 90 IDs |
| Drift references | PASS: five main-set IDs, five distinct families and exactly one Fast_Scattering |
| All-protocol leakage check | PASS: selected IDs never appear in train or validation in any of the 130 frozen D-04 splits |
| P2/P3 test membership | PASS: every selected ID belongs to test in every current P2 and P3 split |
| Independent candidate-count reconstruction | PASS: 62,581 after the declared source filters |
| Reversed input-order generation | PASS: byte-identical real JSON |
| Default consumer on delivered pending file | PASS: rejects it with explicit pending-L-05 error |
| Synthetic Git-tag binding | PASS in temporary test repositories; real resolved commit/tagged blobs recorded; selected IDs/digest unchanged |
| Missing tag, mismatched manifest and moved tag | Rejected in tests |
| Uncommitted config changes | Tests confirm binding reads tagged blobs, not working-tree replacements |
| Existing selection version | Identical output reused; a different selection cannot silently overwrite it |

The synthetic Git fixtures use a dummy YAML with `purpose: synthetic_unit_test_only`
and a decision file explicitly stating it is not a real group approval. They
are created and deleted inside temporary test directories. No real L-05
configuration, approval record or `hw-freeze` tag was created for this project.

## Acceptance status

| Requirement | Status |
| --- | --- |
| `data/splits/hardware_subsets.json` | Delivered as a real-ID pending proposal |
| Main 120 balanced including held-out | Implemented and verified |
| Control 40 | Implemented and disjoint from main |
| Three concentration subsets of 30 | Implemented; mutually disjoint, nested within main by documented choice |
| Five drift reference points | Implemented; intentional main-point reuse documented |
| Fixed and reproducible IDs | Verified, including byte-identical regeneration |
| Disjoint where required and documented | Verified for the explicit policy in this proposal |
| Actual approved L-05 configuration and tag | Not available; dependency remains open |
| Binding to actual hardware freeze | Supported but not executed on the real project |
| Group agreement on pool/overlap policy and chosen models | Required with L-05 before accepting D-05 for hardware use |
| QPU/hardware results | None generated or used |

This package completes a tested review proposal. It does not establish completion
of L-05 or final acceptance of D-05. Group approval is not inferred from a Git
tag; the binding records provenance, while review approval and scientific
configuration correctness remain in the L-05 workflow.
