# D-02 validation

Validated on 2026-11-21 with Python 3.12.14, pandas 2.2.3 and NumPy 2.3.5
on Linux. Windows instructions are supplied in `docs/GUIDA_D02.md`; a native
Windows execution of this D-02 snapshot was not performed.

## Executed checks

| Check | Result |
| --- | --- |
| D-01 downloader on the pinned original release | PASS: all 8 files present, published byte sizes and MD5 checksums verified |
| `python -m unittest discover -s tests -v` | PASS: 40 tests, comprising the 16 existing D-01 tests and 24 D-02 tests including the local interface contract |
| Full `python -m data.curate` pipeline on the verified sources | PASS: 676,613 raw rows -> 318,801 curated rows |
| All retained rows independently joined back to the original CSVs | PASS: every ID has a matching source variant with identical label, confidence and all seven feature values; GPS matches the precise peak timestamp converted to float64 seconds |
| Original ID uniqueness assumption | One repeated detector-scoped ID found in the release; curation removes one row. Independent verification accounts for multiple raw variants per ID |
| Independent confidence-filter count | PASS: 183,146 source rows fail the inclusive [0.9, 1] criterion |
| Detector-local temporal separation on retained exact timestamps | PASS: every consecutive retained pair in the same detector is separated by more than 100,000,000 ns |
| Returned IDs, finite/positive features and confidence | PASS: unique IDs, no missing values, finite positive raw features, confidence in [0.9, 1], no retained Q=1e20 under defaults |
| Held-out restriction, non-vacuous | PASS: 56,509 Fast_Scattering rows, all in O3a/O3b, all marked is_held_out=True |
| Per-detector class counts | PASS: independent output counts agree with the generated report |
| Exclusion accounting | PASS: input minus sequential exclusion counts equals output count |
| Detector-stratified cap allocation bounds | PASS: exhaustive checks for H1/L1 counts 1–39 and every feasible target; allocations sum to target and retain both detectors without replacement |
| D-01 preservation | PASS: download.py, profile.py, manifest, original tests, raw summary and original validation document are byte-identical to the supplied D-01; original data-card text is retained before the appended D-02 block |

The full curation command was executed with `--raw-dir` pointing to the existing
D-01 download directory and `--output-dir` pointing to a temporary verification
directory. Other parameters used defaults. These path overrides do not change
the row selection or report contents. The derived CSV was also read back using
round-trip float parsing for the independent source comparison.

## Final default class counts

| Label | H1 | L1 | Total |
| --- | --- | --- | --- |
| Blip | 14,890 | 7,196 | 22,086 |
| Extremely_Loud | 12,614 | 7,581 | 20,195 |
| Koi_Fish | 11,134 | 7,944 | 19,078 |
| Low_Frequency_Burst | 20,601 | 7,271 | 27,872 |
| Low_Frequency_Lines | 4,735 | 14,611 | 19,346 |
| Scattered_Light | 61,469 | 51,220 | 112,689 |
| Tomte | 1,627 | 39,399 | 41,026 |
| Fast_Scattering (held-out) | 1,280 | 55,229 | 56,509 |
| **Total** | **128,350** | **190,451** | **318,801** |

The counts by detector and observing run, sequential exclusion counts,
configuration and source checksums are in `curation_summary.json`. The source
and scientific context remain documented in the D-01 section of `data_card.md`.

## Scope of the tests

The D-02 tests cover the zero-argument loader over eight temporary pinned CSVs;
missing/corrupt/incomplete sources and bad detector provenance; confidence
boundaries; invalid measurements and integer timestamps; configurable extreme-Q
exclusion; O3-only held-out data; repeated IDs; nanosecond-exact coincidence
boundaries; transitive clusters; label-independent duplicate selection; H1/L1
separation; deterministic conflict resolution and sampling under reordered inputs;
per-detector class support; without-replacement caps; report accounting;
idempotent data-card updates; malformed report markers; and CLI output generation.

## Acceptance and remaining integration work

| Requirement | Status |
| --- | --- |
| `data/curate.py` with zero-argument `load_curated() -> pd.DataFrame` | Implemented and locally contract-tested |
| Configurable confidence threshold, default 0.9 | Verified |
| Coincident-trigger deduplication | Verified; detector-local connected clusters with configurable 0.1 s default |
| Detector stratification | Detectors preserved and reported; optional class caps sample within detector strata; split assignment belongs to the later split task |
| 6–8 known classes plus held-out | Seven explicit known labels plus Fast_Scattering; justification and observed counts documented |
| Class counts per detector | Verified on original sources and reported |
| Fast_Scattering only in O3 | Verified on original sources and tested against a deliberately invalid O2 held-out fixture |
| Actual shared L-02 contract suite | Not supplied in this branch; run after integration. Local tests do not establish that an unseen suite passes |
| Seven agreed project features | D-01's provisional list is preserved; agreement by the group still required |
| Coincidence/Q/class-selection policy review | Explicit reproducible defaults supplied for review; these choices were not specified numerically by the issue |
| Pull request and remote repository layout | Not changed by this delivery; use the integration guide and the group's current branch conventions |

This is a tested implementation and review package, not evidence of remote merge
or of approval of the provisional feature list.

## Integration review, 2026-11-24

The current main snapshot now includes the shared L-02 contracts. See
`docs/D02_review_changes.md` for the new 205-test full-suite result, canonical
D-01 test consolidation, common requirements and relocated guide. Historical
2026-11-21 results above describe the original standalone delivery.
