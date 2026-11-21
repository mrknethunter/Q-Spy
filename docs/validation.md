# D-01 validation

Validated on 2026-11-19 using Python 3.12 on Windows.

| Check | Result |
| --- | --- |
| First full `python data/download.py` execution | PASS: downloaded all 8 files, validated published byte sizes and MD5 values |
| Pinned manifest comparison with Zenodo record 5649212 API | PASS: all 8 names, sizes and MD5 checksums match |
| Second full downloader execution from a different working directory | PASS: 8 files reused, exit code 0, all modification timestamps unchanged |
| `python data/profile.py` on all 8 originals | PASS: 676,613 rows, 43 columns in every file |
| Independent pandas check with round-trip float parsing | PASS: per-file row counts, exact class counts and all 7 feature min/max match |
| `python -m unittest discover -s tests -v` | PASS: 16 tests |

The independent pandas comparison is a development check, not a runtime
dependency. Both delivered scripts and the tests use only the Python standard
library. Unit tests cover same-size corruption repair, incorrect checksums,
oversized and incomplete responses, bounded retries, interruption cleanup,
offline verification failure, idempotence, histogram boundaries, quantiles,
missing/non-numeric/non-finite values, invalid detector identifiers, incomplete
datasets, zero confidence and extreme Q values.

Observed raw-data quality flags, preserved without curation:

| File | q_value == 1e20 | ml_confidence == 0 | snr == 7.5 |
| --- | --- | --- | --- |
| L1_O1.csv | 14,047 | 52 | 0 |
| L1_O2.csv | 36,745 | 0 | 0 |
| L1_O3a.csv | 0 | 0 | 1 |

The other five files have zero counts for these three flags. The release does
not establish the meaning of the extreme Q value; treating it as a sentinel
requires a downstream decision.

## Acceptance status

| Requirement | Status |
| --- | --- |
| Reproduce all raw CSVs with one command | Verified |
| Idempotent download and integrity checks | Verified |
| Raw data excluded from the deliverable and Git ignore rules supplied | Implemented |
| Columns, units/caveats, run/detector class counts, confidence, licence and citation | Documented |
| Raw ranges for the seven agreed project features | Ranges computed for a provisional seven-column vector; team confirmation still required |
| Exact physical amplitude normalization/unit | Not explicitly stated in the release; caveat documented |
| L-01 structure, protected branches and remote CONTRIBUTING conventions | Only screenshot instructions available; cannot verify against current private repository |
| Pull request on current develop | Not created; private repository was inaccessible through the available GitHub connection |

This snapshot is ready for review and integration. Do not mark D-01 fully
accepted until the agreed feature list and project dependency are confirmed.
