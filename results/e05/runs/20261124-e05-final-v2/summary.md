# E-05 hardware diagnostics summary

**Status:** `blocked_by_prerequisites` · **Hardware validation:** `not_validated` · **Data source:** `no H-02 campaign`

**Campaign:** `not available` · **Embedding freeze:** n=8, L=2, s=0.1778 · **D-05:** `bound_to_hw_freeze` · **D-05 approval:** `pending` · **L-05 tag:** `hw-freeze` · **L-05 approval:** `pending`
**D-05 selection SHA-256:** `7fee3e50508c951f1161db9e3cd593ea66e609e572d3e38dc7bdbbbc58a0c7c7` · **Gamma:** `not frozen or supplied`

## Diagnostics

No campaign metrics were calculated.

**Generated figures:** None; see figure blockers below.

## Missing figures and limits

- `bloch_rmse_per_qubit.svg`: Matched H-02 HW/L0 Bloch errors from G-03 are unavailable.
- `kernel_alignment.svg`: H-02 kernels, an explicit gamma and G-03 kernel alignment are unavailable.
- `concentration_variance.svg`: Matched H-02 runs and an explicit gamma are unavailable.
- `effective_rank.svg`: Matched H-02 runs and an explicit gamma are unavailable.
- `drift_rmse_per_qubit.svg`: H-02 drift-start/end references and G-03 drift_report are unavailable.
- `eigen_spectrum_<experiment>.svg`: No H-02 group has matched runs and an explicit gamma for its projected-kernel spectrum.

**Blocked prerequisites:**

- gamma is not defined by the frozen configuration and was not supplied explicitly
- docs/decisions.md records L-05 as draft for lead approval
- D-05 documentation still describes the subset as a draft/review proposal without final acceptance
- No complete H-02 campaign matching D-05 IDs and frozen embedding metadata is available
- Pilot-like HW runs were found, but do not match the H-02 campaign run manifest and were not substituted
- G-03 diagnostics.compare is unavailable in this checkout: No module named 'diagnostics.compare'

H-01 pilot data are not substituted for H-02. Synthetic fixture results, if used in tests, are not hardware results. No QPU job was submitted by E-05.
