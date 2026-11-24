# Issue draft — not yet opened

Title: [M-03 follow-up] Kernel novelty detectors on a shared Gram matrix

Requested in E-03 PR #67, issue #29. Implement kernel density, kernel-space
kNN and kernel-PCA reconstruction error using the **same** training/evaluation
Gram matrices, IDs, kernel parameters and preprocessing for all three.

Acceptance criteria:

- Known-only fitting; exclude Fast_Scattering and the outer unknown class from
  all fitting, calibration and selection, including preprocessing.
- Nested known-family LOCO selection for any free parameters; audit search
  budgets, surrogate AUROC and all candidate settings.
- Larger scores mean more anomalous; calibrate operating points on known
  validation only, with no test-based threshold or score-direction selection.
- Tests for kernel distance, centering, reconstruction, out-of-sample scoring
  and numerical degeneracies, checked against small analytical examples.
- Add E-03 ROC, AUROC/AP/TPR@1%FPR and paired uncertainty for all three detectors.
- E-04 uses the exact hardware P2 seed-0 training scaler on both ideal and
  hardware representations.

This follow-up does not assert these detectors are already delivered in PR #67.
