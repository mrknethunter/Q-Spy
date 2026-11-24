# Decisions and pre-registration (L-05)

Status: **draft for lead approval**. Frozen when the lead tags the merge commit `hw-freeze`; after that, any change
needs a new tag (`hw-freeze-v2`) and a written reason here. Machine-readable values: `configs/hardware_frozen.yaml`.

## 1. Hardware configuration

| Parameter | Value | Why |
|---|---|---|
| Qubits n | 8 | ≥ 7 needed to encode all 7 features; label alignment plateaus beyond 8 (E-01); validated by the pilot |
| Layers L | 2 | two-qubit depth 4 on ibm_kingston, no SWAPs; deeper circuits align worse (E-01) |
| Bandwidth s | 0.1778 | maximum label alignment of the E-01 grid; s = 1 (pilot value) is 40% worse |
| Entanglement | CZ chain, on | the product-state ablation (R2) is computed classically, no QPU needed |
| Precision | 0.03 (1112 shots) | no alignment loss vs exact; 0.05 already loses 0.01 (Q-06) |
| Mitigation | TREX + twirling + DD | pilot: hardware at the shot-noise floor; control subset runs without it |
| Feature scaling | D-03 scaler fitted on P2 seed-0 training IDs | same scaling for E-01, the ML experiments and the hardware points; no test leakage |
| Device | ibm_kingston | pilot backend; empty queue at the time of the pilot |
| Budget | 587 s available | estimate 413 s with the 30% margin (318 s from IBM's formula, which matched the pilot within 1 s) |

## 2. Evidence

**E-01, concentration map on real data** (`results/e01/e01_gravityspy.csv` and `.png`): 196 glitches from the P2 seed-0
training pool (28 per known class), exact simulation, n ∈ {4…16}, L ∈ {1, 2, 4, 8}, 9 bandwidths, projected (γ = 1) and
fidelity kernels.
- Bandwidth dominates: at n = 8, L = 2 the kernel-target alignment (KTA) goes from 0.36 at s = 1 to 0.53 at s = 0.18 and
  falls below 0.25 for s > 2 (panel d).
- The fidelity kernel concentrates as n grows (off-diagonal variance ×8 smaller from 4 to 14 qubits at s = 1), less than
  the worst case because only 7 distinct features exist; the projected kernel stays flat (panel a).
- Beyond 8 qubits alignment does not improve (0.534 at n = 8, 0.535 at n = 10, 0.530 at n = 12).
- Honest caveat: at n = 8 the fidelity kernel aligns slightly better than the projected one (0.556 vs 0.534). The
  projected kernel is chosen for hardware cost: 360 circuits for the 120 main points instead of 7140.

**Q-06, emulation on the 120 main hardware points** (all outside training and validation):

| Configuration | KTA exact (L0) | shot noise (L1) | Kingston noise model (L2) |
|---|---|---|---|
| s = 0.1778, precision 0.03 | 0.531 | 0.529 | 0.531 |
| s = 0.1778, precision 0.05 | 0.531 | 0.525 | 0.521 |
| s = 1.0, precision 0.03 | 0.319 | 0.317 | 0.317 |

The Bloch vectors of different glitches differ by 0.23 on average, eight times the measurement noise, so the chosen
configuration is robust to it; 0.531 on unseen points matches the 0.534 of the training sample.

**H-01 pilot** (`docs/qpu_policy.md`): mitigated hardware error 0.033 vs 0.030 from shot noise alone; usage 13 s vs
13.7 s predicted.

**Risk acknowledged:** small rotation angles keep the circuit close to a weakly entangled regime, where classical
surrogates may do as well (Cerezo et al. 2025). Hypothesis H2 below tests exactly this.

## 3. Pre-registered hypotheses

Primary hypotheses, tested once on the frozen configuration, Holm-corrected as one family. Everything else is reported
as exploratory.

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Few-shot (P1, k ≤ 10): the projected-kernel SVM (R3) is not worse than the RBF SVM on the raw features (R1) | paired Wilcoxon on macro-F1 over 20 seeds, per k |
| H2 | Entanglement helps: R3 has higher macro-F1 than the same kernel without CZ gates (R2), k ≤ 10 | paired Wilcoxon over 20 seeds |
| H3 | Novelty (P2, Fast_Scattering held out): One-Class SVM on the projected kernel has higher AUROC than the RBF One-Class SVM | paired over seeds and surrogate rotations |
| H4 | Hardware (RQ1): on the 120 main points the hardware kernel keeps the label alignment within 0.02 of the exact one | ΔKTA with bootstrap 95% CI over points |

## 4. Hyperparameter procedure (values chosen later, no QPU time)

γ, C and ν are not frozen: they are chosen on the stored Bloch vectors with this procedure, identical for every model.
- γ = c / median of squared training distances, c ∈ logspace(−2, 2, 9); C ∈ logspace(−2, 3, 6).
- Equal search budget for every model (60 configurations); ties broken towards smaller C, then smaller γ.
- Selection: leave-one-out for k ≤ 10, 5-fold × 2 for k ≥ 20; a heuristic-only regime (median γ, C = 1) is also reported.
- Novelty: ν from a 5% false-alarm target on known-class validation; γ by nested leave-one-class-out with surrogate
  held-out classes, never Fast_Scattering.

## 5. Hardware campaign (H-02)

| Experiment | Points | Circuits | Estimate (s, with margin) |
|---|---|---|---|
| drift-start | 5 | 15 | 10 |
| main, mitigated | 120 | 360 | 185 |
| control, unmitigated | 40 | 120 | 63 |
| concentration n = 4, 8, 12 | 3 × 30 | 270 | 144 |
| drift-end | 5 | 15 | 10 |
| **Total** | | **780** | **413** |

All jobs run in one Batch. Stop rules: if measured usage exceeds the estimate by more than 30%, or drift-start is further
than 0.08 RMSE from the exact result, stop and re-plan before continuing.
