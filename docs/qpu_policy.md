# QPU usage policy (L-03)

QPU time is the scarcest resource of the project. Every campaign is planned to fit in **10 minutes of QPU usage**,
the free-plan allocation per account per 28-day window. Paid usage is never enabled.

## Access

- Each member creates a free IBM Quantum account and saves credentials **once, outside the repository**:
  `QiskitRuntimeService.save_account(token=..., ..., set_as_default=True)`.
- Check access with `python scripts/check_ibm_access.py`: it lists backends and never submits jobs.
- Never commit tokens, `qiskit-ibm.json` or `apikey.json`; they are git-ignored and GitHub secret scanning is on.

## Who may submit

Only the owner of the hardware tasks (H-01, H-02), from the account agreed with the lead.
Code that can reach a QPU defaults to a dry run and submits only with an explicit `submit=True` / `--submit`.

## Approval: issues labelled `qpu`

1. Run the dry run and post in the issue: backend, circuits, shots, executions, **estimated QPU seconds**.
   Estimate: `1.3 x (2 s x sub-jobs + 0.35 ms x circuits x shots)`.
2. The lead replies `QPU approved: <seconds>` in the issue. Nothing is submitted before that comment.
3. Submit, then paste the `ledger_row` printed by `scripts/qpu_pilot.py` (measured `job.usage()` included) into the
   ledger below, in the same PR as the `store/runs/hw-*` results.

## Rules

- Pilot first: the 10-point pilot (H-01) calibrates the estimate before any larger job.
- One short job per experiment (calibration drift degrades long runs); store job IDs and calibration with the results.
- If measured usage exceeds the estimate by more than 30%, stop and re-plan with the lead.
- Cancel a job only if it is wrong; cancelled and failed jobs still consume the time already used.
- Test the full code path on the fake Heron backend (L2) before every submission.

## Ledger

| Date | Issue | Account | Backend | Purpose | Estimated s | Measured s | Job IDs |
|---|---|---|---|---|---|---|---|
| 2026-11-24 | H-01 | mrknethunter | ibm_kingston | pilot, 10 points, 8 qubits, L=2 | 17.8 | 13 | dc2e0n6q4uuc73a569f0 |

Budget remaining this window: **587 s** (window opened 2026-11-24 00:51 UTC; update after every job).

## H-01 pilot decision: GO

Run `hw-20261124T005515Z` (`store/runs/hw-20261124T005515Z-*`, summary in `results/pilot/`), synthetic features,
8 qubits, L=2, precision 0.03, TREX + twirling + DD, physical chain 67-73 + 79, two-qubit depth 4.

- **Usage estimate is calibrated:** 13 s measured vs 13.7 s from IBM's formula (17.8 s with our 30% margin).
- **Mitigated hardware is at the shot-noise floor:** RMSE vs exact 0.033 (shot noise alone, L1: 0.030), correlation
  0.997. Hardware error beyond shot noise is about 0.014.
- **L2 is a reliable proxy at this depth:** HW vs L2 RMSE 0.047, what two independent 0.03-level noises give (0.047);
  no systematic offset.
- Limits: synthetic points, shallow circuit. Re-check on L2 if the frozen configuration (L-05) is much deeper or wider.
