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
| | | | | | | | |

Budget remaining this window: **600 s** (update after every job).
