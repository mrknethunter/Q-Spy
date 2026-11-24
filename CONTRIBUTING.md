# Contributing to Q-Spy

## Workflow

1. Pick your issue on GitHub. One issue = one branch = one PR.
2. Branch from an up-to-date `main`, named after the task: `<TASK-ID>-<short-slug>`, e.g. `M-01-kernels`.
3. Open a PR into `main` titled `[<TASK-ID>] <issue title>`, with `Closes #<issue number>` in the description.
4. The team lead reviews and merges (squash). Address review comments on the same branch.
5. Before opening the PR, rebase on `main` and run the tests: `make test` (or `python -m pytest`).

Setup once: `python3 -m venv ~/.venvs/qspy && source ~/.venvs/qspy/bin/activate && make install`.
Add new dependencies to `requirements.txt` with an exact version.

## Where code goes

| Directory | Content | Tasks |
|---|---|---|
| `data/` | download, curation, protocol splits, hardware subsets | D-* |
| `features/` | feature transforms and scaling | D-03 |
| `quantum/` | feature map, observables, `embed` | Q-01..Q-03 |
| `backends/` | statevector, fake Heron, IBM Runtime | Q-03..Q-05 |
| `store/` | Bloch-vector store | I-04 |
| `kernels/` | projected, classical and fidelity kernels | M-01, M-05 |
| `models/` | few-shot, novelty, baselines | M-02..M-04 |
| `diagnostics/` | concentration, alignment, geometric difference, tier comparison | G-* |
| `experiments/` | YAML configs and runners | I-03, E-* |
| `tests/` | unit tests; `tests/contracts/` for interface tests, `tests/fixtures/` for synthetic data | all |
| `docs/` | contracts, data card, decisions, QPU policy, task notes | all |

Interfaces between modules are defined in `docs/contracts.md`. Do not change a signature there without the lead's approval.

## Rules

- Never commit raw data, credentials or IBM Quantum tokens.
- Code that submits QPU jobs must default to a dry run; real submissions need the lead's approval (see `docs/qpu_policy.md`).
- Keep files in the layout above; existing files outside it will be moved during the final cleanup.
