"""Check IBM Quantum access without submitting anything (L-03).

Usage: python scripts/check_ibm_access.py
Credentials must already be saved once with QiskitRuntimeService.save_account(...), outside the repository.
"""
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
print("Operational backends:")
for b in service.backends(operational=True, simulator=False):
    status = b.status()
    print(f"  {b.name:<20} {b.num_qubits:>4} qubits   pending jobs: {status.pending_jobs}")
try:
    print("Usage:", service.usage())
except Exception as exc:  # usage() is not exposed by every runtime version or plan
    print("Usage not available from this runtime version:", exc)
