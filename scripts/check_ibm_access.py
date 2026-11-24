# Lists IBM backends without submitting jobs. Save credentials once with QiskitRuntimeService.save_account().
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
print("Operational backends:")
for b in service.backends(operational=True, simulator=False):
    status = b.status()
    print(f"  {b.name:<20} {b.num_qubits:>4} qubits   pending jobs: {status.pending_jobs}")
try:
    print("Usage:", service.usage())
except Exception as exc:  # not every plan exposes usage()
    print("Usage not available from this runtime version:", exc)
