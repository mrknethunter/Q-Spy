from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from quantum.feature_map import feature_map
from quantum.observables import bloch_observables, evs_to_bloch

DEFAULT_DEVICE = "ibm_kingston"


@dataclass(frozen=True)
class EmbedConfig:
    n_qubits: int
    layers: int
    s: float
    entangle: bool = True


@dataclass
class EmbedResult:
    B: np.ndarray | None
    meta: dict = field(default_factory=dict)


def estimate_usage(n_circuits: int, shots: int, n_subjobs: int = 1) -> float:
    """IBM's documented estimate (2 s per sub-job + 0.35 ms per execution) with a 30% margin."""
    return 1.3 * (2.0 * n_subjobs + 0.35e-3 * n_circuits * shots)


def embed(X_tilde: np.ndarray, cfg: EmbedConfig, backend: str = "L0", *, precision: float = 0.03,
          seed: int | None = None, submit: bool = False, device=DEFAULT_DEVICE) -> EmbedResult:
    X_tilde = np.asarray(X_tilde, dtype=np.float64)
    n = cfg.n_qubits
    qc = feature_map(n, cfg.layers, X_tilde.shape[1], cfg.entangle)
    angles = cfg.s * np.pi * X_tilde[:, : qc.num_parameters]
    meta = {"tier": backend, "embed_config": asdict(cfg), "n_points": len(X_tilde)}

    if backend == "L0":
        # exact (no sampling) and ~70x faster than StatevectorEstimator
        est = AerEstimator(options={"backend_options": {"method": "statevector"}, "run_options": {"approximation": True}})
        evs = est.run([(qc, bloch_observables(n), angles)]).result()[0].data.evs
        return EmbedResult(evs_to_bloch(evs, n), meta)

    meta["precision"] = precision
    if backend == "L1":
        est = AerEstimator(options={"default_precision": precision, "run_options": {"seed": seed}})
        evs = est.run([(qc, bloch_observables(n), angles)]).result()[0].data.evs
        return EmbedResult(evs_to_bloch(evs, n), meta)

    if backend == "L2":
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        fake = FakeKingston()
        isa = _compile(qc, fake, seed, meta)
        # shortcut: no server-side mitigation (TREX, twirling), so L2 is pessimistic; recalibrate on the pilot HW_vs_L2
        est = AerEstimator(options={"default_precision": precision, "backend_options": {
            "noise_model": NoiseModel.from_backend(fake), "seed_simulator": seed}})
        evs = est.run([(isa, bloch_observables(n, isa.layout), angles)]).result()[0].data.evs
        return EmbedResult(evs_to_bloch(evs, n), meta)

    if backend == "HW":
        n_circuits, shots = 3 * len(X_tilde), math.ceil(1.0 / precision**2)  # X, Y, Z bases: 3 settings per point
        meta.update(n_circuits=n_circuits, shots=shots, executions=n_circuits * shots,
                    estimated_qpu_s=estimate_usage(n_circuits, shots), dry_run=not submit)
        if not submit:
            from qiskit_ibm_runtime.fake_provider import FakeKingston
            _compile(qc, FakeKingston(), seed, meta)  # catch compile errors and report depth before spending QPU time
            return EmbedResult(None, meta)
        from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
        dev = QiskitRuntimeService().backend(device) if isinstance(device, str) else device
        isa = _compile(qc, dev, seed, meta)
        est = EstimatorV2(mode=dev)
        o = est.options
        o.default_precision = precision
        o.resilience.measure_mitigation = True
        o.twirling.enable_gates = True
        o.twirling.enable_measure = True
        o.dynamical_decoupling.enable = True
        if seed is not None and hasattr(o, "simulator"):
            o.simulator.seed_simulator = seed  # only fake devices take a simulator seed
        meta["calibration"] = _calibration(dev, meta["layout"])
        job = est.run([(isa, bloch_observables(n, isa.layout), angles)])
        meta["job_ids"] = [job.job_id()]
        evs = job.result()[0].data.evs
        try:
            meta["usage_s"] = job.usage()
        except Exception:  # local fake jobs do not report usage
            meta["usage_s"] = None
        return EmbedResult(evs_to_bloch(evs, n), meta)

    raise ValueError(f"unknown tier {backend!r}: expected L0, L1, L2 or HW")


def _compile(qc, device, seed, meta):
    isa = generate_preset_pass_manager(optimization_level=3, backend=device, seed_transpiler=seed).run(qc)
    meta["backend_name"] = device.name
    meta["layout"] = [int(q) for q in isa.layout.final_index_layout()]
    meta["two_qubit_depth"] = int(isa.depth(lambda inst: inst.operation.num_qubits == 2))
    return isa


def _calibration(device, layout):
    """Calibration of the physical qubits and couplers this job uses, as reported at submission."""
    t = device.target
    qubits = {}
    for q in layout:
        props = device.qubit_properties(q)
        qubits[q] = {"t1": getattr(props, "t1", None), "t2": getattr(props, "t2", None),
                     "readout_error": t["measure"][(q,)].error if (q,) in t["measure"] else None}
    two_qubit = {f"{name}:{a}-{b}": p.error for name in t.operation_names if t.operation_from_name(name).num_qubits == 2
                 for (a, b), p in t[name].items() if a in layout and b in layout and p is not None}
    return {"last_update": str(device.properties().last_update_date), "qubits": qubits, "two_qubit_errors": two_qubit}
