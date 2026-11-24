import numpy as np
import pytest

from kernels.projected import projected_kernel
from quantum.embed import EmbedConfig, embed
from tests.fixtures.synthetic import synthetic_bloch, synthetic_curated, synthetic_features

X = synthetic_features(synthetic_curated())[:20]


def test_projected_kernel_is_psd():
    K = projected_kernel(*[synthetic_bloch(40, 5)] * 2, gamma=2.0)
    assert np.linalg.eigvalsh(K).min() > -1e-10


def test_fidelity_kernel_agrees_with_qiskit_machine_learning():
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    from kernels.fidelity import fidelity_kernel
    from quantum.embed import circuit_and_angles
    cfg = EmbedConfig(n_qubits=4, layers=2, s=0.7)
    qc, angles = circuit_and_angles(X, cfg)
    expected = FidelityStatevectorKernel(feature_map=qc).evaluate(angles, angles)
    assert np.allclose(fidelity_kernel(X, X, cfg), expected, atol=1e-8)


def test_fidelity_matches_bloch_vectors_for_one_qubit():
    from kernels.fidelity import fidelity_kernel
    cfg = EmbedConfig(n_qubits=1, layers=2, s=0.9)
    b = embed(X, cfg, "L0").B[:, 0]
    # pure single-qubit states: |<psi|phi>|^2 = (1 + b . b') / 2
    assert np.allclose(fidelity_kernel(X, X, cfg), (1 + b @ b.T) / 2, atol=1e-10)
