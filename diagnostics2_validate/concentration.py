"""Diagnostiche di concentrazione e spettro delle matrici kernel."""

import numpy as np

from ._validation import validate_kernel_matrix


def off_diagonal_variance(K: np.ndarray) -> float:
    """Calcola la varianza degli elementi fuori diagonale di K.

    Per una matrice simmetrica usa il solo triangolo superiore, così ogni
    coppia di punti viene conteggiata una volta. La varianza è quella della
    popolazione (ddof=0).
    """
    K = validate_kernel_matrix(K, min_size=2)
    off_diagonal = K[np.triu_indices(K.shape[0], k=1)]
    return float(np.var(off_diagonal, ddof=0))


def kernel_spectrum(K: np.ndarray) -> np.ndarray:
    """Restituisce gli autovalori di K in ordine decrescente."""
    K = validate_kernel_matrix(K)
    eigenvalues = np.linalg.eigvalsh(K)
    return np.sort(eigenvalues)[::-1]


def effective_rank(K: np.ndarray) -> float:
    """Calcola l'effective rank tramite l'entropia dello spettro.

    La matrice deve essere semidefinita positiva. Piccoli autovalori
    negativi compatibili con errori numerici vengono portati a zero.
    """
    eigenvalues = kernel_spectrum(K)

    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = 1e-10 * scale

    if np.any(eigenvalues < -tolerance):
        raise ValueError(
            "K non è semidefinita positiva: effective rank non definito."
        )

    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(eigenvalues))

    if total <= 0.0:
        raise ValueError(
            "La somma degli autovalori è nulla: effective rank non definito."
        )

    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = -np.sum(probabilities * np.log(probabilities))

    return float(np.exp(entropy))
