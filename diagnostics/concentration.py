"""
Diagnostica della concentrazione dei kernel per Q-Spy.

Funzioni:
- off_diagonal_values(K): estrae una sola volta ogni elemento fuori diagonale.
- off_diagonal_variance(K): misura la variabilità dei valori fuori diagonale.
- eigen_spectrum(K): restituisce gli autovalori in ordine decrescente.
- effective_rank(K): misura quante direzioni spettrali sono usate effettivamente.

Convenzioni:
- K è una matrice kernel quadrata, finita e simmetrica.
- La varianza usa la popolazione (ddof=0) sui soli elementi del triangolo
  superiore: ogni coppia di osservazioni viene quindi contata una volta.
- L'effective rank è exp(entropia di Shannon degli autovalori normalizzati).
  Gli autovalori negativi compatibili con piccoli errori numerici sono
  trattati come zero; autovalori negativi significativi generano un errore.

Il modulo usa soltanto NumPy e non costruisce kernel: analizza matrici K
prodotte dagli altri componenti di Q-Spy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_kernel_matrix(K: ArrayLike) -> FloatArray:
    """Converte K in float64 e controlla forma, valori e simmetria."""
    matrix = np.asarray(K, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("K deve essere una matrice quadrata.")
    if matrix.shape[0] == 0:
        raise ValueError("K non può essere vuota.")
    if not np.isfinite(matrix).all():
        raise ValueError("K contiene NaN o valori infiniti.")
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
        raise ValueError("K deve essere simmetrica.")

    # Rimuove piccole differenze numeriche tra K[i,j] e K[j,i].
    return (matrix + matrix.T) * 0.5


def off_diagonal_values(K: ArrayLike) -> FloatArray:
    """
    Restituisce gli elementi fuori diagonale di K, senza duplicati.

    Per una matrice N x N restituisce N*(N-1)/2 valori, prendendo
    esclusivamente il triangolo superiore.
    """
    matrix = _as_kernel_matrix(K)
    if matrix.shape[0] < 2:
        raise ValueError("Servono almeno due osservazioni.")

    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols].copy()


def off_diagonal_variance(K: ArrayLike) -> float:
    """
    Calcola la varianza (ddof=0) degli elementi fuori diagonale.

    Valori vicini a zero indicano che le similarità tra coppie distinte
    sono quasi tutte uguali: è un possibile segnale di concentrazione.
    """
    values = off_diagonal_values(K)
    return float(np.var(values, ddof=0))


def eigen_spectrum(K: ArrayLike) -> FloatArray:
    """
    Calcola gli autovalori di K in ordine decrescente.

    Usa eigvalsh, adatto a matrici reali simmetriche. Non tronca gli
    autovalori: eventuali piccoli valori negativi numerici restano visibili.
    """
    matrix = _as_kernel_matrix(K)
    return np.linalg.eigvalsh(matrix)[::-1].copy()


def effective_rank(K: ArrayLike, *, tol: float = 1e-10) -> float:
    """
    Calcola l'effective rank spettrale: exp(-sum(p_i * log(p_i))).

    p_i sono gli autovalori non negativi normalizzati affinché sommino a 1.
    Il risultato è compreso tra 1 e il rango numerico della matrice
    (e non supera N). Se K è nulla, restituisce 0.

    tol è la tolleranza relativa usata per distinguere piccoli autovalori
    negativi dovuti ad arrotondamenti da una matrice non-PSD.
    """
    if tol < 0:
        raise ValueError("tol deve essere non negativa.")

    eigenvalues = eigen_spectrum(K)
    scale = float(np.max(np.abs(eigenvalues)))
    if scale == 0.0:
        return 0.0
    if float(np.min(eigenvalues)) < -tol * scale:
        raise ValueError(
            "K non è positiva semidefinita: "
            "effective rank richiede una matrice kernel PSD."
        )

    positive = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(positive))
    if total <= tol * scale:
        return 0.0

    probabilities = positive[positive > 0.0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


__all__ = [
    "off_diagonal_values",
    "off_diagonal_variance",
    "eigen_spectrum",
    "effective_rank",
]
