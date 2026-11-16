"""
Kernel-target alignment per Q-Spy.

Funzione principale:
- kernel_target_alignment(K, y): misura quanto la struttura di similarità
  del kernel K è coerente con le etichette di classe y.

Metodo:
- costruisce il kernel target T[i,j] = 1 se y[i] == y[j], altrimenti 0;
- centra K e T;
- calcola la centered kernel-target alignment:
      <K_c, T_c>_F / (||K_c||_F * ||T_c||_F)

Il risultato è un numero adimensionale: valori più alti indicano maggiore
allineamento tra kernel e classi. Il modulo è pensato per classificazione
(glitch / non-glitch o più classi), non per target continui di regressione.

K deve essere una matrice kernel quadrata, finita e simmetrica; y deve
contenere un'etichetta per ogni riga/colonna di K.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_kernel_matrix(K: ArrayLike) -> np.ndarray:
    """Converte K in float64 e controlla che sia una matrice kernel valida."""
    matrix = np.asarray(K, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("K deve essere una matrice quadrata.")
    if matrix.shape[0] < 2:
        raise ValueError("K deve contenere almeno due osservazioni.")
    if not np.isfinite(matrix).all():
        raise ValueError("K contiene NaN o valori infiniti.")
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
        raise ValueError("K deve essere simmetrica.")

    return (matrix + matrix.T) * 0.5


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Centra una matrice kernel usando H = I - 11^T / N."""
    row_mean = np.mean(K, axis=1, keepdims=True)
    column_mean = np.mean(K, axis=0, keepdims=True)
    overall_mean = float(np.mean(K))
    return K - row_mean - column_mean + overall_mean


def kernel_target_alignment(K: ArrayLike, y: ArrayLike) -> float:
    """
    Calcola la centered kernel-target alignment (CKTA).

    Il kernel target è costruito dalle etichette: T[i,j] vale 1 quando
    y[i] e y[j] appartengono alla stessa classe, 0 altrimenti.

    Args:
        K: matrice kernel N x N.
        y: vettore di N etichette di classe.

    Returns:
        Alignment normalizzato. In condizioni standard è compreso tra
        0 e 1; valori maggiori indicano una migliore coerenza tra K e y.

    Raises:
        ValueError: se dimensioni/valori non sono validi, se y contiene
        meno di due classi o se uno dei kernel centrati ha norma nulla.
    """
    matrix = _as_kernel_matrix(K)
    labels = np.asarray(y)

    if labels.ndim != 1:
        raise ValueError("y deve essere un vettore monodimensionale.")
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError("y deve contenere un'etichetta per ogni osservazione.")
    if np.issubdtype(labels.dtype, np.number) and not np.isfinite(labels).all():
        raise ValueError("y contiene NaN o valori infiniti.")

    if labels.dtype == object:
        if any(label is None for label in labels):
            raise ValueError("y contiene etichette mancanti.")
    
    classes, inverse = np.unique(labels, return_inverse=True)
    if classes.size < 2:
        raise ValueError("y deve contenere almeno due classi.")

    # T[i,j] = 1 se le due osservazioni hanno la stessa etichetta.
    target = (inverse[:, None] == inverse[None, :]).astype(np.float64)

    centered_kernel = _center_kernel(matrix)
    centered_target = _center_kernel(target)

    numerator = float(np.sum(centered_kernel * centered_target))
    kernel_norm = float(np.linalg.norm(centered_kernel, ord="fro"))
    target_norm = float(np.linalg.norm(centered_target, ord="fro"))
    if kernel_norm == 0.0 or target_norm == 0.0:
        raise ValueError(
            "Alignment non definito: il kernel centrato ha norma nulla."
        )

    normalized_kernel = centered_kernel / kernel_norm
    normalized_target = centered_target / target_norm

    return float(np.sum(normalized_kernel * normalized_target))


__all__ = ["kernel_target_alignment"]
