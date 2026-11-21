"""Diagnostiche di allineamento tra kernel e target."""

import numpy as np

from ._validation import validate_kernel_matrix


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Calcola il kernel-target alignment non centrato binario.

    Args:
        K: matrice kernel simmetrica N x N.
        y: vettore di N etichette codificate esclusivamente come -1 e +1.

    Returns:
        Coseno tra K e il kernel target T = outer(y, y), compreso tra -1 e 1.

    Note:
        Questa implementazione è per classificazione binaria e usa la
        definizione non centrata dell'allineamento.
    """
    K = validate_kernel_matrix(K)

    try:
        raw_y = np.asarray(y)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("y deve essere un array di etichette valido.") from exc

    if np.iscomplexobj(raw_y):
        raise ValueError("Le etichette devono essere reali.")

    try:
        y = np.asarray(raw_y, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Le etichette devono essere numeriche.") from exc

    if y.ndim != 1 or y.size != K.shape[0]:
        raise ValueError("y deve contenere un'etichetta per ogni punto.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contiene valori NaN o infiniti.")

    if not np.all(np.isin(y, [-1.0, 1.0])):
        raise ValueError("Le etichette devono essere codificate come -1 e +1.")

    target = np.outer(y, y)
    numerator = float(np.sum(K * target))
    denominator = float(
        np.linalg.norm(K, ord="fro") * np.linalg.norm(target, ord="fro")
    )

    if denominator == 0.0:
        raise ValueError("Alignment non definito: K è una matrice nulla.")

    # Evita piccoli superamenti di +/-1 dovuti all'aritmetica floating point.
    return float(np.clip(numerator / denominator, -1.0, 1.0))
