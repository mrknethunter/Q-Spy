"""Funzioni interne per la validazione degli input delle diagnostiche."""

import numpy as np


def validate_kernel_matrix(
    K: np.ndarray,
    *,
    min_size: int = 1,
) -> np.ndarray:
    """Valida una matrice kernel e restituisce un array NumPy float64.

    Controlla che K sia reale, numerica, finita, quadrata, simmetrica
    e che abbia almeno ``min_size`` righe e colonne.

    La semidefinitezza positiva non viene verificata qui: è un controllo
    specifico di alcune diagnostiche, come ``effective_rank``.
    """
    if (
        isinstance(min_size, bool)
        or not isinstance(min_size, (int, np.integer))
        or min_size < 1
    ):
        raise ValueError("min_size deve essere un intero positivo.")

    try:
        raw = np.asarray(K)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("K deve essere convertibile in un array.") from exc

    if np.iscomplexobj(raw):
        raise ValueError("K deve contenere valori reali.")

    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "K deve contenere esclusivamente valori numerici reali."
        ) from exc

    if matrix.ndim != 2:
        raise ValueError("K deve essere una matrice bidimensionale.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("K deve essere una matrice quadrata.")

    if matrix.shape[0] < min_size:
        raise ValueError(
            f"K deve contenere almeno {min_size} elementi per dimensione."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("K contiene valori NaN o infiniti.")

    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
        raise ValueError("K deve essere simmetrica.")

    return matrix
