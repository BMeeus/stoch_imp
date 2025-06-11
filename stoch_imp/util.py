from typing import Union, Any

import numpy as np
from sympy import nsimplify, simplify, sqrt, Matrix, Expr, ShapeError

from .core_classes import Mat


def calc_curr_like(m: Mat, p: Union[Mat, Matrix, np.ndarray]) -> Mat:
    """
    Calculate expressions of current-like form: J_mn = W_mn * P_n - W_nm * P_m.

    :param m: (Mat) The matrix to be used (must be square).
    :param p: (Mat | Matrix) The vector to be used (length must match m).
    :return: Matrix of current-like values.
    """
    return _sym_curr(m, p) if m.is_symbolic else _num_curr(m, p)


def deep_simp(e: Any) -> Any:
    """Simplify an arbitrary symbolic expression."""
    return simplify(nsimplify(e, full=True))


def gram_schmidt(
        v_arr: Union[list[Matrix], np.ndarray, list[Mat]],
        Peq: Union[Mat, Matrix, np.ndarray, None] = None
        ) -> Union[list[Matrix], np.ndarray]:
    """
    Orthogonalize a set of vectors using Gram-Schmidt procedure.

    :param v_arr: List of symbolic vectors or NumPy array (columns = vectors).
    :param Peq: Equilibrium distribution for weighted inner product.
    :return: Orthogonalized vectors.
    """
    return _sym_gs(v_arr, Peq) if isinstance(v_arr, list) else _num_gs(v_arr, Peq)


def inner(
        v1: Union[Matrix, np.ndarray, Mat],
        v2: Union[Matrix, np.ndarray, Mat],
        Peq: Union[Mat, Matrix, np.ndarray, None] = None
        ) -> Union[float, Expr]:
    """
    Compute inner product between two vectors, optionally weighted by Peq.

    :param v1: First vector.
    :param v2: Second vector.
    :param Peq: Optional equilibrium vector.
    :return: Scalar inner product.
    """
    if len(v1) != len(v2):
        raise ShapeError(f"Vectors are of different shapes: v1 ({len(v1)}), v2 ({len(v2)})")

    if (Peq is None and isinstance(v1, np.ndarray)) or (isinstance(Peq, Mat) and Peq.is_numeric):
        return _num_inner(v1, v2, Peq)
    else:
        return _sym_inner(v1, v2, Peq)


def _sym_inner(v1: Matrix, v2: Matrix, Peq: Union[Matrix, None]) -> Expr:
    if Peq is None:
        Peq = Matrix.ones(len(v1), 1)
    elif len(Peq) != len(v1):
        raise ShapeError(f"Inconsistent shapes: v1 ({len(v1)}), Peq ({len(Peq)})")

    return deep_simp(sum(v1[i] * v2[i] / Peq[i] for i in range(len(v1))))


def _num_inner(v1: np.ndarray, v2: np.ndarray, Peq: Union[np.ndarray, None]) -> float:
    if Peq is None:
        Peq = np.ones_like(v1)

    if len(Peq) != len(v1):
        raise ShapeError(f"Inconsistent shapes: v1 ({len(v1)}), Peq ({len(Peq)})")

    return np.sum(v1 * v2 / Peq)


def _sym_gs(v_arr: list[Matrix], Peq: Union[Matrix, None]) -> list[Matrix]:
    orthogonal: list[Matrix] = []

    for i, v in enumerate(v_arr):
        for j in range(i):
            proj = inner(orthogonal[j], v, Peq)
            v -= proj * orthogonal[j]
        v /= sqrt(inner(v, v, Peq))
        orthogonal.append(deep_simp(v))
    return orthogonal


def _num_gs(v_arr: np.ndarray, Peq: Union[np.ndarray, None]) -> np.ndarray:
    n = v_arr.shape[1]
    orthogonal = np.zeros_like(v_arr)

    if Peq is None:
        Peq = np.ones(v_arr.shape[0])

    for i in range(n):
        v = v_arr[:, i].copy()
        for j in range(i):
            proj = np.sum(orthogonal[:, j] * v / Peq)
            v -= proj * orthogonal[:, j]
        v /= np.sqrt(np.sum(v * v / Peq))
        orthogonal[:, i] = v
    return orthogonal


def _sym_curr(m: Mat, p: Union[Mat, Matrix]) -> Mat:
    if len(p) != m.dim:
        raise ShapeError(f"Inconsistent dimensions: m ({m.dim}), p ({len(p)})")

    curr1 = Mat([[deep_simp(m[row, col] * p[row]) for col in range(m.dim)] for row in range(m.dim)])
    curr2 = Mat([[deep_simp(m[col, row] * p[col]) for col in range(m.dim)] for row in range(m.dim)])
    return curr1 - curr2


def _num_curr(m: Mat, p: np.ndarray) -> Mat:
    if len(p) != m.dim:
        raise ShapeError(f"Inconsistent dimensions: m ({m.dim}), p ({len(p)})")

    return m * p - (m * p).T
