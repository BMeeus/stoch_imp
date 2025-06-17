from typing import Union, Any

from numpy import ndarray, ones, ones_like, zeros_like
from numpy import sqrt as np_sqrt
from numpy import sum as np_sum
from sympy import nsimplify, simplify, sqrt, Matrix, Expr, ShapeError

from .core_classes import Mat, ConstantMatrix


def calc_curr_like(m: Mat, p: Union[ConstantMatrix, Matrix, ndarray]) -> ConstantMatrix:
    """
    Calculate expressions of current-like form: J_mn = W_mn * P_n - W_nm * P_m.

    :param m: (Mat) The matrix to be used (must be square).
    :param p: (ConstantMatrix | Matrix | ndarray) The vector to be used (length must match m).
    :return: (ConstantMatrix) Matrix of current-like values.
    """
    return _sym_curr(m, p) if m.is_symbolic else _num_curr(m, p)


def deep_simp(e: Any) -> Any:
    """Simplify an arbitrary symbolic expression."""
    if hasattr(e, 'mat'):
        e.mat = simplify(nsimplify(e.mat, full=True))
        return e.mat
    return simplify(nsimplify(e, full=True))


def gram_schmidt(
        v_arr: Union[list[Matrix], ndarray, list[ConstantMatrix]],
        Peq: Union[ConstantMatrix, Matrix, ndarray, None] = None
) -> Union[list[Matrix], ndarray]:
    """
    Orthogonalize a set of vectors using Gram-Schmidt procedure.

    :param v_arr: (list[Matrix] | ndarray | list[ConstantMatrix]) List of symbolic vectors or NumPy array (columns = vectors).
    :param Peq: (ConstantMatrix | Matrix | ndarray | None) Equilibrium distribution for weighted inner product.
    :return: (list[Matrix] | ndarray) Orthogonalized vectors.
    """
    return _sym_gs(v_arr, Peq) if isinstance(v_arr, list) else _num_gs(v_arr, Peq)


def inner(
        v1: Union[Matrix, ndarray, ConstantMatrix],
        v2: Union[Matrix, ndarray, ConstantMatrix],
        Peq: Union[ConstantMatrix, Matrix, ndarray, None] = None
) -> Union[float, Expr]:
    """
    Compute inner product between two vectors, optionally weighted by Peq.

    :param v1: (Matrix | ndarray | ConstantMatrix) First vector.
    :param v2: (Matrix | ndarray | ConstantMatrix) Second vector.
    :param Peq: (ConstantMatrix | Matrix | ndarray | None) Optional equilibrium vector.
    :return: (float | Expr) Scalar inner product.
    """
    if len(v1) != len(v2):
        raise ShapeError(f"Vectors are of different shapes: v1 ({len(v1)}), v2 ({len(v2)})")

    if ((Peq is None and isinstance(v1, ndarray))
            or (isinstance(Peq, ConstantMatrix) and not Peq.is_symbolic)
            or (isinstance(Peq, ndarray))):
        return _num_inner(v1, v2, Peq)
    else:
        return _sym_inner(v1, v2, Peq)


def _sym_inner(v1: Matrix, v2: Matrix, Peq: Union[Matrix, None]) -> Expr:
    """
    Symbolic inner product of two vectors, weighted by Peq.

    :param v1: (Matrix) First vector.
    :param v2: (Matrix) Second vector.
    :param Peq: (Matrix | None) Optional weighting vector.
    :return: (Expr) Symbolic inner product.
    """
    if Peq is None:
        Peq = Matrix.ones(len(v1), 1)
    elif len(Peq) != len(v1):
        raise ShapeError(f"Inconsistent shapes: v1 ({len(v1)}), Peq ({len(Peq)})")

    return deep_simp(sum(v1[i] * v2[i] / Peq[i] for i in range(len(v1))))


def _num_inner(v1: Union[Mat, ndarray], v2: Union[Mat, ndarray], Peq: Union[Mat, ndarray, None]) -> float:
    """
    Numerical inner product of two vectors, weighted by Peq.

    :param v1: (Mat | ndarray) First vector.
    :param v2: (Mat | ndarray) Second vector.
    :param Peq: (Mat | ndarray | None) Optional weighting vector.
    :return: (float) Numerical inner product.
    """
    if Peq is None:
        Peq = ones_like(v1)

    if len(Peq) != len(v1):
        raise ShapeError(f"Inconsistent shapes: v1 ({len(v1)}), Peq ({len(Peq)})")

    if hasattr(v1, 'nmat'):
        v1 = v1.nmat
    if hasattr(v2, 'nmat'):
        v2 = v2.nmat
    if hasattr(Peq, "nmat"):
        Peq = Peq.nmat
    return np_sum(v1 * v2 / Peq)


def _sym_gs(v_arr: list[Matrix], Peq: Union[Matrix, None]) -> list[Matrix]:
    """
    Perform symbolic Gram-Schmidt orthogonalization.

    :param v_arr: (list[Matrix]) List of symbolic vectors.
    :param Peq: (Matrix | None) Optional weighting vector.
    :return: (list[Matrix]) List of orthogonalized vectors.
    """
    orthogonal: list[Matrix] = []

    for i, v in enumerate(v_arr):
        for j in range(i):
            proj = inner(orthogonal[j], v, Peq)
            v -= proj * orthogonal[j]
        v /= sqrt(inner(v, v, Peq))
        orthogonal.append(deep_simp(v))
    return orthogonal


def _num_gs(v_arr: ndarray, Peq: Union[ndarray, None]) -> ndarray:
    """
    Perform numerical Gram-Schmidt orthogonalization.

    :param v_arr: (ndarray) Array with column vectors.
    :param Peq: (ndarray | None) Optional weighting vector.
    :return: (ndarray) Array of orthogonalized vectors.
    """
    n = v_arr.shape[1]
    orthogonal = zeros_like(v_arr)

    if Peq is None:
        Peq = ones(v_arr.shape[0])

    for i in range(n):
        v = v_arr[:, i].copy()
        for j in range(i):
            proj = np_sum(inner(orthogonal[:, j], v, Peq))
            v -= proj * orthogonal[:, j]
        v /= np_sqrt(inner(v, v, Peq))
        orthogonal[:, i] = v
    return orthogonal


def _sym_curr(m: Mat, p: Union[ConstantMatrix, Matrix]) -> ConstantMatrix:
    """
    Compute symbolic current-like matrix from m and p.

    :param m: (Mat) Symbolic square matrix.
    :param p: (ConstantMatrix | Matrix) Vector for weighting.
    :return: (ConstantMatrix) Resulting current matrix.
    """
    if len(p) != m.dim:
        raise ShapeError(f"Inconsistent dimensions: m ({m.dim}), p ({len(p)})")

    curr1 = ConstantMatrix([[deep_simp(m[row, col] * p[col]) for col in range(m.dim)] for row in range(m.dim)])
    curr2 = ConstantMatrix([[deep_simp(m[col, row] * p[row]) for col in range(m.dim)] for row in range(m.dim)])
    return curr1 - curr2


def _num_curr(m: Mat, p: Union[ndarray, ConstantMatrix]) -> ConstantMatrix:
    """
    Compute numerical current-like matrix from m and p.

    :param m: (Mat) Numeric matrix.
    :param p: (ndarray | ConstantMatrix) Vector for weighting.
    :return: (ConstantMatrix) Resulting current matrix.
    """
    if len(p) != m.dim:
        raise ShapeError(f"Inconsistent dimensions: m ({m.dim}), p ({len(p)})")

    m = m.nmat

    if hasattr(p, "nmat"):
        p = p.nmat
    return ConstantMatrix((m.T * p).T - m.T * p)
