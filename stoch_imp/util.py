from typing import Union, Any

from numpy import array, ndarray, ones, ones_like, zeros_like
from numpy import sqrt as np_sqrt
from numpy import sum as np_sum
from sympy import nsimplify, simplify, sqrt, Matrix, Expr, ShapeError

from .core_classes import Mat, ConstantMatrix


def calc_curr_like(m: Union[Mat, Matrix, ndarray], p: Union[ConstantMatrix, Matrix, ndarray]) -> ConstantMatrix:
    """
    Calculate expressions of current-like form: :math:`J_{mn} = W_{mn} P_n - W_{nm} P_m`.

    :param m: The matrix to be used (must be square).
    :type m: Mat
    :param p: The vector to be used (length must match m).
    :type p: ConstantMatrix | Matrix | ndarray
    :return: Resulting matrix :math:`J_{mn}`.
    :rtype: ConstantMatrix
    """
    if hasattr(m, 'is_symbolic'):
        return _sym_curr(m, p) if m.is_symbolic else _num_curr(m, p)
    elif hasattr(m, 'free_symbols'):
        return _sym_curr(m, p) if m.free_symbols else _num_curr(m, p)
    else:
        return _num_curr(m, p)

def deep_simp(e: Any, **flags) -> Any:
    """Simplify an arbitrary symbolic expression."""
    if hasattr(e, 'mat'):
        e.mat = simplify(nsimplify(e.mat, full=True, **flags))
        return e.mat
    return simplify(nsimplify(e, full=True, **flags))


def gram_schmidt(
        v_arr: Union[list[Matrix], ndarray, list[ConstantMatrix]],
        Peq: Union[ConstantMatrix, Matrix, ndarray, None] = None
) -> Union[list[Matrix], ndarray]:
    """
    Orthogonalize a set of vectors using Gram-Schmidt procedure.

    :param v_arr: List of symbolic vectors or NumPy array (columns = vectors).
    :type v_arr: list[Matrix] | ndarray | list[ConstantMatrix]
    :param Peq: Equilibrium distribution for weighted inner product.
    :type Peq: ConstantMatrix | Matrix | ndarray | None
    :return: Orthogonalized vectors.
    :rtype: list[Matrix] | ndarray
    """
    return _sym_gs(v_arr, Peq) if isinstance(v_arr, list) else _num_gs(v_arr, Peq)


def inner(
        v1: Union[Matrix, ndarray, ConstantMatrix],
        v2: Union[Matrix, ndarray, ConstantMatrix],
        Peq: Union[ConstantMatrix, Matrix, ndarray, None] = None
) -> Union[float, Expr]:
    """
    Compute inner product between two vectors, optionally weighted by Peq.

    :param v1: First vector.
    :type v1: Matrix | ndarray | ConstantMatrix
    :param v2: Second vector.
    :type v2: Matrix | ndarray | ConstantMatrix
    :param Peq: Optional equilibrium vector.
    :type Peq: ConstantMatrix | Matrix | ndarray | None
    :return: Scalar inner product.
    :rtype: float | Expr
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

    :param v1: First vector.
    :type v1: Matrix
    :param v2: Second vector.
    :type v2: Matrix
    :param Peq: Optional weighting vector.
    :type Peq: Matrix | None
    :return: Symbolic inner product.
    :rtype: Expr
    """
    if Peq is None:
        Peq = Matrix.ones(len(v1), 1)
    elif len(Peq) != len(v1):
        raise ShapeError(f"Inconsistent shapes: v1 ({len(v1)}), Peq ({len(Peq)})")

    return deep_simp(sum(v1[i] * v2[i] / Peq[i] for i in range(len(v1))))


def _num_inner(v1: Union[Mat, ndarray], v2: Union[Mat, ndarray], Peq: Union[Mat, ndarray, None]) -> float:
    """
    Numerical inner product of two vectors, weighted by Peq.

    :param v1: First vector.
    :type v1: Mat | ndarray
    :param v2: Second vector.
    :type v2: Mat | ndarray
    :param Peq: Optional weighting vector.
    :type Peq: Mat | ndarray | None
    :return: Numerical inner product.
    :rtype: float
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

    :param v_arr: List of symbolic vectors.
    :type v_arr: list[Matrix]
    :param Peq: Optional weighting vector.
    :type Peq: Matrix | None
    :return: List of orthogonalized vectors.
    :rtype: list[Matrix]
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

    :param v_arr: Array with column vectors.
    :type v_arr: ndarray
    :param Peq: Optional weighting vector.
    :type Peq: ndarray | None
    :return: Array of orthogonalized vectors.
    :rtype: ndarray
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

    :param m: Symbolic square matrix.
    :type m: Mat
    :param p: Vector for weighting.
    :type p: ConstantMatrix | Matrix
    :return: Resulting current matrix.
    :rtype: ConstantMatrix
    """
    if len(p) != m.dim:
        raise ShapeError(f"Inconsistent dimensions: m ({m.dim}), p ({len(p)})")

    curr1 = ConstantMatrix([[deep_simp(m[row, col] * p[col]) for col in range(m.dim)] for row in range(m.dim)])
    curr2 = ConstantMatrix([[deep_simp(m[col, row] * p[row]) for col in range(m.dim)] for row in range(m.dim)])
    return curr1 - curr2


def _num_curr(m: Mat, p: Union[ndarray, ConstantMatrix]) -> ConstantMatrix:
    """
    Compute numerical current-like matrix from m and p.

    :param m: Numeric matrix.
    :type m: Mat
    :param p: Vector for weighting.
    :type p: ndarray | ConstantMatrix
    :return: Resulting current matrix.
    :rtype: ConstantMatrix
    """
    if len(p) != m.dim:
        raise ShapeError(f"Inconsistent dimensions: m ({m.dim}), p ({len(p)})")

    try:
        m = m.nmat
    except AttributeError:
        m = array(m)
    if hasattr(p, "nmat"):
        p = p.nmat
    return ConstantMatrix((m.T * p).T - m.T * p)

