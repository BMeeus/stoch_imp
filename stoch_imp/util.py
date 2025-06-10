from sympy import (nsimplify, ShapeError, simplify, sqrt, Matrix, Expr)


from .core_classes import Mat


def calc_curr_like(m: Mat, p: Mat | Matrix) -> Mat:
    """
    Calculate expressions of current-like form

    J_mn = W_mn P_n - W_nm P_m

    for matrix m and vector p.

    :param m: (Mat) The matrix to be used
    :param p:(Mat | Matrix) The vector to be used
    :return: The matrix containing the current-like values
    """

    if len(p) != m.dim:
        raise ShapeError("Matrix and vector do not have same shape: ({}, {}), ({})".format(m.dim, m.dim, p.dim))

    curr1 = Mat([[deep_symp(m[row, col] * p[row]) for col in range(m.dim)] for row in range(m.dim)])
    curr2 = Mat([[deep_symp(m[col, row] * p[col]) for col in range(m.dim)] for row in range(m.dim)])
    return curr1 - curr2


def deep_symp(e):
    """Simplify an arbitrary expression, both numerically and symbolically"""
    e = nsimplify(e, rational=True, full=True)
    e = simplify(e)
    return e


def gram_schmidt(v_arr: list[Matrix], Peq: Matrix | None = None) -> list[Matrix]:
    """
    Orthogonalise a set of vectors given as the columns of an array using Gram-Schmidt procedure.

    :param v_arr: array of vectors to orthogonalise
    :param Peq: The equilibrium vector used in the definition of the inner product
    :return: array of orthogonalised vectors
    """
    # Orthogonalised, To Be Returned
    orthogonal = []

    # At each step, take vector
    for i in range(len(v_arr)):
        v = v_arr[i]

        # Subtract off the "components" from current orthogonal set.
        for j in range(i):
            v -= inner(orthogonal[j], v, Peq) * orthogonal[j]
        # Normalization
        v /= sqrt(inner(v, v, Peq))
        deep_symp(v)
        orthogonal.append(v)
    return orthogonal


def inner(v1: Matrix, v2: Matrix, Peq: Matrix | None = None) -> float| Expr:
    """
    Calculate inner product between two vectors, weighted by the Equilibrium distribution.
    If Peq is not given, the standard inner product is returned.

    :param v1: (Mat | Matrix) First vector
    :param v2: (Mat | Matrix) Second vector
    :param Peq: (Mat | Matrix | None) Equilibrium distribution of the system
    :return: (float | Expr) The inner product of the vectors
    """
    if len(v1) != len(v2):
        raise ShapeError("Vectors are of different shapes: v1 is ({}), v2 ({})".format(len(v1), len(v2)))
    if Peq is None:
        Peq = [1 for _ in range(len(v1))]
    elif len(Peq) != len(v1) or len(Peq) != len(v2):
        raise ShapeError("Vectors are of different shapes: v1 is ({}), v2 is ({}), Peq is ({})".format(len(v1), len(v2), len(Peq)))

    return deep_symp(sum([v1[i] * v2[i] / Peq[i] for i in range(len(v1))]))
