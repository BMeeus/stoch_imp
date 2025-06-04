from .core_classes import (Mat)

from sympy import (nsimplify, ShapeError, simplify, sqrt)


def calc_curr_like(m: Mat, p) -> Mat:
    """
    Calculate expressions of current-like form

    J_mn = W_mn P_n - W_nm P_m

    for matrix m and vector p
    :param m: The matrix to be used
    :param p: The vector to be used
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


def gram_schmidt(v_arr, Peq=None):
    """
    Orthogonalise a set of vectors given as the columns of an array using Gram-Schmidt procedure.

    :param v_arr: array of vectors to orthogonalise
    :param Peq: The equilibrium vector used in the definition of the inner product
    :return: array of orthogonalised vectors
    """
    n = len(v_arr[0])
    if Peq is None:
        Peq = [1 for _ in range(n)]

    if type(Peq) == Mat:
        Peq = Peq.mat

    # Orthogonalized, To Be Returned
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


def inner(v1, v2, Peq=None):
    n = len(v1)
    if Peq is None:
        Peq = [1 for _ in range(n)]

    return deep_symp(sum([v1[i] * v2[i] / Peq[i] for i in range(n)]))
