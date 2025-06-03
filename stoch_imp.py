import sympy as sp
from sympy import NonSquareMatrixError, ShapeError
from itertools import product
import numpy as np


class CoeffArray:
    def __init__(self, n):
        self.mat = np.zeros((n, n, n))

    def __getitem__(self, item):
        return self.mat.__getitem__(item)

    def __setitem__(self, key, value):
        self.mat.__setitem__(key,value)

    def __str__(self):
        return sp.pretty(self.mat)


class Mat:
    """
    Top level class for all matrix like objects (Vectors, WMatrix, EqMatrix, ...)
    """
    def __init__(self, arr, zi=False):
        m = arr_to_mat(arr)

        self.mat = m                                                     # The Sympy Matrix object
        self.dim = self.mat.rows                                         # The dimension of the associated system
        self.iter = product(range(self.mat.rows), range(self.mat.cols))  # An iterator going over both rows and cols
        self.zero_index = zi                                             # Whether the system starts at 0

    def __str__(self):
        # This is a QoL improvement, ensures nice printing of matrix like objects
        return sp.pretty(self.mat)

    def __setitem__(self, key, value):
        if type(key) == int:
            self.mat[key] = value  # Vector element assignment
        else:
            self.mat[*key] = value  # Matrix element assignment

    def __iter__(self):
        return iter(self.mat)

    def __getitem__(self, item):
        if type(item) == int:
            return self.mat[item]  # Vector element assignment
        else:
            return self.mat[*item]  # Matrix element assignment

    def __sub__(self, other):
        return Mat(self.mat - other.mat)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Mat(self.mat + other.mat)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(self.__class__, type(other)))

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return Mat(self.mat * other.mat)
        else:
            return Mat(self.mat*other)

    def __rmul__(self, other):
        if isinstance(other, self.__class__):
            return Mat(other.mat * self.mat)
        else:
            return self.__mul__(other)


class TrMatrix(Mat):
    def __init__(self, arr, zi=False):
        if type(arr) == int:
            if arr == 1:
                raise ShapeError("Transfer matrix must be at least (2, 2) dimensional")
            else:
                arr = sp.zeros(arr, arr)
        super().__init__(arr, zi)
        _check_diag(self.mat, err=True)
        _check_rates(self.mat, err=True)

    def add_trans(self, i, j, r=1.0, ri=None, symm=True, simp=True):
        if not self.zero_index:
            i -= 1
            j -= 1

        if simp:
            r = sp.nsimplify(r, rational=True)
        self.mat[j, i] = r
        self.mat[i, i] -= r

        if symm:
            if not ri:
                ri = r ** (-1)
            elif simp:
                ri = sp.nsimplify(ri, rational=True)
            self.mat[i, j] = ri
            self.mat[j, j] -= ri
        if simp:
            sp.simplify(self.mat)

    def calc_diag(self):
        for col in range(self.dim):
            self.mat[col, col] = - sum([self.mat[row, col] for row in range(self.dim) if row != col])

    def check_diag(self):
        return _check_diag(self.mat)

    def check_rates(self, verbose=False):
        return _check_rates(self.mat, verbose=verbose)


class WMatrix(TrMatrix):
    def __init__(self, arr, ds=None, eq=0, zi=False):
        super().__init__(arr, zi)

        symb_list = list(self.mat.free_symbols)
        if ds is None:
            if not symb_list:
                raise AttributeError("No driving symbol found or given")
            elif len(symb_list) > 1:
                raise ValueError("Ambiguity in driving symbol, please provide a specific symbol")
            else:
                ds = symb_list[0]

        self.ds = ds
        self.weq = None
        self.w1 = None
        self.coeff = None
        self.eq = eq
        self.conds = None


    def calc_weq(self, eq=None, verbose=False):
        if eq is None:
            eq = self.eq
        self.weq = EqMatrix(self.mat.subs({self.ds: eq}), zi=self.zero_index)
        if verbose:
            print("Equilibrium matrix calculated")
        return self.weq

    def calc_w1(self, eq=None, verbose=False):
        if eq is None:
            eq = self.eq
        if self.ds not in list(self.mat.free_symbols):
            raise AttributeError("Driving symbol not found in matrix")

        self.w1 = Mat(sp.diff(self.mat, self.ds).subs({self.ds: eq}), zi=self.zero_index)
        if verbose:
            print("Driving matrix calculated")
        return self.w1

    def calc_coeff(self, force=False, verbose=True):
        return _calc_coeff(self, force, verbose=verbose)

    def calc_cond(self, curr=None, force=False, verbose=True):
        return _calc_cond(self, curr, force, verbose=verbose)


class EqMatrix(TrMatrix):
    def __init__(self, arr, zi=False):
        super().__init__(arr, zi)
        self.peq = None
        self.vals = None
        self.vecs = None

    def calc_peq(self, verbose=False):
        self.calc_eig(verbose=verbose)
        return self.peq

    def calc_eig(self, verbose=False):
        eig_syst = self.mat.eigenvects(error_when_incomplete=True)

        eig_syst.sort(key=lambda x: x[0], reverse=True)

        if eig_syst[0][0] != 0:
            raise ValueError("no eigenvalue 0 was found")
        elif eig_syst[0][1] != 1:
            raise ValueError("multiple steady states found")

        self.peq = Mat(eig_syst[0][-1][0]/sum(eig_syst[0][-1][0]))
        if verbose:
            print("Equilibrium distribution calculated")

        if not self.check_db():
            raise ValueError("detailed balance not fulfilled")

        vals = []
        vecs = []

        for space in eig_syst:
            val = space[0]
            if sp.im(val) > 10**-15:
                raise ValueError("Complex eigenvalue found: {}".format(val))
            for degen in range(space[1]):
                vec = space[-1][degen]
                if (self.mat*vec - val*vec).norm() > 10**-13:
                    raise ValueError("Incorrect computation of eigenvectors")

                vals.append(val)
                vecs.append(vec)

        vecs = gram_schmidt(vecs, self.peq)
        self.vals = vals
        self.vecs = vecs
        if verbose:
            print("Eigensystem calculated")
        return self.vals, self.vecs


    def check_db(self, tol=10**-15):
        if self.peq is None:
            self.calc_peq()

        db_mat = _calc_curr_like(self, self.peq)

        for el in self.iter:
            if db_mat[*el] > tol:
                return False
        return True


def deep_symp(e):
    e = sp.nsimplify(e, rational=True, full=True)
    e = sp.simplify(e)
    e = sp.nsimplify(e, rational=True, full=True)
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
        v /= sp.sqrt(inner(v, v, Peq))
        deep_symp(v)
        orthogonal.append(v)
    return orthogonal


def arr_to_mat(arr):
    m = sp.Matrix(arr)

    # Check if matrix is vector, transpose to col vec if necessary
    if (m.rows == 1) ^ (m.cols == 1):
        if m.cols > 1:
            return m.T
        else:
            return m
    # If not vector must be square matrix of dim at least (2, 2)
    elif not m.is_square:
        raise NonSquareMatrixError("Matrix object must be square or vector, is neither: ({}, {})".format(m.rows, m.cols))
    elif m.rows == 1:
        raise ShapeError("Matrix must be at least (2, 2) dimensional")
    else:
        return m


def _check_diag(m, err=False):
    for col in range(m.cols):
        if sum(m.col(col)) != 0:
            if err:
                raise ValueError("Column {} does not sum to 0 (sum = {})".format(col, sum(m.col(col))))
            else:
                return False
    return True


def _check_rates(m, err=False, verbose=False):
    for col in range(m.cols):
        for row in range(m.rows):
            try:
                if row != col and m[row, col] < 0:
                    if err:
                        raise ValueError(
                            "Transfer rate ({} -> {}) is negative ({})".format(col + 1, row + 1, m[row, col]))
                    else:
                        return False
            except TypeError as e:
                if verbose:
                    print("Positivity of transfer rate ({} -> {}) undetermined ({})".format(col + 1, row + 1,
                                                                                            m[row, col]))
    return True


def _calc_curr_like(m, p):
    m = m.mat
    curr1 = Mat([[deep_symp(m[row, col] * p[row]) for col in range(m.cols)] for row in range(m.rows)])
    curr2 = Mat([[deep_symp(m[row, col] * p[col]) for row in range(m.cols)] for col in range(m.rows)])
    return curr1 - curr2


def inner(v1, v2, Peq=None):
    n = len(v1)
    if Peq is None:
        Peq = [1 for _ in range(n)]

    return deep_symp(sum([v1[i]*v2[i]/Peq[i] for i in range(n)]))


def _calc_coeff(w, force=False, verbose=True):
    if (w.weq is None) or force:
        w.calc_weq(verbose=verbose)

    weq = w.weq

    if (weq.vals is None) or force:
        weq.calc_eig(verbose=verbose)
    peq = weq.peq
    vals= weq.vals
    vecs = weq.vecs

    if (w.w1 is None) or force:
        w.calc_w1(verbose=verbose)
    w1 = w.w1

    p1coeffs = [-inner(vecs[k], w1*peq, peq)/vals[k] for k in range(1, len(vals))]

    coeffs = CoeffArray(w.dim)
    for k in range(w.dim):
        if k == 0:
            coeffs[:, :, k] = _calc_curr_like(w1, peq).mat
        else:
            coeffs[:, :, k] = (p1coeffs[k-1] * _calc_curr_like(weq, vecs[k])).mat

    w.coeff = coeffs
    return coeffs


def _calc_cond(w, curr=None, force=False, verbose=True):
    if (w.coeff is None) or force:
        w.calc_coeff(verbose=verbose)

    if curr is None:
        curr = []
        for i in range(w.dim):
            for j in range(i + 1, w.dim):
                if w.weq[i, j] != 0 or w.weq[j, i] != 0 or w.w1[i, j] != 0 or w.w1[j, i] != 0:
                    curr.append([i, j])
    elif type(curr) == list:
        if type(curr[0]) != list:
            curr = [curr]
        for c in curr:
            try:
                if len(c) != 2:
                    raise TypeError("Currents should be given as lists of length 2, is length {}".format(len(c)))
            except TypeError:
                raise TypeError("Currents should be given as lists, is {}".format(type(c)))
            finally:
                pass
    else:
        raise TypeError("Current list should be given as list of lists, is {}".format(type(curr)))

    om_arr = np.logspace(-10, 2, 10000)
    res_lst = []
    for c in curr:
        c_i, c_j = c  # Extract transition
        res_arr = cond(w, om_arr, c_i, c_j, normal=True)  # calculate conductance
        res_lst.append([c, res_arr])
    w.conds = res_lst
    return res_lst

def cond(w, om, i, j, normal=False):
    """
    Calculate the conductance of the transition i --> j. This can be normalised using the conductance at zero driving.

    :param w: the W-matrix of which the conductivities are calculated
    :param om: Float/np array: The driving frequency or array of frequencies
    :param i: Int: The site of origin of the transition
    :param j: Int: The destination site of the transition
    :param normal: Bool: if True returns the conductance normalised using conductance at zero frequency

    :return: Float/np array: The conductance of the transition i --> j.
    """
    c = deep_symp(sum([w.coeff[i, j, k] * (1 if k == 0 else (w.weq.vals[k] / (1j * om - w.weq.vals[k]))) for k in range(len(w.coeff[0, 0, :]))]))

    if normal:
        n = deep_symp(sum([w.coeff[i, j, k] * (1 if k == 0 else -1) for k in range(len(w.coeff[0, 0, :]))]))
        return deep_symp(c / n)
    else:
        return c

if __name__ == '__main__':
    pass