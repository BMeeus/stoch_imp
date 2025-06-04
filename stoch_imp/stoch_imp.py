import sympy as sp
from sympy import ShapeError, NonSquareMatrixError
import numpy as np
from itertools import product


class CoeffArray:
    """
    Class handling the array containing the coefficients.
    """

    def __init__(self, n):
        """
        Initialising an instance of CoeffArray

        :param n: size of the system
        """
        self.mat = np.zeros((n, n, n), dtype=object)

    # Let numpy handle item getting and setting
    def __getitem__(self, item):
        return self.mat.__getitem__(item)

    def __setitem__(self, key, value):
        self.mat.__setitem__(key, value)

    def __str__(self):
        return sp.pretty(self.mat)


class Mat:
    """
    Top level class for all matrix like objects (Vectors, WMatrix, EqMatrix, ...)
    """

    def __init__(self, arr, zi=False):
        """
        Initialises a Mat instance

        :param arr (Array like): The Array with which the matrix is set.
        :param zi (Bool): Whether the system is zero-indexed. Can be useful in systems where there can be no particles.
        """
        m = arr_to_mat(arr)

        self.mat = m  # The Sympy Matrix object
        self.dim = self.mat.rows  # The dimension of the associated system
        self.iter = product(range(self.mat.rows), range(self.mat.cols))  # An iterator going over both rows and cols
        self.zero_index = zi  # Whether the system starts at 0

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
            return Mat(self.mat * other)

    def __rmul__(self, other):
        if isinstance(other, self.__class__):
            return Mat(other.mat * self.mat)
        else:
            return self.__mul__(other)

    def __len__(self):
        return self.mat.__len__()


class TrMatrix(Mat):
    """
    Class handling all transition matrices. These matrices should sum ot zero along columns and have positive
    off-diagonal elements.
    """

    def __init__(self, arr, zi=False):
        """
        Initialise a new Transfer matrix

        :param arr (Array like or Int): The Array with which the matrix is set. If arr is an integer, an empty
            (arr,arr) Sympy Matrix is used
        :param zi (Bool): Whether the system is zero-indexed. Can be useful in systems where there can be no particles.
        """
        if type(arr) == int:
            if arr <= 1:
                raise ShapeError("Transfer matrix must be at least (2, 2) dimensional")
            else:
                arr = sp.zeros(arr, arr)

        super().__init__(arr, zi)
        check_diag(self.mat, err=True)
        check_rates(self.mat, err=True)
        return

    def add_trans(self, i: int, j: int, r: float = 1.0, ri: float = None, symm: bool = True, simp: bool = True) -> None:
        """
        Add a transition i --> j, with rate r.

        If symm is True, a symmetric transition j --> i is added with rate 1/r. Optionally, ri can be
        used to specify a rate for the symmetric transition.

        :param i: (Int) The starting site of the transition
        :param j: (Int) The target site of the transition
        :param r: (Float) The transfer rate with which the transition should be set. Default is 1.0
        :param ri: (Float) The transfer rate used for the inverse transition. Default is 1/r
        :param symm: (Bool) If True, also adds a symmetric transition j --> i.
        :param simp: (Bool) Whether to simplify the rates before setting.
        :return:
        """

        if i == j:
            raise ValueError("Start and target state are equal, must be different")

        if not self.zero_index:
            i -= 1
            j -= 1

        if simp:
            r = sp.nsimplify(r, rational=True)

        try:
            if r < 0:
                raise ValueError(f"Rate should be positive, is {r}")
        except TypeError:
            pass

        self.mat[j, i] = r  # Set the matrix element
        self.mat[i, i] -= r  # Update diagonal element

        if symm or (ri is not None):
            if not ri:
                ri = r ** (-1)
            elif simp:
                ri = sp.nsimplify(ri, rational=True)
            try:
                if ri < 0:
                    raise ValueError(f"Inverse rate should be positive, is {ri}")
            except TypeError:
                pass
            self.mat[i, j] = ri
            self.mat[j, j] -= ri
        if simp:
            sp.simplify(self.mat)
        return

    def calc_diag(self) -> None:
        """Force calculation of the diagonal elements"""
        for col in range(self.dim):
            self.mat[col, col] = - sum([self.mat[row, col] for row in range(self.dim) if row != col])

    def check_diag(self) -> bool:
        """Check that columns sum to 0"""
        return check_diag(self.mat)

    def check_rates(self, verbose: bool = False) -> bool:
        """Check if off-diagonal elements are positive"""
        return check_rates(self.mat, verbose=verbose)


class EqMatrix(TrMatrix):
    def __init__(self, arr, zi: bool = False):
        super().__init__(arr, zi)
        self.peq = None
        self.vals = None
        self.vecs = None

    def calc_peq(self, verbose: bool = False) -> Mat:
        self.calc_eig(verbose=verbose)
        return self.peq

    def calc_eig(self, verbose: bool = False) -> tuple[list, list]:
        eig_syst = self.mat.eigenvects(error_when_incomplete=True)

        eig_syst.sort(key=lambda x: x[0], reverse=True)

        if eig_syst[0][0] != 0:
            raise ValueError("no eigenvalue 0 was found")
        elif eig_syst[0][1] != 1:
            raise ValueError("multiple steady states found")

        self.peq = Mat(eig_syst[0][-1][0] / sum(eig_syst[0][-1][0]))
        if verbose:
            print("Equilibrium distribution calculated")

        if not self.check_db():
            raise ValueError("detailed balance not fulfilled")

        vals = []
        vecs = []

        for space in eig_syst:
            val = space[0]
            if sp.im(val) > 10 ** -15:
                raise ValueError("Complex eigenvalue found: {}".format(val))
            for degen in range(space[1]):
                vec = space[-1][degen]
                if (self.mat * vec - val * vec).norm() > 10 ** -13:
                    raise ValueError("Incorrect computation of eigenvectors")

                vals.append(val)
                vecs.append(vec)

        vecs = gram_schmidt(vecs, self.peq)
        self.vals = vals
        self.vecs = vecs
        if verbose:
            print("Eigensystem calculated")
        return self.vals, self.vecs

    def check_db(self, tol: bool = 10 ** -15) -> bool:
        if self.peq is None:
            self.calc_peq()

        db_mat = calc_curr_like(self, self.peq)

        for el in self.iter:
            if db_mat[*el] > tol:
                return False
        return True


class WMatrix(TrMatrix):
    def __init__(self, arr, ds: sp.Symbol = None, eq: float = 0, zi: bool = False):
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

    def calc_weq(self, eq: float = None, verbose: bool = False) -> EqMatrix:
        if eq is None:
            eq = self.eq
        self.weq = EqMatrix(self.mat.subs({self.ds: eq}), zi=self.zero_index)
        if verbose:
            print("Equilibrium matrix calculated")
        return self.weq

    def calc_w1(self, eq: float = None, verbose: bool = False) -> Mat:
        if eq is None:
            eq = self.eq
        if self.ds not in list(self.mat.free_symbols):
            raise AttributeError("Driving symbol not found in matrix")

        self.w1 = Mat(sp.diff(self.mat, self.ds).subs({self.ds: eq}), zi=self.zero_index)
        if verbose:
            print("Driving matrix calculated")
        return self.w1

    def calc_coeff(self, force: bool = False, verbose: bool = True) -> CoeffArray:
        return calc_coeff(self, force, verbose=verbose)

    def get_cond(self, i: int, j: int, normal: bool = False, force: bool = False, verbose: bool = True):
        if (self.coeff is None) or force:
            self.calc_coeff(force=force, verbose=verbose)
        if not self.zero_index:
            i -= 1
            j -= 1

        om = sp.Symbol("omega", real=True, positive=True)
        c = deep_symp(
            sum([self.coeff[j, i, k] * (1 if k == 0 else (self.weq.vals[k] / (1j * om - self.weq.vals[k]))) for k in
                 range(self.dim)]))
        if normal:
            n = deep_symp(sum([self.coeff[j, i, k] * (1 if k == 0 else -1) for k in range(self.dim)]))
            return sp.lambdify([om], deep_symp(c / n))
        return sp.lambdify([om], c)

    def get_conds(self, conds: tuple[int, int] | list[list[int]] | None = None,
                  normal: bool = False,
                  force: bool = False,
                  verbose: bool = True) -> list[tuple[callable, list[int]]]:

        if conds is None:
            self.calc_weq()
            self.calc_w1()
            conds = []
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    if self.weq[i, j] != 0 or self.weq[j, i] != 0 or self.w1[i, j] != 0 or self.w1[j, i] != 0:
                        conds.append([i, j])
        elif type(conds[0]) == int:
            conds = [conds]
        return [(self.get_cond(*cond, normal=normal, force=force, verbose=verbose), cond) for cond in conds]



def arr_to_mat(arr):
    """Cast array-like object to sympy mat and check shape requirements"""
    m = sp.Matrix(arr)

    # Check if matrix is vector, transpose to col vec if necessary
    if (m.rows == 1) ^ (m.cols == 1):
        if m.cols > 1:
            return m.T
        else:
            return m
    # If not vector must be square matrix of dim at least (2, 2)
    elif not m.is_square:
        raise NonSquareMatrixError(
            "Matrix object must be square or vector, is neither: ({}, {})".format(m.rows, m.cols))
    elif m.rows == 1:
        raise ShapeError("Matrix must be at least (2, 2) dimensional")
    else:
        return m


def calc_coeff(w, force=False, verbose=True):
    if (w.weq is None) or force:
        w.calc_weq(verbose=verbose)

    weq = w.weq

    if (weq.vals is None) or force:
        weq.calc_eig(verbose=verbose)
    peq = weq.peq
    vals = weq.vals
    vecs = weq.vecs

    if (w.w1 is None) or force:
        w.calc_w1(verbose=verbose)
    w1 = w.w1

    p1coeffs = [-inner(vecs[k], w1 * peq, peq) / vals[k] for k in range(1, len(vals))]

    coeffs = CoeffArray(w.dim)
    for k in range(w.dim):
        if k == 0:
            coeffs[:, :, k] = calc_curr_like(w1, peq).mat
        else:
            coeffs[:, :, k] = (p1coeffs[k - 1] * calc_curr_like(weq, vecs[k])).mat

    w.coeff = coeffs
    return coeffs


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


def check_diag(m, err=False):
    """Check if columns sum to 0"""
    for col in range(m.cols):
        if sum(m.col(col)) != 0:
            if err:
                raise ValueError("Column {} does not sum to 0 (sum = {})".format(col, sum(m.col(col))))
            else:
                return False
    return True


def check_rates(m, err=False, verbose=False):
    """Check if off-diagonal elements are 0"""
    for col in range(m.cols):
        for row in range(m.rows):
            try:
                if row != col and m[row, col] < 0:
                    if err:
                        raise ValueError(
                            "Transfer rate ({} -> {}) is negative ({})".format(col + 1, row + 1, m[row, col]))
                    else:
                        return False
            except TypeError:
                if verbose:
                    print("Positivity of transfer rate ({} -> {}) undetermined ({})".format(col + 1, row + 1,
                                                                                            m[row, col]))
    return True


def deep_symp(e):
    """Simplify an arbitrary expression, both numerically and symbolically"""
    e = sp.nsimplify(e, rational=True, full=True)
    e = sp.simplify(e)
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


def inner(v1, v2, Peq=None):
    n = len(v1)
    if Peq is None:
        Peq = [1 for _ in range(n)]

    return deep_symp(sum([v1[i] * v2[i] / Peq[i] for i in range(n)]))


