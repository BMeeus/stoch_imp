# import sympy as sp

from itertools import product
from numpy import zeros
from sympy import (Matrix, NonSquareMatrixError, nsimplify, pretty, ShapeError, simplify, zeros)

class CoeffArray:
    """
    Class handling the array containing the coefficients.
    """

    def __init__(self, n):
        """
        Initialising an instance of CoeffArray

        :param n: size of the system
        """
        self.mat = zeros((n, n, n), dtype=object)

    # Let numpy handle item getting and setting
    def __getitem__(self, item):
        return self.mat.__getitem__(item)

    def __setitem__(self, key, value):
        self.mat.__setitem__(key, value)

    def __str__(self):
        return pretty(self.mat)


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
        return pretty(self.mat)

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
                arr = zeros(arr, arr)

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
            r = nsimplify(r, rational=True)

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
                ri = nsimplify(ri, rational=True)
            try:
                if ri < 0:
                    raise ValueError(f"Inverse rate should be positive, is {ri}")
            except TypeError:
                pass
            self.mat[i, j] = ri
            self.mat[j, j] -= ri
        if simp:
            simplify(self.mat)
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


def arr_to_mat(arr):
    """Cast array-like object to sympy mat and check shape requirements"""
    m = Matrix(arr)

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
