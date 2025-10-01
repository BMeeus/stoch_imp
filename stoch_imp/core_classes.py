from __future__ import annotations

from typing import Any
from itertools import product

from numpy import (array, float64, ndarray)
from numpy import zeros as np_zeros
from sympy import (Expr, Matrix, N, NonSquareMatrixError, nsimplify, pretty, ShapeError, simplify, zeros)


class CoeffArray:
    """
    (n, n, n) dim coefficient array class for storing transition coefficients. Prints pretty.

    :ivar mat: The array of coefficients
    :ivar iter: A simple iterator to iterate over the whole array. Use as `(elem in self.iter)` or `(i, j, k in self.iter)`
    """
    def __init__(self, n: int):
        """
        Initialize a 3D numpy array of objects with shape (n, n, n).

        :param n: Dimension size of the 3D array
        :type n: int

        """
        self.mat = np_zeros((n, n, n), dtype=object)
        self.iter = product(range(n), range(n), range(n))

    def __getitem__(self, item):
        # Use native NumPy indexing
        return self.mat[item]

    def __setitem__(self, key, value):
        # Use native NumPy assignment
        self.mat[key] = value

    def __str__(self):
        # Pretty-print the full array using SymPy formatting
        return pretty(self.mat)

    def to_num(self):
        """Convert array to numeric type"""
        for i, j, k in self.iter:
            if len(self.mat[i, j, k].free_symbols) != 0:
                raise TypeError("Can not convert expression containing symbols to numeric")
            self.mat[i, j, k] = N(self.mat[i, j, k])
        self.mat = self.mat.astype(float64)

    def to_sym(self):
        """Convert array to symbolic type"""
        self.mat = self.mat.astype(object)

class ConstantObject:
    """
    Stores the numeric matrix form of a symbolic matrix when symbolic evaluation is disabled.

    :ivar nmat: Numeric matrix representation if symbolic flag is False; otherwise None.
    """

    def __init__(self, nmat, *args, **kwargs):
        """
        Store numeric matrix form of symbolic matrix if symbolic flag is False.

        :param nmat: Matrix input
        :type nmat: ndarray or array-like
        """
        super().__init__(nmat, *args, **kwargs)  # forwards all unused arguments
        if "sym" not in kwargs.keys() or kwargs["sym"]:
            self.nmat = None
        elif isinstance(nmat, ndarray):
            self.nmat = nmat.astype(float64)
        elif hasattr(nmat, "free_symbols") and len(nmat.free_symbols) > 0:
            raise TypeError("Can not convert expression containing symbols to numeric")
        else:
            self.nmat = array(N(nmat)).astype(float64)



# noinspection PyTypeChecker
class Mat:
    """
    Top level class for all matrix-like objects such as Vectors, WMatrix, EqMatrix, etc.

    :ivar mat: The internal sympy.Matrix representation of the object.
    :ivar dim: Number of rows in the matrix (system size).
    :ivar iter: Iterator over all (row, column) index pairs in the matrix.
    :ivar zero_index: Boolean indicating whether 0-based indexing is used.
    :ivar is_symbolic: Boolean indicating whether the matrix is treated as symbolic.
    :ivar tol: Numerical tolerance used for approximate comparisons.
    """

    def __init__(self, arr, *args, **kwargs):
        """
        Initialize matrix object from array-like input.

        :param arr: Input data for matrix
        :type arr: array-like
        """
        m = arr_to_mat(arr)  # Convert array-like to sympy.Matrix
        self.mat = m
        self.dim = m.rows  # System size inferred from row count
        self.iter = product(range(m.rows), range(m.cols))  # Iterator over all (row, col) pairs
        self.zero_index = kwargs.setdefault("zi", False)  # True if system uses 0-based indexing
        self.is_symbolic = kwargs.setdefault("sym", True)
        self.tol = kwargs.setdefault("tol", 10e-14)

    def __str__(self):
        # Pretty-print matrix using sympy's printer
        return pretty(self.mat)

    def __setitem__(self, key, value):
        # Support assignment via indexing
        self.mat[key] = value

    def __getitem__(self, item):
        # Support retrieval via indexing
        return self.mat[item]

    def __sub__(self, other) -> Mat:
        # Support subtraction with another Mat or scalar
        return Mat(self.mat - getattr(other, 'mat', other), zi=self.zero_index)

    def __mul__(self, other) -> Mat:
        # Matrix multiplication (supports Mat or scalar)
        return Mat(self.mat * getattr(other, 'mat', other), zi=self.zero_index)

    def __rmul__(self, other) -> Mat:
        # Right multiplication (e.g., scalar * Mat)
        return Mat(getattr(other, 'mat', other) * self.mat, zi=self.zero_index)

    def __truediv__(self, other):
        # Scalar division
        return self.mat / other

    def __len__(self):
        # Length = number of elements in matrix
        return len(self.mat)

    def __getattr__(self, name):
        # Forward missing attributes to underlying sympy.Matrix
        return getattr(self.mat, name)

    def __dir__(self):
        # Combine native attributes with sympy.Matrix's for auto-completion
        return list(set(super().__dir__()) | set(dir(self.mat)))


class ConstantMatrix(ConstantObject, Mat):
    """
    Matrix class that supports both symbolic and numeric representations, with conversion between the two.

    Inherits from ConstantObject and Mat to combine symbolic matrix functionality with numeric handling.

    :ivar mat: Internal sympy.Matrix representation.
    :ivar nmat: Numeric matrix representation (if symbolic mode is off).
    :ivar dim: Number of rows in the matrix.
    :ivar iter: Iterator over matrix indices.
    :ivar zero_index: Boolean indicating whether 0-based indexing is used.
    :ivar is_symbolic: Boolean indicating whether the matrix is in symbolic form.
    :ivar tol: Numerical tolerance used for operations.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ConstantMatrix object."""
        super().__init__(*args, **kwargs)

    def __sub__(self, other) -> ConstantMatrix:
        # Support subtraction with another Mat or scalar
        return ConstantMatrix(self.mat - getattr(other, 'mat', other), zi=self.zero_index, sym=self.is_symbolic)

    def __mul__(self, other) -> ConstantMatrix:
        # Matrix multiplication (supports Mat or scalar)
        return ConstantMatrix(self.mat * getattr(other, 'mat', other), zi=self.zero_index, sym=self.is_symbolic)

    def __rmul__(self, other) -> ConstantMatrix:
        # Right multiplication (e.g., scalar * Mat)
        return ConstantMatrix(getattr(other, 'mat', other) * self.mat, zi=self.zero_index, sym=self.is_symbolic)

    def __truediv__(self, other):
        # Scalar division
        return ConstantMatrix(self.mat / other)

    def to_num(self):
        """Convert symbolic matrix to numeric form."""
        if not self.is_symbolic:
            pass
        elif len(self.mat.free_symbols) != 0:
            raise TypeError("Can not convert expression containing symbols to numeric")
        else:
            self.nmat = array(N(self.mat), dtype=float64)
            self.is_symbolic = False

    def to_sym(self):
        """Convert numeric matrix back to symbolic form."""
        if self.is_symbolic:
            pass
        else:
            self.mat = arr_to_mat(self.nmat)
            self.is_symbolic = True


# noinspection PyTypeChecker
class TrMatrix(Mat):
    """
    Class handling all transition matrices. These matrices should sum to zero along columns and have positive
    off-diagonal elements.

    :ivar mat: Internal sympy.Matrix representation.
    :ivar dim: Number of rows in the matrix.
    :ivar iter: Iterator over matrix indices.
    :ivar zero_index: Boolean indicating whether 0-based indexing is used.
    :ivar is_symbolic: Boolean indicating whether the matrix is in symbolic form.
    :ivar tol: Numerical tolerance used for validations.
    """

    def __init__(self, arr: Any, *args, **kwargs):
        """
        Initialize transition matrix and validate constraints.

        :param arr: Input data or dimension
        :type arr: array-like or int
        """
        if isinstance(arr, int):
            if arr <= 1:
                raise ShapeError("Transfer matrix must be at least (2, 2) dimensional")
            arr = zeros(arr, arr)

        super().__init__(arr, *args, **kwargs)
        check_diag(self, err=True, tol=self.tol)  # Validate initial conditions
        check_rates(self, err=True)

    def add_trans(self, i: int, j: int,
                  r: float | Expr = 1.0,
                  ri: float | Expr | None = None,
                  symm: bool = True,
                  simp: bool = True):
        """
        Add a transition `i --> j` with rate `r`, and optionally symmetric transition `j --> i` with rate `ri`.

        :param i: Start index
        :type i: int
        :param j: Target index
        :type j: int
        :param r: Transition rate from i to j
        :type r: float | Expr
        :param ri: Optional inverse rate (j to i)
        :type ri: float | Expr | None
        :param symm: If True, adds j --> i with rate 1/r or ri
        :type symm: bool
        :param simp: If True, simplifies r and ri
        :type simp: bool
        """
        if i == j:
            raise ValueError("Start and target state must differ.")

        if not self.zero_index:
            i -= 1
            j -= 1

        if simp:
            r = nsimplify(r, rational=True)

        try:
            if r < 0:
                raise ValueError(f"Rate should be positive, got {r}")
        except TypeError:
            pass  # Ignore symbolic inequalities

        self[j, i] = r
        self[i, i] -= r

        if symm or (ri is not None):
            if ri is None:
                ri = r ** -1
            if simp:
                ri = nsimplify(ri, rational=True)

            try:
                if ri < 0:
                    raise ValueError(f"Inverse rate should be positive, got {ri}")
            except TypeError:
                pass  # Ignore symbolic inequalities

            self[i, j] = ri
            self[j, j] -= ri

        if simp:
            simplify(self.mat)

    def calc_diag(self):
        """Force recomputation of diagonal terms to preserve column sum = 0."""
        for col in range(self.dim):
            self[col, col] = -sum(self[row, col] for row in range(self.dim) if row != col)

    def check_diag(self, **flags) -> bool:
        """
        Check if each column in matrix m sums to zero.

        :param m: Matrix to check
        :type m: Mat
        :param tol: Allowed numerical tolerance
        :type tol: float
        :param err: Raise exception if check fails
        :type err: bool

        :return: True if all columns sum to zero
        :rtype: bool
        """
        err = flags.setdefault("err", False)
        for col in range(self.dim):
            s = N(sum(self[:, col]))
            if s > self.tol:
                if err:
                    raise ValueError(f"Column {col} does not sum to 0 (sum = {s})")
                return False
        return True

    def check_rates(self, **flags) -> bool:
        """
        Check that all off-diagonal elements are non-negative.

        :param m: Matrix to check
        :type m: Mat
        :param err: Raise exception on negative element
        :type err: bool
        :param verbose: Log indeterminate cases due to symbolic expressions
        :type verbose: bool

        :return: True if all off-diagonal elements are non-negative
        :rtype: bool
        """
        err = flags.setdefault("err", False)
        verbose = flags.setdefault("verbose", False)

        for col, row in self.iter:
            if row == col:
                continue
            try:
                if self[row, col] < 0:
                    if err:
                        raise ValueError(f"Transfer rate ({col + 1} -> {row + 1}) is negative: {self[row, col]}")
                    return False
            except TypeError:
                if verbose:
                    print(f"Positivity of transfer rate ({col + 1} -> {row + 1}) undetermined: {self[row, col]}")
        return True


def arr_to_mat(arr) -> Matrix:
    """
    Convert input array-like to sympy.Matrix.
      - Ensures proper shape.
      - Converts row vectors to column vectors.

    :param arr: Input data
    :type arr: array-like
    :return: SymPy matrix
    :rtype: Matrix
    """
    m = Matrix(arr)

    # Check if matrix is vector, transpose to col vec if necessary
    if (m.rows == 1) ^ (m.cols == 1):
        return m.T if m.cols > 1 else m

    if not m.is_square:
        raise NonSquareMatrixError(f"Matrix must be square or vector, got ({m.rows}, {m.cols})")
    if m.rows == 1:
        raise ShapeError("Matrix must be at least (2, 2) dimensional")

    return m


def check_diag(m: Mat, tol: float = 1e-15, err: bool = False) -> bool:
    """
    Check if each column in matrix m sums to zero.

    :param m: Matrix to check
    :type m: Mat
    :param tol: Allowed numerical tolerance
    :type tol: float
    :param err: Raise exception if check fails
    :type err: bool

    :return: True if all columns sum to zero
    :rtype: bool
    """
    for col in range(m.dim):
        s = N(sum(m[:, col]))
        if s > tol:
            if err:
                raise ValueError(f"Column {col} does not sum to 0 (sum = {s})")
            return False
    return True


def check_rates(m: Mat, **flags) -> bool:
    """
    Check that all off-diagonal elements are non-negative.

    :param m: Matrix to check
    :type m: Mat
    :param err: Raise exception on negative element
    :type err: bool
    :param verbose: Log indeterminate cases due to symbolic expressions
    :type verbose: bool

    :return: True if all off-diagonal elements are non-negative
    :rtype: bool
    """

    err = flags.setdefault('err', False)
    verbose = flags.setdefault('verbose', False)

    for col, row in m.iter:
        if row == col:
            continue
        try:
            if m[row, col] < 0:
                if err:
                    raise ValueError(f"Transfer rate ({col + 1} -> {row + 1}) is negative: {m[row, col]}")
                return False
        except TypeError:
            if verbose:
                print(f"Positivity of transfer rate ({col + 1} -> {row + 1}) undetermined: {m[row, col]}")
    return True
