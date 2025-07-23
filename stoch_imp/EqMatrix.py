from typing import (Optional, Union)

from numpy import (array, argsort, flip, float64, hstack, matmul, ndarray, real)
from scipy.linalg import (eig, norm)
from sympy import (N, Matrix, re)

from .core_classes import (ConstantMatrix, TrMatrix, ConstantObject, arr_to_mat)
from .util import (calc_curr_like, deep_simp, gram_schmidt)


class EqMatrix(ConstantObject, TrMatrix):
    """
    Class handling all Equilibrium Transfer Matrices.

    This class extends ConstantObject and TrMatrix to provide functionality for equilibrium systems,
    including eigenvalue/eigenvector computation and equilibrium distribution (peq) calculation.

    :ivar mat: Internal sympy.Matrix representation of the transition matrix.
    :ivar dim: Number of rows in the matrix.
    :ivar iter: Iterator over matrix indices.
    :ivar zero_index: Boolean indicating whether 0-based indexing is used.
    :ivar is_symbolic: Boolean indicating whether the matrix is in symbolic form.
    :ivar tol: Numerical tolerance used for validations.
    :ivar nmat: Numeric matrix representation if symbolic flag is False; otherwise None.
    :ivar peq: Equilibrium distribution as a ConstantMatrix, or None if not yet calculated.
    :ivar vals: List of symbolic eigenvalues, or None if not yet calculated.
    :ivar nvals: Numpy array of numeric eigenvalues, or None if not yet calculated.
    :ivar vecs: List of symbolic eigenvectors, or None if not yet calculated.
    :ivar nvecs: Numpy array of numeric eigenvectors, or None if not yet calculated.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the EqMatrix object.
        """
        super().__init__(*args, **kwargs)
        self.peq: Optional[ConstantMatrix] = None

        self.vals: Optional[list[float]] = None
        self.nvals: Optional[ndarray] = None

        self.vecs: Optional[list[Matrix]] = None
        self.nvecs: Optional[ndarray] = None

    def calc_peq(self, **flags) -> ConstantMatrix:
        """
        Calculate the Equilibrium distribution.

        :keyword force: Force recalculation
        :type force: bool
        :keyword verbose: Print verbose output
        :type verbose: bool
        :keyword chk_db: Check detailed balance
        :type chk_db: bool

        :return: Equilibrium distribution matrix
        :rtype: ConstantMatrix
        """
        if self.peq is None or flags.setdefault('force', False):
            self.calc_eig(**flags)
        return self.peq

    def calc_eig(self, **flags) -> Union[tuple[list, list[Matrix]], tuple[ndarray, ndarray]]:
        """
        Calculate eigenvalues and eigenvectors.

        :param force: Force recalculation
        :type force: bool
        :param verbose: Print verbose output
        :type verbose: bool
        :param chk_db: Check detailed balance
        :type chk_db: bool

        :return: Eigenvalues and eigenvectors
        """
        if self.is_symbolic:
            return _sym_calc_eig(self, tol=self.tol, **flags)
        else:
            return _num_calc_eig(self, tol=self.tol, **flags)

    def check_db(self) -> bool:
        """
        Check detailed balance condition.

        :return: True if detailed balance is fulfilled
        :rtype: bool
        """
        if self.peq is None:
            self.calc_peq()
        db_mat = calc_curr_like(self, self.peq)
        return all(db_mat[*el] <= self.tol for el in self.iter)

    def to_num(self):
        """Convert symbolic matrices to numeric format."""
        if not self.is_symbolic:
            pass
        elif len(self.mat.free_symbols) != 0:
            raise TypeError("Can not convert expression containing symbols to numeric")
        else:
            self.nmat = array(N(self.mat), dtype=float64)

            if self.peq is not None:
                self.peq.to_num()
                self.nvals = array([N(val) for val in self.vals])
                self.nvecs = hstack([N(vec) for vec in self.vecs])

            self.is_symbolic = False

    def to_sym(self):
        """Convert numeric matrices back to symbolic format."""
        if self.is_symbolic:
            pass
        else:
            self.mat = arr_to_mat(self.nmat)

            if self.peq is not None:
                self.peq.to_sym()
                self.vals = list(self.nvals)
                self.vecs = [Matrix(self.nvecs[:, i]) for i in range(self.dim)]

            self.is_symbolic = True


def _sym_calc_eig(weq, tol, **flags) -> tuple[list, list[Matrix]]:
    """
    Compute symbolic eigenvalues and eigenvectors.
    """

    force = flags.setdefault('force', False)
    verbose = flags.setdefault('force', False)
    chk_db = flags.setdefault('force', True)

    if weq.vals is not None and weq.vecs is not None and not force:
        return weq.vals, weq.vecs

    eig_syst = weq.eigenvects(error_when_incomplete=True)
    eig_syst.sort(key=lambda x: re(x[0]), reverse=True)

    lead_val = eig_syst[0][0]
    if re(lead_val) > tol:
        raise ValueError("no eigenvalue 0 was found")
    elif eig_syst[0][1] != 1:
        raise ValueError("multiple steady states found")

    eigvec_0 = eig_syst[0][-1][0]
    peq_vec = eigvec_0 / sum(eigvec_0)
    weq.peq = ConstantMatrix(peq_vec.as_real_imag()[0], zi=weq.zero_index, sym=True)
    if chk_db and not weq.check_db():
        raise ValueError("detailed balance not fulfilled")
    if verbose:
        print("Equilibrium distribution calculated")

    vals = [0]
    vecs = [weq.peq]

    for val, mult, basis in eig_syst[1:]:
        val_re, val_im = val.as_real_imag()
        if val_im > tol:
            raise ValueError(f"Complex eigenvalue found: {val}")
        for vec in basis[:mult]:
            vec_re, _ = vec.as_real_imag()
            res = deep_simp(weq.mat * vec_re) - deep_simp(val_re * vec_re)
            try:
                if N(res.norm()) > tol:
                    raise ValueError("Incorrect computation of eigenvectors")
            except TypeError:
                if res.norm().is_zero:
                    raise ValueError("Incorrect computation of eigenvectors")
            vals.append(val_re)
            vecs.append(vec_re)

    weq.vecs = gram_schmidt(vecs, weq.peq)
    weq.vals = vals
    if verbose:
        print("Eigensystem calculated")
    return weq.vals, weq.vecs


def _num_calc_eig(weq, tol, **flags):
    """
    Compute numeric eigenvalues and eigenvectors.
    """

    force = flags.setdefault('force', False)
    verbose = flags.setdefault('force', False)
    chk_db = flags.setdefault('force', True)

    if weq.nvals is not None and weq.nvecs is not None and not force:
        return weq.nvals, weq.nvecs

    # noinspection PyTupleAssignmentBalance
    vals, vecs = eig(weq.nmat)
    nvals = real(vals)
    nvecs = real(vecs)
    if any(im > tol for im in vals - nvals):
        raise ValueError(f"Complex eigenvalue found: {vals}")

    ind_arr = flip(argsort(nvals))  # Sort eigenvalues in descending order and rearrange eigenvecs accordingly

    nvals = nvals[ind_arr]
    nvecs = nvecs[:, ind_arr]

    if nvals[0] > tol:
        raise ValueError("no eigenvalue 0 was found")
    if nvals[1] == nvals[0]:
        raise ValueError("multiple steady states found")

    nvals[0] = 0
    nvecs[:, 0] /= sum(nvecs[:, 0])  # First eigenvector is Peq, normalize and rename
    weq.peq = ConstantMatrix(nvecs[:, 0], zi=weq.zero_index, sym=False)
    if chk_db and not weq.check_db():
        raise ValueError("detailed balance not fulfilled")
    if verbose:
        print("Equilibrium distribution calculated")

    nvecs = gram_schmidt(nvecs, nvecs[:, 0])

    for i in range(len(vecs)):
        res = norm(matmul(weq.nmat, vecs[:, i]) - vals[i] * vecs[:, i])
        if res >= tol:
            raise ValueError(f"Incorrect computation of eigenvectors (residue {res})")

    weq.nvals = nvals
    weq.nvecs = nvecs
    if verbose:
        print("Eigensystem calculated")
    return nvals, nvecs
