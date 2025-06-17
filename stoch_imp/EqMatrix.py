from typing import (Optional, Union)

from numpy import (array, argsort, flip, float64, hstack, matmul, ndarray, real)
from scipy.linalg import (eig, norm)
from sympy import (N, Matrix, re)

from .core_classes import (ConstantMatrix, TrMatrix, ConstantObject, arr_to_mat)
from .util import (calc_curr_like, deep_simp, gram_schmidt)


class EqMatrix(ConstantObject, TrMatrix):
    """Class handling all Equilibrium Transfer Matrices."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the EqMatrix object.

        :param args: (tuple) Positional arguments for parent constructors
        :param kwargs: (dict) Keyword arguments for parent constructors
        """
        super().__init__(*args, **kwargs)
        self.peq: Optional[ConstantMatrix] = None

        self.vals: Optional[list[float]] = None
        self.nvals: Optional[ndarray] = None

        self.vecs: Optional[list[Matrix]] = None
        self.nvecs: Optional[ndarray] = None

    def calc_peq(self, force: bool = False, verbose: bool = False) -> ConstantMatrix:
        """
        Calculate the Equilibrium distribution.

        :param force: (bool) Force recalculation
        :param verbose: (bool) Print verbose output

        :return: (ConstantMatrix) Equilibrium distribution matrix
        """
        if self.peq is None or force:
            self.calc_eig(force=force, verbose=verbose)
        return self.peq

    def calc_eig(self, tol: float = 1e-14, force: bool = False, verbose: bool = False) -> Union[
        tuple[list, list[Matrix]], tuple[ndarray, ndarray]]:
        """
        Calculate eigenvalues and eigenvectors.

        :param tol: (float) Tolerance level for eigenvalue validation
        :param force: (bool) Force recalculation
        :param verbose: (bool) Print verbose output

        :return: (Union[tuple[list, list[Matrix]], tuple[ndarray, ndarray]]) Eigenvalues and eigenvectors
        """
        if self.is_symbolic:
            return _sym_calc_eig(self, tol=tol, force=force, verbose=verbose)
        else:
            return _num_calc_eig(self, tol=tol, force=force, verbose=verbose)

    def check_db(self, tol: float = 1e-15) -> bool:
        """
        Check detailed balance condition.

        :param tol: (float) Tolerance level for checking detailed balance

        :return: (bool) True if detailed balance is fulfilled
        """
        if self.peq is None:
            self.calc_peq()
        db_mat = calc_curr_like(self, self.peq)
        return all(db_mat[*el] <= tol for el in self.iter)

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


def _sym_calc_eig(weq, tol: float = 1e-14, force: bool = False, verbose: bool = False) -> tuple[list, list[Matrix]]:
    """
    Compute symbolic eigenvalues and eigenvectors.

    :param weq: (EqMatrix) Equilibrium matrix object
    :param tol: (float) Tolerance for validation
    :param force: (bool) Force recalculation
    :param verbose: (bool) Print verbose output

    :return: (tuple[list, list[Matrix]]) Eigenvalues and eigenvectors
    """
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
    if not weq.check_db(tol=tol):
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


def _num_calc_eig(weq, tol=1e-15, force=False, verbose=False):
    """
    Compute numeric eigenvalues and eigenvectors.

    :param weq: (EqMatrix) Equilibrium matrix object
    :param tol: (float) Tolerance for validation
    :param force: (bool) Force recalculation
    :param verbose: (bool) Print verbose output

    :return: (tuple[ndarray, ndarray]) Eigenvalues and eigenvectors
    """
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
    if not weq.check_db(tol=tol):
        raise ValueError("detailed balance not fulfilled")
    if verbose:
        print("Equilibrium distribution calculated")

    nvecs = gram_schmidt(nvecs, nvecs[:, 0])

    for i in range(len(vecs)):
        res = norm(matmul(weq.nmat, vecs[:, i]) - vals[i] * vecs[:, i])
        if res >= tol:
            raise ValueError("Incorrect computation of eigenvectors")

    weq.nvals = nvals
    weq.nvecs = nvecs
    if verbose:
        print("Eigensystem calculated")
    return nvals, nvecs
