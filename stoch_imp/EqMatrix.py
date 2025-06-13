from typing import Optional, Union

from numpy import array, hstack, real, flip, argsort, matmul, ndarray, float64
from sympy import N, Matrix, re
from scipy.linalg import eig, norm

from .core_classes import ConstantMatrix, TrMatrix, ConstantObject, arr_to_mat
from .util import calc_curr_like, deep_simp, gram_schmidt


class EqMatrix(ConstantObject, TrMatrix):
    """Class handling all Equilibrium Transfer Matrices."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.peq: Optional[ConstantMatrix] = None

        self.vals: Optional[list[float]] = None
        self.nvals: Optional[ndarray] = None

        self.vecs: Optional[list[Matrix]] = None
        self.nvecs: Optional[ndarray] = None

    def calc_peq(self, force: bool = False, verbose: bool = False) -> ConstantMatrix:
        """Calculate the Equilibrium distribution."""
        if self.peq is None or force:
            self.calc_eig(force=force, verbose=verbose)
        return self.peq

    def calc_eig(self, tol: float = 1e-14, force: bool = False, verbose: bool = False) -> tuple[list, list[Matrix]]:
        if self.vals is not None and self.vecs is not None and not force:
            return self.vals, self.vecs
    def to_num(self):
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
        if self.is_symbolic:
            pass
            self.mat = arr_to_mat(self.nmat)

            if self.peq is not None:
                self.peq.to_sym()
                self.vals = list(self.nvals)
                self.vecs = [Matrix(self.nvecs[:, i]) for i in self.dim]

            self.is_symbolic = True


def _sym_calc_eig(weq, tol: float = 1e-14, force: bool = False, verbose: bool = False) -> tuple[list, list[Matrix]]:
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
        self.peq = ConstantMatrix(peq_vec.as_real_imag()[0])
        if verbose:
            print("Equilibrium distribution calculated")

        if not self.check_db(tol=tol):
            raise ValueError("detailed balance not fulfilled")

        vals = [0]
        vecs = [self.peq]

        for val, mult, basis in eig_syst[1:]:
            val_re, val_im = val.as_real_imag()
            if val_im > tol:
                raise ValueError(f"Complex eigenvalue found: {val}")
            for vec in basis[:mult]:
                vec_re, _ = vec.as_real_imag()
                res = deep_simp(self.mat * vec_re) - deep_simp(val_re * vec_re)
                try:
                    if N(res.norm()) > tol:
                        raise ValueError("Incorrect computation of eigenvectors")
                except TypeError:
                    if res.norm().is_zero:
                        raise ValueError("Incorrect computation of eigenvectors")
                vals.append(val_re)
                vecs.append(vec_re)

        self.vecs = gram_schmidt(vecs, self.peq)
        self.vals = vals
        if verbose:
            print("Eigensystem calculated")
        return self.vals, self.vecs

    def check_db(self, tol: float = 1e-15) -> bool:
        if self.peq is None:
            self.calc_peq()
        db_mat = calc_curr_like(self, self.peq)
        return all(db_mat[*el] <= tol for el in self.iter)

    def to_num(self):
        self.calc_peq()
        self.nmat = array(N(self.mat))
        if hasattr(self.peq, "to_num") and callable(self.peq.to_num):
            self.peq.to_num()
        self.nvals = array([N(val) for val in self.vals])
        self.nvecs = hstack([N(vec) for vec in self.vecs])
        self.is_symbolic = False
        self.is_numeric = True
