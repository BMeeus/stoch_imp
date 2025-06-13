from typing import Optional

from numpy import (array, hstack)
from sympy import N, Matrix, re

from .core_classes import ConstantMatrix, TrMatrix, ConstantObject
from .util import calc_curr_like, deep_simp, gram_schmidt


class EqMatrix(ConstantObject, TrMatrix):
    """Class handling all Equilibrium Transfer Matrices."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.peq: Optional[ConstantMatrix] = None
        self.vals: Optional[list[float]] = None
        self.nvals = None
        self.vecs: Optional[list[Matrix]] = None
        self.nvecs = None

    def calc_peq(self, force: bool = False, verbose: bool = False) -> ConstantMatrix:
        """Calculate the Equilibrium distribution."""
        if self.peq is None or force:
            self.calc_eig(force=force, verbose=verbose)
        return self.peq

    def calc_eig(self, tol: float = 1e-14, force: bool = False, verbose: bool = False) -> tuple[list, list[Matrix]]:
        if self.vals is not None and self.vecs is not None and not force:
            return self.vals, self.vecs

        eig_syst = self.eigenvects(error_when_incomplete=True)
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
