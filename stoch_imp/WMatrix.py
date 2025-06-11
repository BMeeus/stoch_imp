from typing import Callable, Optional, Union

import numpy as np
import sympy as sp
from sympy import diff, lambdify, Symbol, pretty

from .EqMatrix import EqMatrix
from .core_classes import CoeffArray, TrMatrix, ConstantMatrix
from .util import calc_curr_like, deep_simp, inner


class WMatrix(TrMatrix):
    """Class handling driven transition matrices."""

    def __init__(self, arr, ds: Optional[Symbol] = None, eq: float = 0, zi: bool = False):
        super().__init__(arr, zi)
        if ds is None:
            symbols = list(self.free_symbols)
            if not symbols:
                raise AttributeError("No driving symbol found or given")
            if len(symbols) > 1:
                raise ValueError("Ambiguity in driving symbol, please provide a specific symbol")
            ds = symbols[0]
            print(f"Driving symbol set to {pretty(ds)}")

        self.ds: Symbol = ds
        self.eq: float = eq
        self.weq: Optional[EqMatrix] = None
        self.w1: Optional[ConstantMatrix] = None
        self.coeff: Optional[CoeffArray] = None

    def calc_weq(self, eq: float = None, force: bool = False, verbose: bool = False) -> EqMatrix:
        if self.weq is not None and not force:
            return self.weq
        self.weq = EqMatrix(self.subs({self.ds: eq or self.eq}), zi=self.zero_index)
        if verbose:
            print("Equilibrium matrix calculated")
        return self.weq

    def calc_w1(self, eq: float = None, force: bool = False, verbose: bool = False) -> ConstantMatrix:
        if self.w1 is not None and not force:
            return self.w1
        eq = eq if eq is not None else self.eq
        if self.ds not in self.free_symbols:
            raise AttributeError("Driving symbol not found in matrix")
        self.w1 = ConstantMatrix(diff(self, self.ds).subs({self.ds: eq}), zi=self.zero_index)
        if verbose:
            print("Driving matrix calculated")
        return self.w1

    def calc_coeff(self, force: bool = False, verbose: bool = True) -> CoeffArray:
        return _calc_coeff(self, force, verbose=verbose)

    def get_cond(self, i: int, j: int, normal: bool = False, force: bool = False, verbose: bool = True) -> Callable[
        [float | np.ndarray], float]:
        if self.coeff is None or force:
            self.calc_coeff(force=force, verbose=verbose)
        if not self.zero_index:
            i -= 1
            j -= 1
        omega = Symbol("omega", real=True, positive=True)
        coeff = self.coeff
        vals = self.weq.vals
        numerator = deep_simp(sum(
            coeff[j, i, k] * (1 if k == 0 else (vals[k] / (1j * omega - vals[k])))
            for k in range(self.dim)))
        if normal:
            denominator = deep_simp(sum(
                coeff[j, i, k] * (1 if k == 0 else -1)
                for k in range(self.dim)))
            return lambdify([omega], deep_simp(numerator / denominator))
        return lambdify([omega], numerator)

    def get_conds(self,
                  conds: Union[tuple[int, int], list[tuple[int, int]], None] = None,
                  normal: bool = False,
                  force: bool = False,
                  verbose: bool = True) -> list[tuple[tuple[int, int], Callable[[float], float]]]:
        if conds is None:
            self.calc_weq()
            self.calc_w1()
            conds = [(i, j) for i in range(self.dim) for j in range(i + 1, self.dim)
                     if any(elem != 0 for elem in (self.weq[i, j], self.weq[j, i], self.w1[i, j], self.w1[j, i]))]
        elif isinstance(conds[0], int):
            conds = [conds]
        return [(cond, self.get_cond(*cond, normal=normal, force=force, verbose=verbose)) for cond in conds]

    def to_num(self, force: bool = False):
        if self.weq is not None:
            self.weq.to_num()
        if self.w1 is not None:
            self.w1.to_num()
        if self.coeff is not None:
            self.coeff = sp.N(self.coeff)
        self.is_symbolic = False
        self.is_numeric = True


def _calc_coeff(w: WMatrix, force: bool = False, verbose: bool = True) -> CoeffArray:
    w.calc_weq(force=force, verbose=verbose)
    weq = w.weq
    weq.calc_eig(force=force, verbose=verbose)
    peq, vals, vecs = weq.peq, weq.vals, weq.vecs

    w.calc_w1(force=force, verbose=verbose)
    w1 = w.w1

    p1coeffs = [-inner(vecs[k], w1 * peq, peq) / vals[k] for k in range(1, len(vals))]

    coeffs = CoeffArray(w.dim)
    coeffs[:, :, 0] = calc_curr_like(w1, peq)
    for k in range(1, w.dim):
        coeffs[:, :, k] = p1coeffs[k - 1] * calc_curr_like(weq, vecs[k])

    w.coeff = coeffs
    return coeffs
