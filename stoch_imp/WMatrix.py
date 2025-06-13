from typing import Callable, Optional, Union

from numpy import matmul, ndarray
from numpy import sum as np_sum
from sympy import diff, lambdify, Symbol, pretty

from .EqMatrix import EqMatrix
from .core_classes import CoeffArray, TrMatrix, ConstantMatrix
from .util import calc_curr_like, deep_simp, inner


class WMatrix(TrMatrix):
    """Class handling driven transition matrices."""

    def __init__(self, *args, ds: Optional[Symbol] = None, eq: float = 0, **kwargs):
        """
        Initialize a WMatrix instance.

        :param args: Additional positional arguments for TrMatrix
        :param ds: (Optional[Symbol]) Driving symbol for differentiation
        :param eq: (float) Equilibrium point
        :param kwargs: Additional keyword arguments for TrMatrix
        """
        super().__init__(*args, **kwargs)
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
        """
        Calculate the equilibrium matrix.

        :param eq: (float) Equilibrium value
        :param force: (bool) Force recalculation even if already computed
        :param verbose: (bool) Print verbose output

        :return: (EqMatrix) the equilibrium matrix.
        """
        if self.weq is not None and not force:
            return self.weq

        self.weq = EqMatrix(self.subs({self.ds: eq or self.eq}), zi=self.zero_index, sym=self.is_symbolic)

        if verbose:
            print("Equilibrium matrix calculated")
        return self.weq

    def calc_w1(self, eq: float = None, force: bool = False, verbose: bool = False) -> ConstantMatrix:
        """
        Calculate the first-order driving matrix.

        :param eq: (float) Equilibrium value
        :param force: (bool) Force recalculation
        :param verbose: (bool) Print verbose output

        :return: (ConstantMatrix) the first-order driving matrix.
        """
        if self.w1 is not None and not force:
            return self.w1

        eq = eq if eq is not None else self.eq

        if self.ds not in self.free_symbols:
            raise AttributeError("Driving symbol not found in matrix")

        self.w1 = ConstantMatrix(diff(self, self.ds).subs({self.ds: eq}), zi=self.zero_index, sym=self.is_symbolic)

        if verbose:
            print("Driving matrix calculated")
        return self.w1

    def calc_coeff(self, force: bool = False, verbose: bool = True) -> CoeffArray:
        """
        Calculate coefficient array.

        :param force: (bool) Force recalculation
        :param verbose: (bool) Print verbose output

        :return: (CoeffArray) Array containing the coefficients indexed such that the k-th coefficient for transition
            i --> j is CoeffArray[i, j, k].
        """
        return _calc_coeff(self, force, verbose=verbose)

    def get_cond(self, i: int, j: int, normal: bool = False, force: bool = False, verbose: bool = True) -> Callable[
        [float | ndarray], float]:
        """
        Return a callable for the frequency-dependent conductivity.

        :param i: (int) Index i
        :param j: (int) Index j
        :param normal: (bool) Normalize result
        :param force: (bool) Force recalculation
        :param verbose: (bool) Print verbose output

        :return: (Callable[[float | ndarray], float])
        """
        if self.coeff is None or force:
            self.calc_coeff(force=force, verbose=verbose)

        if not self.zero_index:
            i -= 1
            j -= 1

        omega = Symbol("omega", real=True, positive=True)
        coeff = self.coeff
        vals = self.weq.vals if self.is_symbolic else self.weq.nvals

        cond = sum(
            coeff[j, i, k] * (1 if k == 0 else (vals[k] / (1j * omega - vals[k])))
            for k in range(self.dim))

        if self.is_symbolic:
            cond = deep_simp(cond)
        if normal:
            norm_val = sum(
                coeff[j, i, k] * (1 if k == 0 else -1)
                for k in range(self.dim))
            if self.is_symbolic:
                norm_val = deep_simp(norm_val)
            return lambdify([omega], deep_simp(cond / norm_val))
        return lambdify([omega], cond)

    def get_conds(self,
                  conds: Union[tuple[int, int], list[tuple[int, int]], None] = None,
                  normal: bool = False,
                  force: bool = False,
                  verbose: bool = True) -> list[tuple[tuple[int, int], Callable[[float], float]]]:
        """
        Return a list of conductivity callables for specified index pairs.

        :param conds: (Union[tuple[int, int], list[tuple[int, int]], None]) Index pairs (i, j) encoding the transition
            i --> j.
        :param normal: (bool) Normalize results
        :param force: (bool) Force recalculation
        :param verbose: (bool) Print verbose output

        :return: (list[tuple[tuple[int, int], Callable[[float], float]]]) list containing tuples (ind, cond), with cond
            the conductivity of the transition i --> j.
        """
        if conds is None:
            self.calc_weq()
            self.calc_w1()
            conds = [(i, j) for i in range(self.dim) for j in range(i + 1, self.dim)
                     if any(elem != 0 for elem in (self.weq[i, j], self.weq[j, i], self.w1[i, j], self.w1[j, i]))]

        elif isinstance(conds[0], int):
            conds = [conds]

        return [(cond, self.get_cond(*cond, normal=normal, force=force, verbose=verbose)) for cond in conds]

    def to_num(self):
        """Convert internal data to numeric form."""
        if not self.is_symbolic:
            pass
        else:
            if self.weq is not None:
                self.weq.to_num()
            if self.w1 is not None:
                self.w1.to_num()
            if self.coeff is not None:
                self.coeff.to_num()
            self.is_symbolic = False

    def to_sym(self):
        """Convert internal data to symbolic form."""
        if self.is_symbolic:
            pass
        else:
            if self.weq is not None:
                self.weq.to_sym()
            if self.w1 is not None:
                self.w1.to_sym()
            self.is_symbolic = True


def _calc_coeff(w: WMatrix, force: bool = False, verbose: bool = True) -> CoeffArray:
    """
    Wrapper to calculate coefficients based on symbolic/numeric status.

    :param w: (WMatrix) Matrix instance
    :param force: (bool) Force recalculation
    :param verbose: (bool) Print verbose output

    :return: (CoeffArray) Coefficients
    """
    w.calc_weq(force=force, verbose=verbose)
    w.weq.calc_eig(force=force, verbose=verbose)
    w.calc_w1(force=force, verbose=verbose)

    return _sym_calc_coeff(w) if w.is_symbolic else _num_calc_coeff(w)


def _sym_calc_coeff(w: WMatrix) -> CoeffArray:
    """
    Calculate symbolic coefficients for conductivity expansion.

    :param w: (WMatrix) Matrix instance

    :return: (CoeffArray) Coefficients
    """
    weq = w.weq
    peq, vals, vecs = weq.peq, weq.vals, weq.vecs

    w1 = w.w1

    p1coeffs = [-inner(vecs[k], w1 * peq, peq) / vals[k] for k in range(1, len(vals))]

    coeffs = CoeffArray(w.dim)
    coeffs[:, :, 0] = calc_curr_like(w1, peq)
    for k in range(1, w.dim):
        coeffs[:, :, k] = p1coeffs[k - 1] * calc_curr_like(weq, vecs[k])

    w.coeff = coeffs
    return coeffs


def _num_calc_coeff(w: WMatrix) -> CoeffArray:
    """
    Calculate numeric coefficients for conductivity expansion.

    :param w: (WMatrix) Matrix instance

    :return: (CoeffArray)
    """
    weq = w.weq
    w1 = w.w1
    Peq, nvals, nvecs = weq.peq, weq.nvals, weq.nvecs

    # Calculate Coeffs of Pad expanded in eigenvecs
    p1coeffs = np_sum(nvecs[:, 1:].T * matmul(w1.nmat, Peq.nmat) / Peq.nmat, axis=1) / nvals[1:]

    # Calculate coefficients and store in array
    coeffs = CoeffArray(w.dim)
    coeffs[:, :, 0] = calc_curr_like(w1, Peq)
    for k in range(1, w.dim):
        coeffs[:, :, k] = p1coeffs[k - 1] * calc_curr_like(weq, nvecs[k])
    w.coeff = coeffs
    return coeffs
