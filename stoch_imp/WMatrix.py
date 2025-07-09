from tabnanny import verbose
from typing import Callable, Optional, Union

from numpy import matmul, ndarray
from numpy import sum as np_sum
from sympy import diff, lambdify, Symbol, pretty

from .EqMatrix import EqMatrix
from .core_classes import CoeffArray, TrMatrix, ConstantMatrix
from .util import calc_curr_like, deep_simp, inner


class WMatrix(TrMatrix):
    """
    Class handling driven transition matrices.

    :ivar mat: Internal sympy.Matrix representation of the transition matrix.
    :ivar dim: Dimension of the system.
    :ivar iter: Iterator over matrix indices.
    :ivar ds: Symbol representing the driving parameter.
    :ivar eq: Value at which to evaluate equilibrium (default is 0).
    :ivar weq: Cached equilibrium matrix (EqMatrix), or None if not yet computed.
    :ivar w1: First-order derivative matrix (ConstantMatrix), or None if not yet computed.
    :ivar coeff: Coefficient array for conductivity calculations, or None if not yet computed.
    :ivar tol: Numerical tolerance used for validations.
    :ivar is_symbolic: Boolean indicating whether the matrix is in symbolic form.
    :ivar zero_index: Boolean indicating whether 0-based indexing is used.

    """

    def __init__(self, *args, ds: Optional[Symbol] = None, eq: float = 0, **kwargs):
        """
        Initialize a WMatrix instance.

        :param args: Additional positional arguments for TrMatrix
        :param ds: Driving symbol for differentiation
        :type ds: Optional[Symbol]
        :param eq: Equilibrium point
        :type eq: float
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

    def calc_weq(self, eq: float = None, **flags) -> EqMatrix:
        """
        Calculate the equilibrium matrix.

        :param eq: Equilibrium value
        :type eq: float
        :param force: Force recalculation even if already computed
        :type force: bool
        :param verbose: Print verbose output
        :type verbose: bool

        :return: the equilibrium matrix.
        :rtype: EqMatrix
        """
        force = flags.setdefault('force', False)
        verbose = flags.setdefault('verbose', False)

        if self.weq is not None and not force:
            return self.weq

        self.weq = EqMatrix(self.subs({self.ds: eq or self.eq}), zi=self.zero_index, sym=self.is_symbolic, tol=self.tol)

        if verbose:
            print("Equilibrium matrix calculated")

        return self.weq

    def calc_w1(self, eq: float = None, **flags) -> ConstantMatrix:
        """
        Calculate the first-order driving matrix.

        :param eq: Equilibrium value
        :type eq: float
        :param force: Force recalculation
        :type force: bool
        :param verbose: Print verbose output
        :type verbose: bool

        :return: the first-order driving matrix.
        :rtype: ConstantMatrix
        """
        force = flags.setdefault('force', False)
        verbose = flags.setdefault('verbose', False)

        if self.w1 is not None and not force:
            return self.w1

        eq = eq if eq is not None else self.eq

        if self.ds not in self.free_symbols:
            raise AttributeError("Driving symbol not found in matrix")

        self.w1 = ConstantMatrix(diff(self, self.ds).subs({self.ds: eq}), zi=self.zero_index, sym=self.is_symbolic, tol=self.tol)

        if verbose:
            print("Driving matrix calculated")
        return self.w1

    def calc_coeff(self, **flags) -> CoeffArray:
        """
        Calculate coefficient array.

        :param force: Force recalculation
        :type force: bool
        :param verbose: Print verbose output
        :type verbose: bool

        :return: Array containing the coefficients indexed such that the k-th coefficient for transition
            i --> j is CoeffArray[i, j, k].
        :rtype: CoeffArray
        """
        if 'force' not in flags.keys():
            flags['force'] = False

        if verbose not in flags.keys():
            flags['verbose'] = True

        self.calc_weq(**flags)
        self.weq.calc_eig(**flags)
        self.calc_w1(**flags)

        return _sym_calc_coeff(self) if self.is_symbolic else _num_calc_coeff(self)

    def get_cond(self, i: int, j: int, normal: bool = False, **flags) -> Callable[[Union[float, ndarray]], float]:
        """
        Return a callable for the frequency-dependent conductivity.

        :param i: Index i
        :type i: int
        :param j: Index j
        :type j: int
        :param normal: Normalize result
        :type normal: bool
        :param force: Force recalculation
        :type force: bool
        :param verbose: Print verbose output
        :type verbose: bool

        :return: Callable returning conductivity value
        :rtype: Callable[[float | ndarray], float]
        """

        if 'force' not in flags.keys():
            flags['force'] = False

        if verbose not in flags.keys():
            flags['verbose'] = True

        if self.coeff is None or flags['force']:
            self.calc_coeff(**flags)

        if not self.zero_index:
            i -= 1
            j -= 1

        omega = Symbol("omega", real=True, positive=True)
        coeff = self.coeff
        vals = self.weq.vals if self.is_symbolic else self.weq.nvals

        cond = sum(
            coeff[j, i, k] * (1 if k == 0 else (-vals[k] / (1j * omega - vals[k])))
            for k in range(self.dim))

        if self.is_symbolic:
            cond = deep_simp(cond)
        if flags.setdefault('normal', False):
            norm_val = sum(
                coeff[j, i, k] * (1 if k == 0 else -1)
                for k in range(self.dim))
            if self.is_symbolic:
                norm_val = deep_simp(norm_val)
            return lambdify([omega], deep_simp(cond / norm_val))
        return lambdify([omega], cond)

    def get_conds(self,
                  conds: Union[tuple[int, int], list[tuple[int, int]], None] = None,
                  **flags) -> list[tuple[tuple[int, int], Callable[[float], float]]]:
        """
        Return a list of conductivity callables for specified index pairs.

        :param conds: Index pairs (i, j) encoding the transition i --> j.
        :type conds: Union[tuple[int, int], list[tuple[int, int]], None]
        :param normal: Normalize results
        :type normal: bool
        :param force: Force recalculation
        :type force: bool
        :param verbose: Print verbose output
        :type verbose: bool

        :return: list containing tuples (ind, cond), with cond the conductivity of the transition i --> j.
        :rtype: list[tuple[tuple[int, int], Callable[[float], float]]]
        """
        if conds is None:
            self.calc_weq()
            self.calc_w1()
            conds = [(i, j) for i in range(self.dim) for j in range(i + 1, self.dim)
                     if any(elem != 0 for elem in (self.weq[i, j], self.weq[j, i], self.w1[i, j], self.w1[j, i]))]

        elif isinstance(conds[0], int):
            conds = [conds]

        return [(cond, self.get_cond(*cond, **flags)) for cond in conds]

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
            if self.coeff is not None:
                self.coeff.to_sym()
            self.is_symbolic = True


def _sym_calc_coeff(w: WMatrix) -> CoeffArray:
    """
    Calculate symbolic coefficients for conductivity expansion.

    :param w: Matrix instance
    :type w: WMatrix

    :return: Coefficients
    :rtype: CoeffArray
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

    :param w: Matrix instance
    :type w: WMatrix

    :return: CoeffArray
    :rtype: CoeffArray
    """
    weq = w.weq
    w1 = w.w1
    Peq, nvals, nvecs = weq.peq, weq.nvals, weq.nvecs

    # Calculate Coeffs of Pad expanded in eigenvecs
    p1coeffs = np_sum(nvecs[:, 1:].T * matmul(w1.nmat, Peq.nmat).T / Peq.nmat.T, axis=1) / nvals[1:]

    # Calculate coefficients and store in array
    coeffs = CoeffArray(w.dim)
    coeffs[:, :, 0] = calc_curr_like(w1, Peq)
    for k in range(1, w.dim):
        coeffs[:, :, k] = p1coeffs[k - 1] * calc_curr_like(weq, nvecs[:, k])
    w.coeff = coeffs
    return coeffs
