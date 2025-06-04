from sympy import (diff, im, lambdify, Symbol, Matrix)

from .core_classes import (CoeffArray, Mat, TrMatrix)
from .util import (calc_curr_like, deep_symp, gram_schmidt, inner)


class EqMatrix(TrMatrix):
    """Class handling all Equilibrium Transfer Matrices."""
    def __init__(self, arr, zi: bool = False) -> None:
        """
        Initiate an instance of EqMatrix using an Array or int for an empty matrix.

        :param arr (Array like or Int): The Array with which the matrix is set. If arr is an integer, an empty
            (arr,arr) Sympy Matrix is used
        :param zi (Bool): Whether the system is zero-indexed. Can be useful in systems where there can be no particles.
        """
        super().__init__(arr, zi)
        self.peq = None
        self.vals = None
        self.vecs = None

    def calc_peq(self, verbose: bool = False) -> Mat:
        """Calculate the Equilibrium distribution"""
        self.calc_eig(verbose=verbose)
        return self.peq

    def calc_eig(self, tol: float = 10**(-15), verbose: bool = False) -> tuple[list, list[Matrix]]:
        """
        Calculate the eigensystem of the matrix. As a byproduct, peq is also calculated but not returned.

        :param tol: (float) The tolerance with which comparisons to zero are done
        :param verbose: (bool) If True, prints progress statements
        :return vals: The eigenvalues of the system, sorted in descending order and repeated according to multiplicity.
        :return vecs: The eigenvectors of the system, sorted such that the index of the vector matches the associated
        eigenvalue in vals.
        """
        eig_syst = self.mat.eigenvects(error_when_incomplete=True)

        # sort the system in descending order of eigenvalue
        eig_syst.sort(key=lambda x: x[0], reverse=True)

        if eig_syst[0][0] != 0:
            raise ValueError("no eigenvalue 0 was found")
        elif eig_syst[0][1] != 1:
            raise ValueError("multiple steady states found")

        # Calculate Equilibrium Distribution from first eigenvector
        self.peq = Mat(eig_syst[0][-1][0] / sum(eig_syst[0][-1][0]))
        if verbose:
            print("Equilibrium distribution calculated")

        if not self.check_db(tol=tol):
            raise ValueError("detailed balance not fulfilled")

        vals = []
        vecs = []

        # unzip eigenspaces and check if real and well-calculated
        for space in eig_syst:
            val = space[0]
            if im(val) > tol:
                raise ValueError("Complex eigenvalue found: {}".format(val))
            for degen in range(space[1]):
                vec = space[-1][degen]
                if (self.mat * vec - val * vec).norm() > tol:
                    raise ValueError("Incorrect computation of eigenvectors")

                vals.append(val)
                vecs.append(vec)

        # orthonormalize eigenvectors
        vecs = gram_schmidt(vecs, self.peq)
        self.vals = vals
        self.vecs = vecs

        if verbose:
            print("Eigensystem calculated")
        return self.vals, self.vecs

    def check_db(self, tol: float = 10 ** -15) -> bool:
        """Check if detailed balance is satisfied up to specified tolerance"""
        if self.peq is None:
            self.calc_peq()

        db_mat = calc_curr_like(self, self.peq)

        for el in self.iter:
            if db_mat[*el] > tol:
                return False
        return True


class WMatrix(TrMatrix):
    """Class handling driven transition matrices"""
    def __init__(self, arr, ds: Symbol = None, eq: float = 0, zi: bool = False):
        """
        Initialise an instance of the WMatrix class. It is important that a driving symbol is given. If it is not given,
        an attempt is made to find one. If this is not successful or the result is ambiguous, an error is thrown.

        :param arr (Array like or Int): The Array with which the matrix is set. If arr is an integer, an empty
            (arr,arr) Sympy Matrix is used.
        :param ds (Symbol): The symbol containing the driving of the system. If it is not given, one will try to be inferred.
        :param eq (float): The equilibrium value of the driving. Default is 0.
        :param zi (Bool): Whether the system is zero-indexed. Can be useful in systems where there can be no particles.
        """
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

        self.w1 = Mat(diff(self.mat, self.ds).subs({self.ds: eq}), zi=self.zero_index)
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

        om = Symbol("omega", real=True, positive=True)
        c = deep_symp(
            sum([self.coeff[j, i, k] * (1 if k == 0 else (self.weq.vals[k] / (1j * om - self.weq.vals[k]))) for k in
                 range(self.dim)]))
        if normal:
            n = deep_symp(sum([self.coeff[j, i, k] * (1 if k == 0 else -1) for k in range(self.dim)]))
            return lambdify([om], deep_symp(c / n))
        return lambdify([om], c)

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

