from sympy import (diff, im, lambdify, Symbol, Matrix, pretty)

from .core_classes import (CoeffArray, Mat, TrMatrix)
from .util import (calc_curr_like, deep_symp, gram_schmidt, inner)


class EqMatrix(TrMatrix):
    """Class handling all Equilibrium Transfer Matrices."""
    def __init__(self, arr, zi: bool = False) -> None:
        """
        Initiate an instance of EqMatrix using an Array or int for an empty matrix.

        :param arr: (array like or int) The Array with which the matrix is set. If arr is an integer, an empty
            (arr,arr) Sympy Matrix is used
        :param zi: (bool) Whether the system is zero-indexed. Can be useful in systems where there can be no particles.
        """
        super().__init__(arr, zi)
        self.peq = None
        self.vals = None
        self.vecs = None

    def calc_peq(self, force: bool = False, verbose: bool = False) -> Mat:
        """Calculate the Equilibrium distribution"""
        if (self.peq is None) or force:
            self.calc_eig(force=force, verbose=verbose)
        return self.peq

    def calc_eig(self, tol: float = 10**(-15), force: bool = False, verbose: bool = False) -> tuple[list, list[Matrix]]:
        """
        Calculate the eigensystem of the matrix. As a byproduct, peq is also calculated but not returned.

        :param tol: (float) The tolerance with which comparisons to zero are done.
        :param force: (bool) If True, the calculation is performed even if there is already a calculated Eigensystem
        :param verbose: (bool) If True, prints progress statements
        :return: (vals, vecs):
            vals ( list(Expr | float, ...) ): The eigenvalues of the system, sorted in descending order and repeated
            according to multiplicity.
            vecs ( list(Matrix, ...) ): The eigenvectors of the system, sorted such that the index of the vector matches
            the associated
            eigenvalue in vals.
        """
        if (self.vals is not None) and (self.vecs is not None) and (not force):
            return self.vals, self.vecs

        eig_syst = self.eigenvects(error_when_incomplete=True)

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
                if (self * vec - val * vec).norm() > tol:
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

        :param arr: (Array like or Int) The Array with which the matrix is set. If arr is an integer, an empty
            (arr,arr) Sympy Matrix is used.
        :param ds: (Symbol) The symbol containing the driving of the system. If it is not given, one will try to be inferred.
        :param eq: (float) The equilibrium value of the driving. Default is 0.
        :param zi: (Bool) Whether the system is zero-indexed. Can be useful in systems where there can be no particles.
        """
        super().__init__(arr, zi)

        # If only one free symbol is present, it is assumed this is the driving.
        if ds is None:
            symb_list = list(self.free_symbols)
            if not symb_list:
                raise AttributeError("No driving symbol found or given")
            elif len(symb_list) > 1:
                raise ValueError("Ambiguity in driving symbol, please provide a specific symbol")
            else:
                ds = symb_list[0]
                print(f"Driving symbol set to {pretty(ds)}")

        self.ds = ds
        self.weq = None
        self.w1 = None
        self.coeff = None
        self.eq = eq

    def calc_weq(self, eq: float = None, force: bool = False, verbose: bool = False) -> EqMatrix:
        """
        Calculate the Equilibrium Matrix for the WMatrix by substituting in the equilibrium value.

        :param eq: (float) The equilibrium value to be substituted. Default is the value attributed to the WMatrix.
        :param force: (bool) If True, the calculation is performed even if there is already a calculated
            Equilibrium matrix
        :param verbose:  (bool) If True, prints progress statements
        :return: (EqMatrix) The equilibrium matrix associated to the system
        """

        if (self.weq is not None) and (not force):
            return self.weq

        if eq is None:
            eq = self.eq

        self.weq = EqMatrix(self.subs({self.ds: eq}), zi=self.zero_index)

        if verbose:
            print("Equilibrium matrix calculated")

        return self.weq

    def calc_w1(self, eq: float = None, force: bool = False, verbose: bool = False) -> Mat:
        """
        Calculate driving matrix (the first Taylor expansion coefficient of the WMatrix).

        :param eq: (Float) The equilibrium value to be substituted. Default is the value attributed to the WMatrix.
        :param force: (bool) If True, the calculation is performed even if there is already a calculated
            Driving matrix
        :param verbose: (Bool) If True, prints progress statements
        :return: (Mat) The Driving Matrix
        """

        if (self.w1 is not None) and (not force):
            return self.w1

        if eq is None:
            eq = self.eq

        if self.ds not in list(self.free_symbols):
            raise AttributeError("Driving symbol not found in matrix")

        self.w1 = Mat(diff(self, self.ds).subs({self.ds: eq}), zi=self.zero_index)

        if verbose:
            print("Driving matrix calculated")

        return self.w1

    def calc_coeff(self, force: bool = False, verbose: bool = True) -> CoeffArray:
        """Calculate the coefficients A^k_mn for the WMatrix"""
        return _calc_coeff(self, force, verbose=verbose)

    def get_cond(self, i: int, j: int, normal: bool = False, force: bool = False, verbose: bool = True) -> callable:
        """
        Calculate the conductivity of transition i --> j, and return it as a callable function. The conductivity can be
        normalised so that it starts at 1.

        :param i: (Int) The start state of the transition
        :param j: (Int) The end state of the transition
        :param normal: (Bool) If True, the conductivity is normalised such that it starts at 1.
        :param force: (Bool) If False, previously stored results will be used for the calculations. If True, all
            matrices and eigenspaces will be recalculated.
        :param verbose: (Bool) If True, prints progress statements
        :return: (Callable) A function taking the driving frequency as input and outputting the conductivity of the
            transition
        """

        if (self.coeff is None) or force:
            self.calc_coeff(force=force, verbose=verbose)

        if not self.zero_index:
            i -= 1
            j -= 1

        om = Symbol("omega", real=True, positive=True)

        # A more clear form of this formula can be found in the accompanying pdf.
        c = deep_symp(
            sum([self.coeff[j, i, k] * (1 if k == 0 else (self.weq.vals[k] / (1j * om - self.weq.vals[k]))) for k in
                 range(self.dim)]))
        if normal:
            # This is just the above formula for zero driving.
            n = deep_symp(sum([self.coeff[j, i, k] * (1 if k == 0 else -1) for k in range(self.dim)]))
            return lambdify([om], deep_symp(c / n))
        return lambdify([om], c)

    def get_conds(self, conds: tuple[int, int] | list[tuple[int, int]] | None = None,
                  normal: bool = False,
                  force: bool = False,
                  verbose: bool = True) -> list[tuple[tuple[int, int], callable]]:
        """
        A helper function to calculate the conductivities for multiple transitions at a time. If no transitions are
        given, a simple heuristic is used to find all transitions. Returns a list of tuples containing the indices of the
        transition and the callable function returned by get_cond.

        :param conds: ( [(int, int), ... ] )A list of transitions i --> j, given in the form (i, j). If no transitions
            are given, the transitions are sought by looking for non-zero elements of the Equilibrium and Driving matrix.
            Note that this heuristic does not take into account the direction of the transition, so results may be
            mirrored.
        :param normal: (Bool) If True, the conductivity is normalised such that it starts at 1.
        :param force: (Bool) If False, previously stored results will be used for the calculations. If True, all
            matrices and eigenspaces will be recalculated.
        :param verbose: (Bool) If True, prints progress statements
        :return: Returns a list of tuples (ind, cond) which contains the indices of the transition and the associated
            conductivity function.
        """

        if conds is None:
            # Make sure weq and w1 are calculated
            self.calc_weq()
            self.calc_w1()
            conds = []
            # Iterate over lower triangle of matrix, if any relevant element is non-zero, adds the indices to conds
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    if any(elem != 0 for elem in (self.weq[i, j], self.weq[j, i], self.w1[i, j], self.w1[j, i])):
                        conds.append((i, j))

        elif type(conds[0]) == int:
            conds = [conds]

        return [(cond, self.get_cond(*cond, normal=normal, force=force, verbose=verbose)) for cond in conds]


def _calc_coeff(w: WMatrix, force: bool = False, verbose: bool = True) -> CoeffArray:
    """
    Calculates the coefficients A^k_mn for a WMatrix w. An explanation and mathematical justification can be found in
    the accompanying pdf file.

    :param w: (WMatrix) The WMatrix of which the coefficients are calculated. The only requirement is that it is filled
        in.
    :param force: (Bool) If False, previously stored results will be used for the calculations. If True, all
        matrices and eigenspaces will be recalculated.
    :param verbose: (Bool) If True, prints progress statements
    :return: (CoeffArray) A 3-dimensional array containing the coefficients A^k_mn, indexed as [n, m, k]. This ensures
        'on paper' layout of the matrix A^k_mn when printing [:, :, k].
    """
    # Ensure weq is calculated
    w.calc_weq(force=force, verbose=verbose)
    weq = w.weq

    # Ensure Eigensystem is calculated
    weq.calc_eig(force=force, verbose=verbose)
    peq = weq.peq
    vals = weq.vals
    vecs = weq.vecs

    # Ensure w1 is calculated
    w.calc_w1(force=force, verbose=verbose)
    w1 = w.w1

    # Calculate expansion coefficients of p1 in eigenvector basis
    p1coeffs = [-inner(vecs[k], w1 * peq, peq) / vals[k] for k in range(1, len(vals))]

    coeffs = CoeffArray(w.dim)
    for k in range(w.dim):
        if k == 0:
            coeffs[:, :, k] = calc_curr_like(w1, peq)
        else:
            coeffs[:, :, k] = (p1coeffs[k - 1] * calc_curr_like(weq, vecs[k]))

    w.coeff = coeffs
    return coeffs

