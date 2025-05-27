import sympy as sp
from sympy import NonSquareMatrixError, ShapeError
from itertools import product


class Mat:
    """
    Top level class for all matrix like objects (Vectors, WMatrix, EqMatrix, ...)
    """
    def __init__(self, arr, zi=False):
        m = arr_to_mat(arr)
        self.mat = m
        self.dim = self.mat.cols
        self.iter = product(range(self.mat.rows), range(self.mat.cols))
        self.zero_index = zi                                             # Whether the system starts at 0

    def __str__(self):
        return sp.pretty(self.mat)

    def __setitem__(self, key, value):
        if type(key) == int:
            self.mat[key] = value
        else:
            self.mat[*key] = value

    def __iter__(self):
        return iter(self.mat)

    def __getitem__(self, item):
        if type(item) == int:
            return self.mat[item]
        else:
            return self.mat[*item]

    def __sub__(self, other):
        return Mat(self.mat - other.mat)

class TrMatrix(Mat):
    def __init__(self, arr, zi=False):
        if type(arr) == int:
            if arr == 1:
                raise ShapeError("Transfer matrix must be at least (2, 2) dimensional")
            else:
                arr = sp.zeros(arr, arr)
        super().__init__(arr, zi)
        _check_diag(self.mat, err=True)
        _check_rates(self.mat, err=True)

    def add_trans(self, i, j, r=1.0, ri=None, symm=True, simp=True):
        if not self.zero_index:
            i -= 1
            j -= 1

        if simp:
            r = sp.nsimplify(r, rational=True)
        self.mat[j, i] = r
        self.mat[i, i] -= r

        if symm:
            if not ri:
                ri = r ** (-1)
            elif simp:
                ri = sp.nsimplify(ri, rational=True)
            self.mat[i, j] = ri
            self.mat[j, j] -= ri
        if simp:
            sp.simplify(self.mat)

    def calc_diag(self):
        for col in range(self.dim):
            self.mat[col, col] = - sum([self.mat[row, col] for row in range(self.dim) if row != col])

    def check_diag(self):
        return _check_diag(self.mat)

    def check_rates(self, verbose=False):
        return _check_rates(self.mat, verbose=verbose)


class WMatrix(TrMatrix):
    def __init__(self, arr, ds=None, zi=False):
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

    def weq(self, eq=0):
        return EqMatrix(self.mat.subs({self.ds: eq}), zi=self.zero_index)

    def w1(self, eq=0):
        if self.ds not in list(self.mat.free_symbols):
            raise AttributeError("Driving symbol not found in matrix")
        return Mat(sp.diff(self.mat, self.ds).subs({self.ds: eq}))


class EqMatrix(TrMatrix):
    def __init__(self, arr, zi=False):
        super().__init__(arr, zi)

    def peq(self):
        spc = self.mat.eigenvects(error_when_incomplete=True)
        try:
            nspace = [sp for sp in spc if sp[0] == 0][0]
        except IndexError:
            raise ValueError("no eigenvalue 0 was found")

        if nspace[1] > 1:
            raise ValueError("multiple steady states found")
        else:
            p = nspace[-1][0]
            return Mat(p/sum(p))

    def check_db(self, tol=10**-15):
        peq = self.peq()

        db_mat = curr_from_arr(self, peq)

        for el in self.iter:
            if db_mat[*el] > tol:
                return False
        return True



def arr_to_mat(arr):
    m = sp.Matrix(arr)

    if (m.rows == 1) ^ (m.cols == 1):
        if m.cols > 1:
            return m.T
        else:
            return m
    elif not m.is_square:
        raise NonSquareMatrixError("Matrix object must be square or vector, is neither: ({}, {})".format(m.rows, m.cols))
    elif m.rows == 1:
        raise ShapeError("Matrix must be at least (2, 2) dimensional")

    return m


def _check_diag(m, err=False):
    for col in range(m.cols):
        if sum(m.col(col)) != 0:
            if err:
                raise ValueError("Column {} does not sum to 0 (sum = {})".format(col, sum(m.col(col))))
            else:
                return False
    return True


def _check_rates(m, err=False, verbose=False):
    for col in range(m.cols):
        for row in range(m.rows):
            try:
                if row != col and m[row, col] < 0:
                    if err:
                        raise ValueError(
                            "Transfer rate ({} -> {}) is negative ({})".format(col + 1, row + 1, m[row, col]))
                    else:
                        return False
            except TypeError as e:
                if verbose:
                    print("Positivity of transfer rate ({} -> {}) undetermined ({})".format(col + 1, row + 1,
                                                                                            m[row, col]))
    return True

def curr_from_arr(m, p):
    m = m.mat
    curr1 = Mat([[m[row, col] * p[row] for col in range(m.cols)] for row in range(m.rows)])
    print(curr1)
    curr2 = Mat([[m[row, col] * p[col] for row in range(m.cols)] for col in range(m.rows)])
    print(curr2)
    return curr1 - curr2

if __name__ == '__main__':
    weq = EqMatrix([[-1 , 0 , 1 ],
                    [ 1 ,-1 , 0 ],
                    [ 0 , 1 ,-1 ]])
    peq = weq.peq()

    print(weq.check_db())