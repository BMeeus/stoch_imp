import sympy as sp
from sympy import NonSquareMatrixError, ShapeError


class Mat:
    def __init__(self, arr):
        m = arr_to_mat(arr)
        self.mat = m
        self.dim = self.mat.cols

    def __str__(self):
        return sp.pretty(self.mat)

    def __setitem__(self, key, value):
        self.mat[*key] = value


class TrMatrix(Mat):
    def __init__(self, arr, zi=False):
        if type(arr) == int:
            if arr == 1:
                raise ShapeError("Transfer matrix must be at least (2, 2) dimensional")
            else:
                arr = sp.zeros(arr, arr)
        super().__init__(arr)
        self.zero_index = zi

    def add_trans(self, i, j, r=1.0, ri=None, symm=True):
        if not self.zero_index:
            i -= 1
            j -= 1

        self.mat[j, i] = r
        self.mat[i, i] -= r

        if symm:
            if not ri:
                ri = r ** (-1)
            self.mat[i, j] = ri
            self.mat[j, j] -= ri

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
        return TrMatrix(sp.diff(self.mat, self.ds).subs({self.ds: eq}))


class EqMatrix(TrMatrix):
    def __init__(self, arr, zi):
        super().__init__(arr, zi)


def arr_to_mat(arr, driven=False, ds=None):
    m = sp.Matrix(arr)

    if not m.is_square:
        raise NonSquareMatrixError("W matrix must be square")

    if m.rows == 1:
        raise ShapeError("W matrix must be at least (2, 2) dimensional")

    _check_diag(m, err=True)
    _check_rates(m, err=True)

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


if __name__ == '__main__':
    w = WMatrix(3)
    w.add_trans(2, 3, 1)
    w.add_trans(1, 3, 3)
    print(w)