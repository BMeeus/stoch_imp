import numpy as np
import sympy as sp
import scipy as scp

class WMatrix:
    def __init__(self, n):
        self.dim = n
        self.mat = sp.zeros(n, n)

    def __str__(self):
        return sp.pretty(self.mat)

    def add_trans(self, i, j, r=1, ri=None, zero_index=False):
        if not zero_index:
            i -= 1
            j -= 1
        if not ri:
            ri = r**(-1)
        self.mat[j, i] = r
        self.mat[i, j] = ri

        self.mat[i, i] = - sum([self.mat[k, i] for k in range(self.dim) if k != i])
        self.mat[j, j] = - sum([self.mat[k, j] for k in range(self.dim) if k != j])



if __name__ == '__main__':
    w = WMatrix(3)
    w.add_trans(2, 3, 1)
    w.add_trans(1, 3, 3)
    print(w)