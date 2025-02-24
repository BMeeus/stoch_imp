import sympy as sp

curr = [0, 1]

weq = sp.Matrix([[0, 1], [2, 3]])

print(weq[*curr])