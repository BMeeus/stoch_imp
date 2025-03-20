import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def pprint(e):
    sp.pretty_print(e)
    return



F, A, B = sp.symbols("F, A, B", real=True)


W = sp.Matrix([[-2*A-B, B, B, A],
               [A, -A-B, B, 0],
               [A, A, -A-2*B, B],
               [B, 0, A, -A-B]])

Wr = W.rref()[0]

W = W.subs({A: sp.exp(F), B: sp.exp(-F)})

Wr = Wr.subs({A: sp.exp(F), B: sp.exp(-F)})
Wr = sp.simplify(Wr)



Pad = sp.Matrix([-Wr[0, 3], -Wr[1,3], -Wr[2, 3], 1])

Pad /= sum(Pad)

Pad = sp.simplify(Pad)

J = sp.simplify(W[1, 0] * Pad[1] - W[0, 1] * Pad[0])


J2 = (sp.exp(F) - sp.exp(-F))/4

J_f = sp.lambdify(F, J)
J2_f = sp.lambdify(F, J2)

plt.plot(np.linspace(-10, 10, 100), J_f(np.linspace(-10, 10, 100)))
plt.plot(np.linspace(-10, 10, 100), J2_f(np.linspace(-10, 10, 100)))
plt.show()

