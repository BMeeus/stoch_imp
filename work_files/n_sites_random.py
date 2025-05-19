import sympy as sp
import numpy as np

from calculate_coeffs import calculate_coeffs

foldername = "n_sites_lin"


def pprint(e):
    sp.pretty_print(e)
    return

def laprint(e):
    print(r"\begin{equation}")
    print(sp.latex(e))
    print(r"\end{equation}", end="\n\n\n")
    return

n = 50

mu = sp.symbols("mu", real=True)


mu_L = mu
mu_R = 0

for n in range(2, 51, 2):
    for sd in [0, 0.5, 1]:
        name = f"{n}_sites_{sd}"
        en_arr = np.random.normal(0, sd, n+2)
        """
        Build matrix W_base based on sites
        """

        wbase = sp.zeros(n + 2, n + 2)

        wbase[0, 1] = 1 - 1 / (1 + sp.exp(en_arr[1] - mu))
        wbase[1, 0] = 1 / (1 + sp.exp(en_arr[1] - mu))

        wbase[-2, -1] = 1 - 1 / (1 + sp.exp(en_arr[-2]))
        wbase[-1, -2] = 1 / (1 + sp.exp(en_arr[-2]))

        for i in range(1, n):
            wbase[i, i + 1] = sp.exp(- (en_arr[i] - en_arr[i + 1]) / 2)
            wbase[i + 1, i] = sp.exp(- (en_arr[i + 1] - en_arr[i]) / 2)

        w = wbase[:-1, :-1]
        w[-1, 0] = wbase[-1, -2]
        w[0, -1] = wbase[-2, -1]

        for col in range(n + 1):
            w[col, col] = -sum(w[:, col])

        weq = np.array(sp.N(w.subs({mu: 0})), dtype=np.float64)           # Equilibrium transfer matrix

        w1 = np.array(sp.N(sp.diff(w, mu).subs({mu: 0})), dtype=np.float64)  # First term in taylor expansion
        print(weq)
        print(w1)

        """
        Calculate coefficients and save to npz file
        """
        print(n, sd)
        calculate_coeffs(weq, w1, f"{foldername}/{name}")
