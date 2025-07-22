import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import stoch_imp as si

n = 4                            # Number of sites (excl. basins)
name = f"{n}_loop"
foldername = "n_sites_loop"
F = sp.symbols("F", real=True)   # variable driving force
FEq = 0                          # base value of driving

t = 1.0

w = si.WMatrix(n, ds=F)

for i in range(n):
    w.add_trans(i, i + 1, t * sp.exp(F))

w.add_trans(1, 3, t * sp.exp(F))


with si.create_fig(1) as (fig, ax):

    # Define range of frequencies
    om_arr = np.logspace(-10, 2, 10000)

    for ind, cond in w.get_conds([(1, 2), (3, 4), (1, 3)]):
        res_arr = cond(om_arr)
        # plot result
        line = ax.plot(np.real(res_arr), np.imag(res_arr), label="${}\\to{}$".format(*ind))[0]
        si.add_arrow(line)

    # Add axes
    si.complex_axes(ax, r"\sigma_{mn}(\omega)")

    ax.set_title("Periodically driven current", fontname="serif", fontsize=18)
    ax.legend()