import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from stoch_imp import *

n = 3
mu = sp.Symbol("mu", real=True)
muEq = 0

w = WMatrix(n+1, zi=True, eq=muEq, ds=mu)

for i in range(1, n):
    w.add_trans(i, i+1)

w.add_trans(0, n, 1/2, ri=1/2)
w.add_trans(0, 1, 1/(1+sp.exp(-mu)), ri=1-1/(1+sp.exp(-mu)))

fig, ax = plt.subplots()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})

# Define range of frequencies
om_arr = np.logspace(-10, 2, 10000)

# for curr in [[i, (i+1)%(n+1)] for i in range(n+1)]:
#     c_i, c_j = curr  # Extract transition
#     cond = w.get_cond(c_i, c_j)  # calculate conductance
#     res_arr = cond(om_arr)
#     # plot result
#     line = ax.plot(np.real(res_arr), np.imag(res_arr), label=f"${c_i}\\to{c_j}$")[0]
#     add_arrow(line)

for ind, cond in w.get_conds():
    res_arr = cond(om_arr)
        # plot result
    line = ax.plot(np.real(res_arr), np.imag(res_arr), label="${}\\to{}$".format(*ind))[0]
    add_arrow(line)

# Add axes
complex_axes(ax, r"\sigma_{mn}(\omega)")

# Polish figure
ax.tick_params(labelfontfamily="serif")
ax.set_title("Periodically driven current", fontname="serif", fontsize=18)
# ax.legend()
plt.tight_layout()

plt.show()