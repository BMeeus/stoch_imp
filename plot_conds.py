from pyplot_funcs import *

import numpy as np
import matplotlib.pyplot as plt


out = np.load("loop.npz")

vals = out["vals"]
coeffs = out["coeffs"]


def cond(om, i, j):
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    return c


fig, ax = plt.subplots()

for c in [[1, 0], [2, 1], [3, 2], [0, 3], [2, 0]]:
    c_i, c_j = c
    om_arr = np.linspace(0, 300, 10000)
    res_arr = cond(om_arr, c_i, c_j)
    line = ax.plot(np.real(res_arr), np.imag(res_arr))[0]
    add_arrow(line)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})

complex_axes(ax, r"\frac{\sigma_{mn}}{\sigma(0)}")
ax.tick_params(labelfontfamily="serif")

plt.show()
