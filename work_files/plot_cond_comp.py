from pyplot_funcs import *

import numpy as np
import matplotlib.pyplot as plt


def cond(om, i, j):
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    return c


name = "loop"

fig, ax = plt.subplots(2, 1)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})


out = np.load(f"np_files/coeff_files/coeff_{name}_no.npz")

vals = out["vals"]
coeffs = out["coeffs"]


for curr in [[0, 1], [1, 2], [2, 3], [3, 0]]:
    c_i, c_j = curr
    om_arr = np.linspace(0, 300, 10000)
    res_arr = cond(om_arr, c_i, c_j)
    line = ax[0].plot(np.real(res_arr), np.imag(res_arr), label=f"${c_i+1}\\to{c_j+1}$")[0]
    add_arrow(line)

complex_axes(ax[0], r"\frac{\sigma_{mn}}{\sigma(0)}")
ax[0].tick_params(labelfontfamily="serif")
ax[0].legend()



out = np.load(f"np_files/coeff_files/coeff_{name}.npz")

vals = out["vals"]
coeffs = out["coeffs"]


for curr in [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]:
    c_i, c_j = curr
    om_arr = np.linspace(0, 300, 10000)
    res_arr = cond(om_arr, c_i, c_j)
    line = ax[1].plot(np.real(res_arr), np.imag(res_arr), label=f"${c_i+1}\\to{c_j+1}$")[0]
    add_arrow(line)

complex_axes(ax[1], r"\frac{\sigma_{mn}}{\sigma(0)}")
ax[1].tick_params(labelfontfamily="serif")
ax[1].legend()



plt.show()