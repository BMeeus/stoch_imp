from pyplot_funcs import *

import numpy as np
import matplotlib.pyplot as plt

name = "loop"
figfolder = "figs_stoch_imp"
figname = "loop_no_inset"
out = np.load(f"np_files/coeff_files/coeff_{name}.npz")

vals = out["vals"]
coeffs = out["coeffs"]


def cond(om, i, j):
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    return c

def ncond(om, i, j):
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    n = sum([coeffs[i, j, k] * (1 if k == 0 else -1) for k in range(len(coeffs[0, 0, :]))])
    return c/n

fig, ax = plt.subplots()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})

for curr in [[0, 1], [2, 3], [0, 2]]:
    c_i, c_j = curr
    om_arr = np.linspace(0, 300, 10000)
    res_arr = cond(om_arr, c_i, c_j)
    line = ax.plot(np.real(res_arr), np.imag(res_arr), label=f"${c_i}\\to{c_j}$")[0]
    add_arrow(line)


complex_axes(ax, r"\sigma_{mn}(\omega)", x_off=[-0.015, 0.03], y_off=[0.2, 0], sz=12)
ax.tick_params(labelfontfamily="serif")


ax.set_title("Periodically driven current", fontname="serif", fontsize=18)
# ax.legend()
ax.set_xlim([0.2, 0.65])
plt.tight_layout()
plt.savefig(f"C:\\Users\\lucp13819\\Pictures\\{figfolder}\\{figname}.png", dpi=600, bbox_inches='tight')
plt.show()

# TODO: Add current finding func