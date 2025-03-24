from pyplot_funcs import *

import numpy as np
import matplotlib.pyplot as plt


figfolder = "figs_stoch_imp"
figname = "range_different_lengths"


def cond(om, i, j):
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    return c

def ncond(om, i, j):
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    n = sum([coeffs[i, j, k] * (1 if k == 0 else -1) for k in range(len(coeffs[0, 0, :]))])
    return c/n

fig, ax = plt.subplots()

range_arr = []
for name in [f"{i}_sites" for i in range(2, 51, 2)]:
    out = np.load(f"np_files/coeff_files/coeff_{name}.npz")

    vals = out["vals"]
    coeffs = out["coeffs"]
    N = len(coeffs[:, 0, 0]) - 1

    om_arr = np.linspace(0, 300, 10000)
    x_res = np.real(ncond(om_arr, N, 0))
    range_arr.append(max(x_res) - min(x_res))


ax.plot(range(2, 51, 2), range_arr, marker="o", label=f"sd={0}")

range_arr = []
for name in [f"{i}_sites_sd_05" for i in range(2, 51, 2)]:
    out = np.load(f"np_files/coeff_files/coeff_{name}.npz")

    vals = out["vals"]
    coeffs = out["coeffs"]
    N = len(coeffs[:, 0, 0]) - 1

    om_arr = np.linspace(0, 300, 10000)
    x_res = np.real(ncond(om_arr, N, 0))
    range_arr.append(max(x_res) - min(x_res))


ax.plot(range(2, 51, 2), range_arr, marker="o", label=f"sd=0.5")

range_arr = []
for name in [f"{i}_sites_sd_1" for i in range(2, 51, 2)]:
    out = np.load(f"np_files/coeff_files/coeff_{name}.npz")

    vals = out["vals"]
    coeffs = out["coeffs"]
    N = len(coeffs[:, 0, 0]) - 1

    om_arr = np.linspace(0, 300, 10000)
    x_res = np.real(ncond(om_arr, N, 0))
    range_arr.append(max(x_res) - min(x_res))


ax.plot(range(2, 51, 2), range_arr, marker="o", label=f"sd=1")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})

plt.rcParams['text.latex.preamble'] = r'\usepackage{palatino, mathpazo}'
plt.rcParams["font.serif"] = ["Palatino"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"
ax.tick_params(labelfontfamily="serif")
ax.set_title("Range of conductivity for different chain lengths", fontfamily="serif")
ax.set_xlabel("Length $N$", fontfamily="serif", math_fontfamily="cm")
ax.set_ylabel("$\\frac{\mathrm{Range\;(Re\;} \sigma_{N}(\omega)\,)}{\sigma(0)}$", fontfamily="serif", math_fontfamily="cm", fontsize=15, rotation=0, labelpad=50)
ax.legend(loc="upper left")
ax.grid(True, which='both')
plt.tight_layout()
plt.savefig(f"C:\\Users\\lucp13819\\Pictures\\{figfolder}\\{figname}.png", dpi=600, bbox_inches='tight')
plt.show()

# TODO: Add current finding func