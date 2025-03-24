from pyplot_funcs import *

import numpy as np
import matplotlib.pyplot as plt


figfolder = "figs_stoch_imp"
figname = "range_different_chains"


def cond(om, i, j):
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    return c

def ncond(om, i, j):
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    n = sum([coeffs[i, j, k] * (1 if k == 0 else -1) for k in range(len(coeffs[0, 0, :]))])
    return c/n

fig, ax = plt.subplots()

for name in [f"{i}_sites" for i in [4, 8, 12, 16, 20]]:
    out = np.load(f"np_files/coeff_files/coeff_{name}.npz")

    vals = out["vals"]
    coeffs = out["coeffs"]
    N = len(coeffs[:, 0, 0]) - 1
    range_arr = np.zeros(N)
    om_arr = np.linspace(0, 300, 10000)

    for site in range(1, N+1):
        x_res = np.real(ncond(om_arr, site, (site + 1)%(N+1)))
        range_arr[site-1] = max(x_res) - min(x_res)


    ax.plot([(i+1)/N for i in range(N)], range_arr, marker="o", label=f"$N={N}$")




plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})

ax.tick_params(labelfontfamily="serif")
ax.set_title("Range of conductivity for different chain lengths", fontfamily="serif")
ax.set_xlabel("Position $\\frac{n}{N}$", fontfamily="serif", math_fontfamily="cm")
ax.set_ylabel("$\\frac{\mathrm{Range\;(Re\;} \sigma_{n, n+1}(\omega)\,)}{\sigma(0)}$", fontfamily="serif", math_fontfamily="cm", fontsize=15, rotation=0, labelpad=50)
ax.legend(loc="upper center")
ax.grid(True, which='both')
plt.tight_layout()
plt.savefig(f"C:\\Users\\lucp13819\\Pictures\\{figfolder}\\{figname}.png", dpi=600, bbox_inches='tight')
plt.show()

# TODO: Add current finding func