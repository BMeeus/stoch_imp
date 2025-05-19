from pyplot_funcs import *

import sympy as sp
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


        """
        Calculate coefficients and save to npz file
        """
        print(n, sd)
        calculate_coeffs(weq, w1, f"{foldername}/{name}")

fig, ax = plt.subplots()

range_arr = []
for name in [f"{i}_sites_0" for i in range(2, 51, 2)]:
    out = np.load(f"n_sites_lin/{name}.npz")

    vals = out["vals"]
    coeffs = out["coeffs"]
    N = len(coeffs[:, 0, 0]) - 1

    om_arr = np.linspace(0, 300, 10000)
    x_res = np.real(ncond(om_arr, N, 0))
    range_arr.append(max(x_res) - min(x_res))


ax.plot(range(2, 51, 2), range_arr, marker="o", label=f"sd={0}")

range_arr = []
for name in [f"{i}_sites_0.5" for i in range(2, 51, 2)]:
    out = np.load(f"n_sites_lin/{name}.npz")

    vals = out["vals"]
    coeffs = out["coeffs"]
    N = len(coeffs[:, 0, 0]) - 1

    om_arr = np.linspace(0, 300, 10000)
    x_res = np.real(ncond(om_arr, N, 0))
    range_arr.append(max(x_res) - min(x_res))


ax.plot(range(2, 51, 2), range_arr, marker="o", label=f"sd=0.5")

range_arr = []
for name in [f"{i}_sites_1" for i in range(2, 51, 2)]:
    out = np.load(f"n_sites_lin/{name}.npz")

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