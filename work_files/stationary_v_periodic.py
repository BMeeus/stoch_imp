from pyplot_funcs import *

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def cond(om, i, j, normal=False):
    """
    Calculate the conductance of the transition i --> j. This can be normalised using the conductance at zero driving.

    :param om: Float/np array: The driving frequency or array of frequencies
    :param i: Int: The site of origin of the transition
    :param j: Int: The destination site of the transition
    :param normal: Bool: if True returns the conductance normalised using conductance at zero frequency

    :return: Float/np array: The conductance of the transition i --> j.
    """
    c = sum([coeffs[i, j, k] * (1 if k == 0 else (vals[k] / (1j * om - vals[k]))) for k in range(len(coeffs[0, 0, :]))])
    if normal:
        n = sum([coeffs[i, j, k] * (1 if k == 0 else -1) for k in range(len(coeffs[0, 0, :]))])
        return c / n
    else:
        return c


F, A, B = sp.symbols("F, A, B", real=True)


W = sp.Matrix([[-2*A-B, B   , B     , A   ],
               [A     , -A-B, B     , 0   ],
               [A     , A   , -A-2*B, B   ],
               [B     , 0   , A     , -A-B]])

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


fig, ax = plt.subplots(1, 2, figsize=(14, 6))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})

ax[0].plot(np.linspace(-10, 10, 100), J2_f(np.linspace(-10, 10, 100)), c="k")
ax[0].plot(np.linspace(-10, 10, 100), J_f(np.linspace(-10, 10, 100)))

ax[0].axhline(0, c="k")
ax[0].axvline(0, c="k")

ax[0].tick_params(labelfontfamily="serif")

ax[0].set_title("(a) Stationary current", fontname="serif", fontsize=18)
ax[0].set_ylim([-6000, 6000])
ax[0].text(10, 100, r"$F$", wrap=True, fontsize=18)
ax[0].text(0.5, 5000, "$J_{21}$", wrap=True, fontsize=18)
"""
Define names and locations
"""
name = "4_loop"                   # Name and folder of data
foldername = "n_sites_loop"

figfolder = "figs_stoch_imp"      # destination name and folder of figure
figname = "loop_no_inset"

save = False                      # Save the figure
local = True                      # If True figure is saved in project. Otherwise, specify entire path!

currents = [[0, 1], [2, 3], [0, 2]]





"""
Read out data, detect transitions if necessary
"""

out = np.load(f"../{foldername}/{name}.npz")

vals = out["vals"]
coeffs = out["coeffs"]
weq = out["weq"]
w1 = out["w1"]


# Define range of frequencies
om_arr = np.linspace(0, 300, 10000)

for curr in currents:
    c_i, c_j = curr  # Extract transition
    res_arr = cond(om_arr, c_i, c_j)  # calculate conductance
    # plot result
    line = ax[1].plot(np.real(res_arr), np.imag(res_arr), label=f"${c_i}\\to{c_j}$")[0]
    add_arrow(line)

# Add axes
ax[1].set_xlim([0.2, 0.65])
ax[1].set_ylim([-0.2, 0.2])
complex_axes(ax[1], r"\sigma_{mn}(\omega)", sz=16, x_off=[-0.04, 0])


# asp2 = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
# fact = np.diff(ax[1].get_ylim())[0] / np.diff(ax[0].get_ylim())[0]
#
# ax[0].set_aspect(asp2 * fact)


# Polish figure
ax[1].tick_params(labelfontfamily="serif")
ax[1].set_title("(b) Periodically driven current", fontname="serif", fontsize=18)

# ax.legend()
plt.tight_layout()

plt.savefig(f"C:\\Users\\lucp13819\\Pictures\\figs_stoch_imp\\stat_period.png", dpi=600)
plt.show()
