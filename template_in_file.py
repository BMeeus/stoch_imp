from calculate_coeffs import calculate_coeffs
from pyplot_funcs import *

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import os

"""
Template file performing the calculation for a periodic of n sites with a shortcut added between site 1 and 3.
"""


"""
Simulation parameters
"""
n = 4                            # Number of sites (excl. basins)
name = f"{n}_loop"
foldername = "n_sites_loop"
F = sp.symbols("F", real=True)   # variable driving force
FEq = 0                          # base value of driving

t = 1                            # Transition rate between sites

figfolder = "figs_stoch_imp"      # destination name and folder of figure
figname = "loop_no_inset"

save = False                      # Save the figure
local = True                      # If True figure is saved in project. Otherwise, specify entire path!

currents = []

"""
Defining useful function, conductance
"""


def add_driven_trans(i, j, rate):
    """
    Adds a symmetric transition to matrix w

    :param i: Outgoing site
    :param j: Incoming site
    :param rate: rate of the transition
    :return: None
    """
    w[i, j] = rate * sp.exp(-F)
    w[j, i] = rate * sp.exp(F)
    return


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


"""
Build transfer matrix W
"""

w = sp.zeros(n, n)               # initialise base matrix

for i in range(n):
    add_driven_trans(i, (i + 1) % n, t)  # Add transition between consecutive sites, periodic bc

add_driven_trans(0, 2, t)  # Add shortcut between site 1 and 3

for i in range(n):
    w[i, i] = - sum(w[:, i])     # add diagonal terms

"""
Calculate Weq, W1
"""

weq = np.array(sp.N(w.subs({F: 0})), dtype=np.float64)             # Equilibrium transfer matrix

w1 = np.array(sp.N(sp.diff(w, F).subs({F: FEq})), dtype=np.float64)  # First term in taylor expansion

coeffs, vals, vecs = calculate_coeffs(weq, w1)

# Detect possible transitions from weq and w1
if not currents:
    for i in range(len(weq[:, 0])):
        for j in range(i+1, len(weq[:, 0])):
            if weq[i, j] != 0 or weq[j, i] != 0 or w1[i, j] != 0 or w1[j, i] != 0:
                currents.append([i, j])


"""
Initialise figure
"""

fig, ax = plt.subplots()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})


"""
Plot currents
"""

# Define range of frequencies
om_arr = np.linspace(0, 300, 10000)

for curr in currents:
    c_i, c_j = curr  # Extract transition
    res_arr = cond(om_arr, c_i, c_j)  # calculate conductance
    # plot result
    line = ax.plot(np.real(res_arr), np.imag(res_arr), label=f"${c_i}\\to{c_j}$")[0]
    add_arrow(line)

# Add axes
complex_axes(ax, r"\sigma_{mn}(\omega)")

# Polish figure
ax.tick_params(labelfontfamily="serif")
ax.set_title("Periodically driven current", fontname="serif", fontsize=18)
# ax.legend()
plt.tight_layout()

if save:
    if local:
        try:
            os.makedirs(f"{foldername}")
        except FileExistsError:
            # directory already exists
            pass
        plt.savefig(f"{figfolder}/{figname}.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(f"C:\\Users\\lucp13819\\Pictures\\{figfolder}\\{figname}.png", dpi=600, bbox_inches='tight')

plt.show()