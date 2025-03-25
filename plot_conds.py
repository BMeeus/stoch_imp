from pyplot_funcs import *

import numpy as np
import matplotlib.pyplot as plt
import os

"""
Define names and locations
"""
name = "3_sites"                   # Name and folder of data
foldername = "n_sites_lin"

figfolder = "figs_stoch_imp"      # destination name and folder of figure
figname = "loop_no_inset"

save = False                      # Save the figure
local = True                      # If True figure is saved in project. Otherwise, specify entire path!

currents = []


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
Read out data, detect transitions if necessary
"""

out = np.load(f"{foldername}/{name}.npz")

vals = out["vals"]
coeffs = out["coeffs"]
weq = out["weq"]
w1 = out["w1"]

# Detect possible transitions from weq and w1
if not currents:
    for i in range(len(weq[:, 0])):
        for j in range(i, len(weq[:, 0])):
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
