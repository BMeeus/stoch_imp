from calculate_coeffs import calculate_coeffs

import numpy as np
import sympy as sp


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


"""
Defining useful function
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


"""
Calculate coefficients and save to npz file
"""

calculate_coeffs(weq, w1, f"{foldername}/{name}")
