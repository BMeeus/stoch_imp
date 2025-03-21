import numpy as np
import sympy as sp


n = 4  # Number of sites
f, om = sp.symbols("f, omega", real=True)  # variable chem. pot. and driving freq
fEq = 0  # base value of chemical potential

t = 1 # Transition rate between sites


def add_trans(i, j, rate):
    """
    Adds a symmetric transition to matrix w

    :param i: Outgoing site
    :param j: Incoming site
    :param rate: rate of the transition
    :return: None
    """
    w[i, j] = rate * sp.exp(-f)
    w[j, i] = rate * sp.exp(f)
    return


"""
Build matrix W_base
"""

w = sp.zeros(n, n)  # initialise base matrix

for i in range(n):
    add_trans(i, (i+1)%n, t)  # Add transition between consecutive sites, periodic bc

add_trans(0, 2, t)

for i in range(n):
    w[i, i] = - sum(w[:, i])  # add diagonal terms


"""
Calculate Weq, W1
"""
weq = np.array(sp.N(w.subs({f: fEq})), dtype=np.float64)

w1 = np.array(sp.N(sp.diff(w, f).subs({f: 0})), dtype=np.float64)  # First term in taylor expansion

np.savez("np_files/loop", weq=weq, w1=w1)


# TODO: add comments