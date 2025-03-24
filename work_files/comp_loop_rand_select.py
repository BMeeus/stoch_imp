import numpy as np
import sympy as sp


name = "loop_rand_c"

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
    w[i, j] = rate
    w[j, i] = rate
    return

def add_driven_trans(i, j, rate):
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
en = np.random.random(5)
for i in range(n):
    if i == n-1:
        add_driven_trans(i, (i+1)%n, en[i])
    else:
        add_driven_trans(i, (i+1)%n, en[i])  # Add transition between consecutive sites, periodic bc

for i in range(n):
    w[i, i] = - sum(w[:, i])  # add diagonal terms

weq = np.array(sp.N(w.subs({f: fEq})), dtype=np.float64)

w1 = np.array(sp.N(sp.diff(w, f).subs({f: 0})), dtype=np.float64)

np.savez(f"np_files/{name}_no", weq=weq, w1=w1, en=en)

"""
Calculate Weq, W1
"""

add_driven_trans(0, 2, en[n])

for i in range(n):
    w[i, i] = 0
    w[i, i] = - sum(w[:, i])


weq = np.array(sp.N(w.subs({f: fEq})), dtype=np.float64)

w1 = np.array(sp.N(sp.diff(w, f).subs({f: 0})), dtype=np.float64)  # First term in taylor expansion



np.savez(f"np_files/{name}", weq=weq, w1=w1, en=en)