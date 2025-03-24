import numpy as np
import sympy as sp
import os

"""
Template file performing the calculation for a linear chain of n sites connected at both ends to a basin.
"""


"""
Simulation parameters
"""
n = 3                             # Number of sites (excl. basins)
name = f"{n}_sites"
foldername = "n_sites_lin"
mu = sp.symbols("mu", real=True)  # variable chem. pot. and driving freq
muEq = 0                          # base value of chemical potential

rr = 1                            # Transition rate right basin
rl = 1                            # Transition rate left basin
t = 1                             # Transition rate between sites


"""
Driving mu = a*mu_left + b*mu_right 
"""

a = 1  # a=1 -> full left driving
b = 0  # b=1 -> full right driving

while a+b != 1:
    print("a and b do not sum to 1, please enter new values:")
    a = input("a: ")
    b = input("b: ")

mul = a*mu  # Driving force left
mur = -b*mu  # Driving force right


"""
Defining useful functions
"""

def resratep(base, m):
    """
    Rate of transition from reservoir into system. Follows Fermi distribution.

    :param base: Base transition rate representing coupling of reservoir to system
    :param m: chemical potential mu
    :return: Transition rate of particle creation
    """
    return base / (sp.exp(-m)+1)


def resratem(base, m):
    """
    Rate of transition from system into reservoir. Sums to base rate with resratep.

    :param base: Base transition rate representing coupling of reservoir to system
    :param m: chemical potential mu
    :return: Transition rate of particle destruction
    """
    return base - resratep(base, m)


def add_trans(i, j, rate):
    """
    Adds a symmetric transition to matrix w

    :param i: Outgoing site
    :param j: Incoming site
    :param rate: rate of the transition
    :return: None
    """
    wbase[i, j] = rate
    wbase[j, i] = rate
    return


"""
Build matrix W_base based on sites
"""

wbase = sp.zeros(n+2, n+2)         # initialise base matrix (sites + 2 basins)
wbase[0, 1] = resratem(rl, mul)    # Add site 1 to left basin
wbase[1, 0] = resratep(rl, mul)    # Add left basin to site 1
wbase[-2, -1] = resratem(rr, mur)  # Add site n to right basin
wbase[-1, -2] = resratep(rr, mur)  # Add right basin to site n

for site in range(1, n):
    add_trans(site, site + 1, t)      # Add transition between consecutive sites

"""
Transform W_base into W_transition
"""

w = wbase[:-1, :-1]           # Keep left basin and sites (basin becomes no particle state)
w[-1, 0] = wbase[-2, -1]      # Map n -> R Basin to n -> no particle
w[0, -1] = wbase[-1, -2]      # Map R Basin -> n to no particle -> n

for col in range(n + 1):
    w[col, col] = - sum(w[:, col])  # add diagonal terms

"""
Calculate Weq, W1
"""
weq = np.array(sp.N(w.subs({mu: muEq})), dtype=np.float64)           # Equilibrium transfer matrix

w1 = np.array(sp.N(sp.diff(w, mu).subs({mu: 0})), dtype=np.float64)  # First term in taylor expansion


"""
Calculate coefficients and save to npz file
"""
try:
    os.makedirs(f"{foldername}")
except FileExistsError:
    # directory already exists
    pass

# Save weq and w1 to be used in calculation
np.savez(f"{foldername}/{name}.npz", weq=weq, w1=w1)

# Calculate coefficients and update .npz file
os.system(f"C:/Users/Branko/anaconda3/python.exe calculate_coeffs.py {name} {foldername}")
