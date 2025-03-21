import numpy as np
import sympy as sp


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



mu, om = sp.symbols("mu, omega", real=True)  # variable chem. pot. and driving freq
muEq = 0  # base value of chemical potential
avg = 0
sd = 1

rr = 1  # Transition rate right basin
rl = 1  # Transition rate left basin
t = 1  # Transition rate between sites

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
Build matrix W_base
"""
for n in range(2, 51, 2):
    en_arr = np.random.normal(avg, sd, n)

    wbase = sp.zeros(n+2, n+2)  # initialise base matrix (sites + 2 basins)
    add_trans(0,1, sp.exp((en_arr[0] - mul)/2))
    add_trans(-2, -1, sp.exp((mur - en_arr[-1]) / 2))

    for i in range(1, n):
        en_diff = en_arr[i] - en_arr[i-1]
        add_trans(i, i+1, sp.exp(en_diff/2))  # Add transition between consecutive sites

    """
    Transform W_base into W_transition
    """

    w = wbase[:-1, :-1]  # Keep left basin and sites (basin becomes no particle state)
    w[-1, 0] = wbase[-2, -1]  # Map n -> R Basin to n -> no particle
    w[0, -1] = wbase[-1, -2]  # Map R Basin -> n to no particle -> n

    for i in range(n+1):
        w[i, i] = - sum(w[:, i])  # add diagonal terms

    """
    Calculate Weq, W1
    """
    weq = np.array(sp.N(w.subs({mu: muEq})), dtype=np.float64)

    w1 = np.array(sp.N(sp.diff(w, mu).subs({mu: 0})), dtype=np.float64)  # First term in taylor expansion

    np.savez(f"np_files/{n}_sites_1.npz", weq=weq, w1=w1)


# TODO: add comments