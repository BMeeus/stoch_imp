import sympy as sp
import stoch_imp as si

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

n = 3
mu, om = sp.symbols("mu, omega", real=True)  # variable chem. pot. and driving freq
muEq = 0  # base value of chemical potential

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

for n in range(2, 51, 2):
    w = si.WMatrix(n+1, ds=mu, eq=muEq, zi=True)

    w.add_trans(0, 1, r=resratep(1, mul), ri=resratem(1, mul))
    w.add_trans(0, -1, r=resratep(1, mur), ri=resratem(1, mur))

    for i in range(1, n):
        w.add_trans(i, i+1, t)

    print(w.calc_weq())