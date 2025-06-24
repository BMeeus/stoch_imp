import sympy as sp
import numpy as np
import stoch_imp as si
import matplotlib.pyplot as plt

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

for n in range(2, 51, 2):
    w = si.WMatrix(n+1, ds=mu, eq=muEq, zi=True)

    en_arr = np.random.normal(avg, sd, n)

    w.add_trans(0, 1, r=resratep(1, en_arr[0] - mul), ri=resratem(1, en_arr[0] - mul), simp=False)
    w.add_trans(0, -1, r=resratep(1, en_arr[-1] - mur), ri=resratem(1, en_arr[-1]-mur), simp=False)

    for i in range(1, n):
        en_diff = en_arr[i] - en_arr[i - 1]
        w.add_trans(i, i + 1, sp.exp(en_diff / 2), simp=False)

    # fig, ax = plt.subplots()
    #
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "mathpazo"
    # })

    w.calc_weq()
    w.weq.calc_peq()
    print(n)
    # om_arr = np.logspace(-10, 2, 10000)
    # for ind, cond in w.get_conds([(i, i+1) for i in range(n)].append((-1, 0))):
    #     res_arr = cond(om_arr)
    #     # plot result
    #     line = ax.plot(np.real(res_arr), np.imag(res_arr), label="${}\\to{}$".format(*ind))[0]
    #     si.add_arrow(line)
    #
    # si.complex_axes(ax, r"\sigma_{mn}(\omega)")