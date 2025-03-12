from pyplot_funcs import *

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


n = 3  # Number of sites (excl. basins)
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
    Adds a symmetric transition to matrix wbase

    :param i: Outgoing site
    :param j: Incoming site
    :param rate: rate of the transition
    :return: None
    """
    wbase[i, j] = rate
    wbase[j, i] = rate
    return


def pprint(o):
    print(sp.pretty_print(o))
    return


def inner(v1, v2):
    """
    Inner product of two vectors

    :param v1: first vector
    :param v2: second vector
    :return: inner product of v1 and v2
    """
    return sum([v1[i] * v2[i] / Peq[i] for i in range(n + 1)])


def gram_schmidt(v_list):
    """
    Orthogonalise a list of sympy vectors using Gram-Schmidt procedure.

    :param v_list: list of vectors to orthogonalise
    :return: list of orthogonalised vectors
    """
    # Orthogonalized, To Be Returned
    orthogonal = []

    # At each step, take vector
    for i in range(len(v_list)):
        v = v_list[i]

        # Subtract off the "components" from current orthogonal set.
        for j in range(i):
            v = v - inner(orthogonal[j], v) * orthogonal[j]
        # Normalization
        v = v/sp.sqrt(inner(v, v))
        orthogonal.append(v)

    return orthogonal


def findtrans(curr):
    """
    Change current format from base numbering to transition numbering. Returns current if current does not contain right
    basin, returns transformed current if current contains right basin

    :param curr: the current to transform. Format: [site_out, site_in]
    :return: The current transformed from base matrix to transfer matrix
    """
    if n+1 in curr:
        return [curr[0], 0]
    else:
        return curr


def getcoeff(curr, k):
    """
    gets the coefficients for a current m, n and an eigenvector k

    :param curr: current m, n considered
    :param k: eigenvector according to which the coefficient is calculated
    :return: the calculated coefficient
    """
    if k == 0:
        curr = findtrans(curr)
        return -sp.simplify(w1[curr[0], curr[1]] * Peq[curr[1]] - w1[curr[1], curr[0]] * Peq[curr[0]])
    else:
        k -= 1
        curr = findtrans(curr)
        return -sp.simplify(p1coeffs[k] * (weq[curr[0], curr[1]] * vecs[k][curr[1]]
                              - weq[curr[1], curr[0]] * vecs[k][curr[0]]))

"""
Build matrix W_base
"""

wbase = sp.zeros(n+2, n+2)  # initialise base matrix (sites + 2 basins)
wbase[0, 1] = resratem(rl, mul)  # Add site 1 to left basin
wbase[1, 0] = resratep(rl, mul)  # Add left basin to site 1
wbase[-2, -1] = resratem(rr, mur) # Add site n to right basin
wbase[-1, -2] = resratep(rr, mur)  # Add right basin to site n

for i in range(1, n):
    add_trans(i, i+1, t)  # Add transition between consecutive sites

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
weq = w.subs({mu: muEq})

w1 = sp.diff(w, mu).subs({mu: 0})  # First term in taylor expansion

"""
Determine non-zero currents from base matrix
"""
currents = []
for i in range(n+2):
    for j in range(i, n+2):
        if wbase[i, j] != 0 or wbase[j, i] != 0:
            currents.append([i, j])

"""
Calculate conductivities using Weq, W1
"""

def calculate_conds(weq, w1, currents, normal=True):
    global Peq, p1coeffs, vals, vecs
    Peq = weq.nullspace()[0]
    Peq /= sum(Peq)  # Force Peq to sum to unity

    eig_syst = sorted(weq.eigenvects(), key=lambda x: x[0],
                      reverse=True)  # Get eigensystem sorted by descending eigenvalue

    vals = []
    vecs = []

    # Unpack eigensystem into two lists
    for spc in eig_syst:
        for degen in range(spc[1]):
            vals.append(spc[0])
            vecs.append(spc[2][degen])

    # Orthogonalize and remove Peq
    vals = vals[1:]
    vecs = gram_schmidt(vecs)[1:]

    # Check validity of eigenvectors
    for i in range(len(vecs)):
        if (sp.N(weq * vecs[i]) - sp.N(vals[i] * vecs[i])).norm() >= 10 ** -15:
            raise ValueError("Incorrect computation of eigenvectors")

    # Calculate Coeffs of Pad expanded in eigenvects
    p1coeffs = [sp.simplify(inner(vecs[i], w1 * Peq) / vals[i]) for i in range(len(vecs))]

    coeffs = sp.Matrix([[getcoeff(current, l) for l in range(n + 1)] for current in currents])

    conds = [sp.simplify(
        sum([coeffs[i, k] * (1 if k == 0 else (vals[k - 1] / (1j * om - vals[k - 1]))) for k in range(n + 1)])) for i in
             range(len(currents))]
    if not normal:
        return conds
    else:
        ad_cond = conds[0].subs({om: 0})
        nconds = [c / ad_cond for c in conds]
        return nconds

nconds = calculate_conds(weq, w1, currents)


fig, ax = plt.subplots()

for cond in nconds:
    condf = sp.lambdify([om], cond)
    om_arr = np.linspace(0, 300, 10000)
    line = ax.plot(np.real(condf(om_arr)), np.imag(condf(om_arr)))[0]
    add_arrow(line)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})

complex_axes(ax, r"\frac{\sigma_{mn}}{\sigma(0)}")
ax.tick_params(labelfontfamily="serif")

plt.show()
