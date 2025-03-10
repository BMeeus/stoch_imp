from pyplot_funcs import *

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


n = 3
mu, om = sp.symbols("mu, omega", real=True)
muEq = 0

rr = 1
rl = 1
t = 1

a = 1
b = 0

while a+b != 1:
    print("a and b do not sum to 1, please enter new values:")
    a = input("a: ")
    b = input("b: ")

mul = a*mu
mur = -b*mu

def resratep(base, m):
    return base / (sp.exp(-m)+1)


def resratem(base, m):
    return base - resratep(base, m)


def add_trans(i, j, rate):
    wbase[i, j] = rate
    wbase[j, i] = rate
    return


def pprint(o):
    print(sp.pretty_print(o))
    return


def w(x):
    return winterim.subs({mu: x})


def inner(v1, v2):
    return sum([v1[i] * v2[i] / peq[i] for i in range(n + 1)])


def norm(v):
    return sp.sqrt(inner(v, v))


def normal(v):
    return v/norm(v)


def gram_schmidt(V):
    # Orthogonalized, To Be Returned
    orthogonal = []

    # At each step, take vector
    for i in range(len(V)):
        v = sp.Matrix(V[i][:])

        # Subtract off the "components" from current orthogonal set.
        for j in range(i):
            v = v - inner(orthogonal[j], v) * orthogonal[j]
        # Normalization
        v = normal(v)
        orthogonal.append(v)

    return orthogonal


def findtrans(curr):
    if n+1 in curr:
        return [curr[0], 0]
    else:
        return curr

def getcoeff(curr, k):
    if k == 0:
        curr = findtrans(curr)
        return -sp.simplify(w1[curr[0], curr[1]] * peq[curr[1]] - w1[curr[1], curr[0]] * peq[curr[0]])
    else:
        k -= 1
        curr = findtrans(curr)
        return -sp.simplify(p1coeffs[k] * (weq[curr[0], curr[1]] * vecs[k][curr[1]]
                              - weq[curr[1], curr[0]] * vecs[k][curr[0]]))


wbase = sp.zeros(n+2, n+2)
wbase[0, 1] = resratem(rl, mul)
wbase[1, 0] = resratep(rl, mul)
wbase[-2, -1] = resratem(rr, mur)
wbase[-1, -2] = resratep(rr, mur)

for i in range(1, n):
    add_trans(i, i+1, t)

wbaseEq = wbase.subs({mu: muEq})

winterim = wbase[:-1, :-1]
winterim[-1, 0] = wbase[-2, -1]
winterim[0, -1] = wbase[-1, -2]

for i in range(n+1):
    winterim[i, i] = - sum(winterim[:, i])

weq = w(0)

w1 = sp.diff(winterim, mu).subs({mu: 0})

peq = weq.nullspace()[0]
peq /= sum(peq)

eig_syst = sorted(weq.eigenvects(), key=lambda x: x[0], reverse=True)

vals = []
vecs = []

for spc in eig_syst:
    for degen in range(spc[1]):
        vals.append(spc[0])
        vecs.append(spc[2][degen])

for i in range(len(vecs)):
    if (weq*vecs[i] - vals[i] * vecs[i]).norm() >= 10**-15:
        raise ValueError("Incorrect computation of eigenvectors")

vals = vals[1:]
vecs = gram_schmidt(vecs)[1:]
p1coeffs = [sp.simplify(inner(vecs[i], w1*peq)/vals[i]) for i in range(len(vecs))]

currents = []
for i in range(n+2):
    for j in range(i, n+2):
        if wbaseEq[i, j] != 0 or wbaseEq[j, i] != 0:
            currents.append([i, j])


coeffs = sp.Matrix([[getcoeff(current, l) for l in range(n+1)] for current in currents])

conds = [sp.simplify(sum([coeffs[i, k] * (1 if k == 0 else (vals[k-1]/(1j*om - vals[k-1]))) for k in range(n+1)])) for i in range(len(currents))]
ad_cond = conds[0].subs({om: 0})
nconds = [c/ad_cond for c in conds]

fig, ax = plt.subplots()
for cond in nconds:
    condf = sp.lambdify([om], cond)
    om_arr = np.linspace(0, 300, 10000)
    line = ax.plot(np.real(condf(om_arr)), np.imag(condf(om_arr)))[0]
    # add_arrow(line)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathpazo"
})
ax.set_aspect('equal')
ax.grid(True, which='both')
y_ax = ax.axvline(x=0, color='k')
x_ax = ax.axhline(y=0, color='k')
x_min, x_max = ax.get_xlim()
x_range = x_max-x_min
y_min, y_max = ax.get_ylim()
y_range = y_max - y_min
ax.text(x_range/50, y_max-y_range/10, r"$\mathrm{Im}\, \frac{\sigma_{mn}}{\sigma(0)}$")
ax.text(x_max-x_range/10, -y_range/15, r"$\mathrm{Re}\, \frac{\sigma_{mn}}{\sigma(0)}$")

plt.show()