import numpy as np
import sympy as sp
import scipy.linalg as lin
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


def inner(v1, v2):
    return sum([v1[i] * v2[i] / peq[i] for i in range(n + 1)])


def orthog(V):
    # Orthogonalized, To Be Returned
    orthogonal = np.zeros(V.shape)

    # At each step, take vector
    for i in range(len(V)):
        v = V[:, i]
        # Subtract off the "components" from current orthogonal set.
        for j in range(i):
            v = v - inner(orthogonal[:, j], v) * orthogonal[:, j]
        # Normalization
        v /= np.sqrt(inner(v, v))
        orthogonal[:, i] = v
    return orthogonal


def curr_to_trans(curr):
    if n+1 in curr:
        return [curr[0], 0]
    else:
        return curr


def getcoeff(curr, k):
    if k == 0:
        curr = curr_to_trans(curr)
        return W1[curr[0], curr[1]] * peq[curr[1]] - W1[curr[1], curr[0]] * peq[curr[0]]
    else:
        k -= 1
        curr = curr_to_trans(curr)
        return p1coeffs[k] * (Weq[curr[0], curr[1]] * vecs[k][curr[1]] - Weq[curr[1], curr[0]] * vecs[k][curr[0]])


def basem_to_transm(w):
    w_trans = w[:-1, :-1]
    w_trans[-1, 0] = w[-2, -1]
    w_trans[0, -1] = w[-1, -2]

    for i in range(n + 1):
        w_trans[i, i] = - sum(w_trans[:, i])
    return w_trans


def compute_coeffs(weq: np.ndarray, w1: np.ndarray) -> np.ndarray:
    global p1coeffs, peq, vals, vecs

    peq = lin.null_space(weq)
    peq /= sum(peq)
    vals, vecs = lin.eig(weq)

    vals = np.real(vals)
    vecs = np.real(vecs)

    vecs = orthog(vecs)

    for i in range(len(vecs[0, :])):
        if np.sum((np.matmul(weq,vecs[:, i]) - vals[i] * vecs[:, i])**2) >= 10 ** -15:
            raise ValueError("Incorrect computation of eigenvectors")

    p1coeffs = [inner(vecs[:, i], np.matmul(w1,peq)) / vals[i] for i in range(1, len(vecs))]

    coeffs = np.zeros((n+1, n+1, n+1))

    for k in range(n+1):
        if k == 0:
            coeffs[:, :, k] = w1*peq - (w1*peq).T
        else:
            coeffs[:, :, k] = p1coeffs[k-1] * (weq * vecs[:, k] - (weq * vecs[:, k]).T)
    return coeffs


wbase = sp.zeros(n+2, n+2)
wbase[0, 1] = rl - rl / (sp.exp(-mul)+1)
wbase[1, 0] = rl / (sp.exp(-mul)+1)
wbase[-2, -1] = rr - rr / (sp.exp(-mur)+1)
wbase[-1, -2] = rr / (sp.exp(-mur)+1)

for i in range(1, n):
    wbase[i, i+1] = t
    wbase[i+1, i] = t

wbaseEq = wbase.subs({mu: muEq})

W = basem_to_transm(wbase)

Weq = np.array(sp.N(W.subs({mu: 0})), dtype=float)

W1 = np.array(sp.N(sp.diff(W, mu).subs({mu: 0})), dtype=float)

cff = compute_coeffs(Weq, W1)

#TODO: write function to plot conds = coeffs * l_vec

# for cond in conds:
#     condf = sp.lambdify([om], cond)
#     om_arr = np.linspace(0, 100, 1000)
#     plt.plot(np.real(condf(om_arr)), np.imag(condf(om_arr)))
#
# plt.show()