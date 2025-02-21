import numpy as np
import numpy.linalg as lin
import sympy as sp


N = 4
mu_L = 0
mu_R = 0
F = 1
t = sp.symbols("t", real=True, positive=True)

G_pL = 1/(1+sp.exp(-(mu_L + F * sp.cos(t))))
G_mL = 1-G_pL

G_pR = 1/(1+sp.exp(-mu_R))
G_mR = 1-G_pR

W = sp.Matrix(np.zeros([N,N]))

W[1, 0] = G_pL
W[0, 1] = G_mL

W[N-1, 0] = G_pR
W[0, N-1] = G_mR

for n in range(1, N-1):
    W[n, n+1] = 1
    W[n+1, n] = 1

for n in range(N):
    W[n, n] = -sum(W[:, n])


Weq = W.subs({t: sp.pi/2})

res = Weq.eigenvects()

evals = []
evecs = []

for spc in res:
    for degen in range(spc[1]):
        evals.append(spc[0])
        evecs.append(spc[2][degen])

evals[-1] = 0

for vec in evecs:
    print(sum(vec))
    vec = vec/sum(vec)

Peq = evecs[-1]
print(evecs)
print(sp.pretty_print(Peq))