import numpy as np
import numpy.linalg as lin

N = 4
mu_L = 1
mu_R = 1

G_pL = 1/(1+np.exp(-mu_L))
G_mL = 1-G_pL

G_pR = 1/(1+np.exp(-mu_R))
G_mR = 1-G_pR

W = np.zeros([N,N])

W[1, 0] = G_pL
W[0, 1] = G_mL

W[N-1, 0] = G_pR
W[0, N-1] = G_mR

for n in range(1, N-1):
    W[n, n+1] = 1
    W[n+1, n] = 1

for n in range(N):
    W[n, n] = -sum(W[:, n])

l, v = lin.eig(W)