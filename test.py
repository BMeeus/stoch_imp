import numpy as np
import numpy.linalg as lin
import random as rand
N = 3

D = np.diag([1 for _ in range(N)])

V = np.array([[rand.random()*(j <= i)for j in range(N)] for i in range(N)])
Vinv = lin.inv(V)

print(np.matmul(V, np.matmul(D, Vinv)))
