import numpy as np
import sympy as sp
import scipy.linalg as lin

def inner(v1, v2):
    """
    Inner product of two vectors

    :param v1: first vector
    :param v2: second vector
    :return: inner product of v1 and v2
    """
    return np.sum(v1*v2/Peq)


def gram_schmidt(v_arr):
    """
    Orthogonalise a set of vectors given as the columns of an array using Gram-Schmidt procedure.

    :param v_arr: array of vectors to orthogonalise
    :return: array of orthogonalised vectors
    """
    # Orthogonalized, To Be Returned
    orthogonal = np.zeros(v_arr.shape)

    # At each step, take vector
    for i in range(len(v_arr)):
        v = v_arr[:, i]

        # Subtract off the "components" from current orthogonal set.
        for j in range(i):
            v -= sum(orthogonal[:, j]*v/Peq) * orthogonal[:, j]
        # Normalization
        v /= np.sqrt(sum(v*v/Peq))
        orthogonal[:, i] = v
    return orthogonal


def getcoeff(curr, k):
    """
    gets the coefficients for a current m, n and an eigenvector k

    :param curr: current m, n considered
    :param k: eigenvector according to which the coefficient is calculated
    :return: the calculated coefficient
    """
    if k == 0:
        return -sp.simplify(w1[curr[0], curr[1]] * Peq[curr[1]] - w1[curr[1], curr[0]] * Peq[curr[0]])
    else:
        k -= 1
        return -sp.simplify(p1coeffs[k] * (weq[curr[0], curr[1]] * vecs[k][curr[1]]
                              - weq[curr[1], curr[0]] * vecs[k][curr[0]]))

name = "loop"
out = np.load(f"{name}.npz")

weq = out["weq"]
w1 = out["w1"]

n = len(weq[0, :])

Peq = lin.null_space(weq)
Peq /= sum(Peq)  # Force Peq to sum to unity

vals, vecs = lin.eig(weq)
vals = np.real(vals)
vecs = np.real(vecs)


ind_arr = np.flip(np.argsort(vals))

vals = vals[ind_arr]
vecs = vecs[:, ind_arr]

vecs[:, 0] /= sum(vecs[:, 0])
Peq = vecs[:, 0]


vecs = gram_schmidt(vecs)

# Check validity of eigenvectors
for i in range(len(vecs)):
    if lin.norm(np.matmul(weq, vecs[:, i]) - vals[i] * vecs[:, i]) >= 10 ** -15:
        raise ValueError("Incorrect computation of eigenvectors")

# Calculate Coeffs of Pad expanded in eigenvects
p1coeffs = np.sum(vecs[:, 1:].T * np.matmul(w1, Peq)/Peq, axis=1) / vals[1:]

coeffs = np.zeros([n, n, n])
for k in range(n):
    if k == 0:
        coeffs[:, :, k] = w1 * Peq - (w1 * Peq).T
    else:
        coeffs[:, :, k] = p1coeffs[k-1] * (weq * vecs[:, k] - (weq * vecs[:, k]).T)

np.savez(f"{name}", coeffs=coeffs, vals=vals, vecs=vecs, weq=weq, w1=w1)
