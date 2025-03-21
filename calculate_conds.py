import numpy as np
import scipy.linalg as lin


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


name = "loop_ran"
out = np.load(f"np_files/{name}.npz")

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
    if lin.norm(np.matmul(weq, vecs[:, i]) - vals[i] * vecs[:, i]) >= 10 ** -14:
        print(lin.norm(np.matmul(weq, vecs[:, i]) - vals[i] * vecs[:, i]))
        raise ValueError("Incorrect computation of eigenvectors")

# Calculate Coeffs of Pad expanded in eigenvects
p1coeffs = np.sum(vecs[:, 1:].T * np.matmul(w1, Peq)/Peq, axis=1) / vals[1:]

coeffs = np.zeros([n, n, n])
for k in range(n):
    if k == 0:
        coeffs[:, :, k] = (w1 * Peq).T - w1 * Peq
    else:
        coeffs[:, :, k] = p1coeffs[k-1] * ((weq * vecs[:, k]).T - weq * vecs[:, k])

np.savez(f"np_files/coeff_{name}", coeffs=coeffs, vals=vals, vecs=vecs, weq=weq, w1=w1)

# TODO: Add comments