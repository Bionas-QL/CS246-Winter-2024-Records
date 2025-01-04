from scipy.linalg import svd, eigh
import numpy as np

if __name__ == '__main__':
    M = [[1, 2], [2, 1], [3, 4], [4, 3]]
    U, Sigma, V_T = svd(M, full_matrices=False)
    print(U)
    print(Sigma)
    print(V_T)

    M = np.array(M)
    Evals, Evecs = eigh(M.T.dot(M).tolist())
    print(Evals)
    print(Evecs)

    Evals, Evecs = eigh(M.dot(M.T).tolist())
    print(Evals)
    print(Evecs)

