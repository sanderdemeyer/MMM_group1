import numpy as np
import numpy.linalg as linalg
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as ssalg

### Code file to compute the inverse of matrix A
### This file exists to compare different methods in terms of speed.

def inversion(A, M_U, inversion_method):
    if inversion_method == 'numpy_nonsparse':
        A_inv = linalg.inv(A)
    elif inversion_method == 'numpy_sparse':
        A_csc = csc_matrix(A)
        A_inv_csc = ssalg.inv(A_csc)
        A_inv = A_inv_csc.toarray()
    elif inversion_method == 'numpy_schur':
        M11 = A[:M_U+1,:M_U+1]
        M12 = A[:M_U+1,M_U+1:]
        M21 = A[M_U+1:,:M_U+1]
        M22 = A[M_U+1:,M_U+1:]
        M22_inv = linalg.inv(M22)
        S = M11 - np.dot(M12,np.dot(M22_inv,M21))
        S_inv = linalg.inv(S)
        A_inv = np.zeros((2*M_U,2*M_U))
        Deel1 = S_inv
        Deel2 =  -np.dot(S_inv,np.dot(M12,M22_inv))
        Deel3 = -np.dot(M22_inv,np.dot(M21,S_inv))
        Deel4 =  M22_inv + np.dot(M22_inv,np.dot(M21,np.dot(S_inv,np.dot(M12,M22_inv))))
        A_inv[:M_U+1,:M_U+1] = Deel1
        A_inv[:M_U+1,M_U+1:] = Deel2
        A_inv[M_U+1:,:M_U+1] = Deel3
        A_inv[M_U+1:,M_U+1:] = Deel4
    elif inversion_method == 'numpy_sparse_schur':
        M11 = csc_matrix(A[:M_U+1,:M_U+1])
        M12 = csc_matrix(A[:M_U+1,M_U+1:])
        M21 = csc_matrix(A[M_U+1:,:M_U+1])
        M22 = csc_matrix(A[M_U+1:,M_U+1:])
        M22_inv = ssalg.inv(M22)
        S = M11 - M12*M22_inv*M21
        S_inv = ssalg.inv(S)
        M_inv = np.zeros((2*M_U,2*M_U))
        Deel1 = S_inv
        Deel2 =  -S_inv*M12*M22_inv
        Deel3 = -M22_inv*M21*S_inv
        Deel4 =  M22_inv + M22_inv*M21*S_inv*M12*M22_inv
        M_inv[:M_U+1,:M_U+1] = Deel1.toarray()
        M_inv[:M_U+1,M_U+1:] = Deel2.toarray()
        M_inv[M_U+1:,:M_U+1] = Deel3.toarray()
        M_inv[M_U+1:,M_U+1:] = Deel4.toarray()
        A_inv = M_inv
    else:
        print('Invalid inversion method')
    return A_inv