import numpy as np
from functions import def_update_matrices_new, def_update_matrices
import numpy.linalg as linalg
from time import perf_counter
from scipy.linalg import lu
import scipy.sparse.linalg as ssalg

from scipy.sparse import csc_matrix


epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8
M = 150
N = 150

epsilon = np.ones((M,N))*epsilon_0
mu = np.ones((M,N))*mu_0
sigma = np.ones((M,N))*0

delta_x = np.ones(M)*10/M
delta_y = np.ones(N)*10/N

#delta_x = [((i+1)**(1/10))*10/M for i in range(M)]

delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
delta_y_matrix = np.array([delta_y for i in range(M)])

courant_number = 1
delta_t = np.min(delta_y)/(c)*courant_number



A,B = def_update_matrices(epsilon,mu,sigma,delta_x,delta_y,delta_t,M)
t1 = perf_counter()
M11 = A[:M,:M]
M12 = A[:M,M:]
M21 = A[M:,:M]
M22 = csc_matrix(A[M:,M:])
M22_inv = ssalg.inv(M22)
M22_inv = M22_inv.toarray()
S = M11 - np.dot(M12,np.dot(M22_inv,M21))
S = csc_matrix(S)
print('it goes well')
t_int2 = perf_counter()
S_inv = ssalg.inv(S)
t_int = perf_counter()
print(t_int-t_int2)
S_inv = S_inv.toarray()
M_inv = np.zeros((2*M,2*M))
M_inv[:M,:M] = S_inv
M_inv[:M,M:] = -np.dot(np.dot(S_inv,M12),M22_inv)
M_inv[M:,:M] = -np.dot(np.dot(M22_inv,M21),S_inv)
M_inv[M:,M:] = M22_inv + np.dot(M22_inv, np.dot(M21,np.dot(S_inv,np.dot(M12,M22_inv))))



t2 = perf_counter()
print(t2 - t1)
print('Time it takes to invert M by using schur matrices with scipy.sparse inversions is %f seconds' %(t2-t1))
A_inv = linalg.inv(A)

t3 = perf_counter()
print(t3-t2)
