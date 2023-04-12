import numpy as np
from functions import def_update_matrices_new
import numpy.linalg as linalg
from time import perf_counter
from scipy.linalg import lu


import scipy.sparse.linalg as ssalg

from scipy.sparse import csc_matrix


epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8
M = 100
N = 100

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



A,B = def_update_matrices_new(epsilon,mu,sigma,delta_x,delta_y,delta_t,M)


t1 = perf_counter()
M11 = csc_matrix(A[:M,:M])
M12 = csc_matrix(A[:M,M:])
M21 = csc_matrix(A[M:,:M])
M22 = csc_matrix(A[M:,M:])
M22 = csc_matrix(M22)
M22_inv = ssalg.inv(M22)
Temp = csc_matrix(M12.multiply(M22_inv))
S = M11 - Temp.multiply(M21)
S = csc_matrix(S)

S = S.toarray()
M22 = M22.toarray()

S_inv = ssalg.inv(S)


t2 = perf_counter()
print(t2 - t1)
print('Time it takes to invert M by using schur matrices with scipy.sparse inversions is %f seconds' %(t2-t1))

A_inv = ssalg.inv(csc_matrix(A))

t3 = perf_counter()
print('Time it takes to invert 2M by using scipy.sparse inversions is %f seconds' %(t3-t2))


A_inv = linalg.inv(A)

t4 = perf_counter()
print('Time it takes to invert 2M by using numpy inversions is %f seconds' %(t4-t3))
