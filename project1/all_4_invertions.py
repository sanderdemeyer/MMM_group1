import numpy as np
from functions import def_update_matrices_new, def_update_matrices
import numpy.linalg as linalg
from time import perf_counter
from scipy.linalg import lu
import scipy.sparse.linalg as ssalg
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix


epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8

def direct_scipy_sparse_inversion(X):
    M = X
    N = X
    
    epsilon = np.ones((M,N))*epsilon_0
    mu = np.ones((M,N))*mu_0
    sigma = np.ones((M,N))*0

    delta_x = np.ones(M)*10/M
    delta_y = np.ones(N)*10/N
    delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
    delta_y_matrix = np.array([delta_y for i in range(M)])
    courant_number = 1
    delta_t = np.min(delta_y)/(c)*courant_number
    A,B = def_update_matrices(epsilon,mu,sigma,delta_x,delta_y,delta_t,M)
    

    t1 = perf_counter()
    A_csc = csc_matrix(A)
    A_inv_csc = ssalg.inv(A_csc)
    t2 = perf_counter()
    print(t2 - t1)
    print('Time it takes to invert A directly with scipy.sparse inversions is %f seconds for a matrix of dimension ' %(t2-t1))
    return t2-t1

def direct_numpy_inversion(X):
    M = X
    N = X
    
    epsilon = np.ones((M,N))*epsilon_0
    mu = np.ones((M,N))*mu_0
    sigma = np.ones((M,N))*0

    delta_x = np.ones(M)*10/M
    delta_y = np.ones(N)*10/N
    delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
    delta_y_matrix = np.array([delta_y for i in range(M)])
    courant_number = 1
    delta_t = np.min(delta_y)/(c)*courant_number
    A,B = def_update_matrices(epsilon,mu,sigma,delta_x,delta_y,delta_t,M)
    t1 = perf_counter()
    A_inv = linalg.inv(A)
    t2 = perf_counter()
    print(t2 - t1)
    print('Time it takes to invert A directly with scipy.sparse inversions is %f seconds for a matrix of dimension' %(t2-t1))
    return t2-t1

def schur_scipy_inversion(X):
    M = X
    N = X
    
    epsilon = np.ones((M,N))*epsilon_0
    mu = np.ones((M,N))*mu_0
    sigma = np.ones((M,N))*0

    delta_x = np.ones(M)*10/M
    delta_y = np.ones(N)*10/N
    delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
    delta_y_matrix = np.array([delta_y for i in range(M)])
    courant_number = 1
    delta_t = np.min(delta_y)/(c)*courant_number
    A,B = def_update_matrices(epsilon,mu,sigma,delta_x,delta_y,delta_t,M)
    t1 = perf_counter()
    M11 = csc_matrix(A[:M,:M])
    M12 = csc_matrix(A[:M,M:])
    M21 = csc_matrix(A[M:,:M])
    M22 = csc_matrix(A[M:,M:])
    M22_inv = ssalg.inv(M22)
    S = M11 - M12*M22_inv*M21
    S_inv = ssalg.inv(S)
    M_inv = np.zeros((2*M,2*M))
    Deel1 = S_inv
    Deel2 =  -S_inv*M12*M22_inv
    Deel3 = -M22_inv*M21*S_inv
    Deel4 =  M22_inv + M22_inv*M21*S_inv*M12*M22_inv
    M_inv[:M,:M] = Deel1.toarray()
    M_inv[:M,M:] = Deel2.toarray()
    M_inv[M:,:M] = Deel3.toarray()
    M_inv[M:,M:] = Deel4.toarray()
    
    t2 = perf_counter()
    print(t2 - t1)
    print('Time it takes to invert A with schur with scipy.sparse inversions is %f seconds for a matrix of dimension' %(t2-t1))
    return t2-t1

def schur_numpy_inversion(X):
    M = X
    N = X
    
    epsilon = np.ones((M,N))*epsilon_0
    mu = np.ones((M,N))*mu_0
    sigma = np.ones((M,N))*0

    delta_x = np.ones(M)*10/M
    delta_y = np.ones(N)*10/N
    delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
    delta_y_matrix = np.array([delta_y for i in range(M)])
    courant_number = 1
    delta_t = np.min(delta_y)/(c)*courant_number
    A,B = def_update_matrices(epsilon,mu,sigma,delta_x,delta_y,delta_t,M)
    t1 = perf_counter()
    M11 = A[:M,:M]
    M12 = A[:M,M:]
    M21 = A[M:,:M]
    M22 = A[M:,M:]
    M22_inv = linalg.inv(M22)
    S = M11 - np.dot(M12,np.dot(M22_inv,M21))
    S_inv = linalg.inv(S)
    M_inv = np.zeros((2*M,2*M))
    Deel1 = S_inv
    Deel2 =  -np.dot(S_inv,np.dot(M12,M22_inv))
    Deel3 = -np.dot(M22_inv,np.dot(M21,S_inv))
    Deel4 =  M22_inv + np.dot(M22_inv,np.dot(M21,np.dot(S_inv,np.dot(M12,M22_inv))))
    M_inv[:M,:M] = Deel1
    M_inv[:M,M:] = Deel2
    M_inv[M:,:M] = Deel3
    M_inv[M:,M:] = Deel4
    
    t2 = perf_counter()
    print(t2 - t1)
    print('Time it takes to invert A with schur with scipy.sparse inversions is %f seconds for a matrix of dimension' %(t2-t1))
    return t2-t1





M_list = [50,100,500,1000]
Time_d_np = []
Time_d_ss = []
Time_schur_np = []
Time_schur_ss = []
for i in range(len(M_list)):
    Time_d_np.append(direct_numpy_inversion(M_list[i]))
    Time_d_ss.append(direct_scipy_sparse_inversion(M_list[i]))
    Time_schur_np.append(schur_numpy_inversion(M_list[i]))
    Time_schur_ss.append(schur_scipy_inversion(M_list[i]))


plt.figure()
plt.scatter(M_list,Time_d_np, color = 'yellow', label = 'direct np')
plt.scatter(M_list,Time_d_ss, color = 'blue', label = 'direct_ss')
plt.scatter(M_list,Time_schur_np, color = 'green', label = 'schur_np')
plt.scatter(M_list,Time_schur_ss, color = 'red', label = 'schur_ss')
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('Magnitude of M')
plt.ylabel('Time required for inverse computation of matrix A in s')
plt.show()