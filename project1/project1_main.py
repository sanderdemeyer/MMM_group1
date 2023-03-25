#%%

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)

M = 20
N = 20

iterations = 50

modulated = True
J0 = 1
tc = 5
sigma_source = 1
omega_c = 20

epsilon = np.ones(M*N)*epsilon_0
mu = np.ones(M*N+1)*mu_0
sigma = np.ones(M*N)*0

delta_x = np.ones(M)*10**(-1)
delta_y = np.ones(N)*10**(-1)
delta_t = 10**(-10)
#delta_t = 1/(c*np.sqrt(1/np.min(delta_x)**2 + 1/np.min(delta_y)**2))
observation_point_ez = np.array([[2, 2], [3, 10], [15, 15]])

def def_jz(modulated):
    jz = np.zeros((M+1, N+1, iterations))
    if modulated:
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    else:
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.sin(omega_c*n*delta_t)*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    return jz
def update_bx(bx_old, ez_old, E):
    bx = bx_old + np.dot(E, ez_old)
    return bx
    #assumes matrices
#    bx = bx_old[:,:-1] - (delta_x/delta_t)*(ez_old[:,1:] - ez_old[:,:-1])
#    bx[:,-1] = 0 # add boundary condition

def update_implicit(ez_old, hy_old, bx, n):
    [A, B, C, D] = def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, n)
    A_inv = linalg.inv(A)
    term1 = np.dot(B, np.concatenate((ez_old, hy_old)))
    term2 = np.dot(C, np.concatenate((np.zeros(M*N), bx)))
    term3 = D
    new_values = np.dot(linalg.inv(A), (np.dot(B, np.concatenate((ez_old, hy_old))) + np.dot(C, np.concatenate((np.zeros(M*N), bx))) + D))
    ez_new = new_values[:M*N]
    hy_new = new_values[M*N:]
    return [new_values[:M*N], new_values[M*N:]]

def def_explicit_update_matrix():
    E = np.zeros((M*N, M*N))
    for b in range(M*N-1):
        i = b // N
        E[b, b+1] = -delta_x[i]/delta_t
        E[b, b] = delta_x[i]/delta_t
    return E

def def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, n):
    # try-except should be changed. This works, but should be changed to boundary conditions (PML)
    # efficiency idea: start with matrices that are the same for each step, then only change the others
    A = np.zeros((2*M*N, 2*M*N))
    B = np.zeros((2*M*N, 2*M*N))
    C = np.zeros((2*M*N, 2*M*N))
    D = np.zeros(2*M*N)

    for i in range(M):
        for j in range(N):
            #i = b // N
            #j = b % N
            b = N*i + j
            # update equation (2)
            A[b, b] = -1/4
            A[b, b+1] = -1/4
            A[b, b+N] = 1/4
            A[b, b+N+1] = 1/4
            try:
                A[b, M*N + b] = -mu[b]/(4*delta_t*delta_y[j])
            except:
                pass
            try:
                A[b, M*N + b+1] = -mu[b+1]/(4*delta_t*delta_y[j+1])
            except:
                pass
            try:
                A[b, M*N + b+N] = -mu[b+N]/(4*delta_t*delta_y[j])
            except:
                pass
            try:
                A[b, M*N + b+N+1] = -mu[b+N+1]/(4*delta_t*delta_y[j+1])
            except:
                pass

            B[b, b] = 1/4
            B[b, b+1] = 1/4
            B[b, b+N] = -1/4
            B[b, b+N+1] = -1/4

            B[b, M*N + b] = -mu[b]/(4*delta_t*delta_y[j])
            try:
                B[b, M*N + b+1] = -mu[b+1]/(4*delta_t*delta_y[j+1])
            except:
                pass
            try:
                B[b, M*N + b+N] = -mu[b+N]/(4*delta_t*delta_y[j])
            except:
                pass
            try:
                B[b, M*N + b+N+1] = -mu[b+N+1]/(4*delta_t*delta_y[j+1])
            except:
                pass

            # update equation (4)
            try:
                A[M*N + b, b] = -epsilon[b]*delta_y[j]*(2*delta_t) - sigma[b]*(delta_y[j] + delta_y[j+1])/8
            except:
                pass
            try:
                A[M*N + b, b+N] = -epsilon[b+N]*delta_y[j]*(2*delta_t) - sigma[b+N]*(delta_y[j] + delta_y[j+1])/8
            except:
                pass
            try:
                A[M*N + b, M*N + b+1] = -1/2
            except:
                pass
            try:
                A[M*N + b, M*N + b+N+1] = 1/2
            except:
                pass
            try:
                B[M*N + b, b] = -epsilon[b]*delta_y[j]/(2*delta_t) + sigma[b]*(delta_y[j] + delta_y[j+1])/8
            except:
                pass
            try:
                B[M*N + b, b+N] = -epsilon[b+N]*delta_y[j]/(2*delta_t) + sigma[b+N]*(delta_y[j] + delta_y[j+1])/8
            except:
                pass
            try:
                B[M*N + b, M*N + b+1] = 1/2
            except:
                pass
            try:
                B[M*N + b, M*N + b+N+1] = 1/2
            except:
                pass

            C[M*N + b, M*N + b-1] = -delta_t/2
            C[M*N + b, M*N + b] = delta_t/2
            try:
                C[M*N + b, M*N + b-1+N] = -delta_t/2
            except:
                pass
            try:
                C[M*N + b, M*N + b+N] = delta_t/2
            except:
                pass

            D[M*N + b] = (delta_y[j]*jz[i+1,j,n] + delta_y[j]*jz[i,j,n] + delta_y[j]*jz[i+1,j,n-1] + delta_y[j]*jz[i,j,n-1])/4

            # to keep A from being singular
            A[2*M*N-1, 2*M*N-1] = 1
    return [A, B, C, D]


def run():
    ez = np.zeros(M*N)
    hy = np.zeros(M*N)
    bx = np.zeros(M*N)

    bx_list = np.zeros((M*N, 100))

    ez_list = np.zeros((iterations, len(observation_point_ez)))

    E = def_explicit_update_matrix()

    for n in range(iterations):
        print(f'iteration {n+1}/{iterations} started')
        [ez, hy, bx] = step(ez, hy, bx, E, n)
        bx_list[:,n] = bx

        for i, point in enumerate(observation_point_ez):
            ez_list[n, i] = ez[point[1]*N + point[0]]

    return bx_list, ez_list

def step(ez_old, hy_old, bx_old, E, n):
    bx_new = update_bx(bx_old, ez_old, E)
    [ez_new, hy_new] = update_implicit(ez_old, hy_old, bx_new, n)
    return [ez_new, hy_new, bx_new]


jz = def_jz(modulated)


[bx_list, ez_list] = run()

plt.plot(range(50), ez_list[:,0])
plt.show()


plt.plot(range(40), ez_list[:-10,0])
plt.show()