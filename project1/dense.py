#%%

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)

M = 4
N = 5

iterations = 50

modulated = True
J0 = 1
tc = 5
sigma_source = 1
omega_c = 20

# last 'extra' element should be the same as the 0th.
epsilon = np.ones((M+1,N))*epsilon_0
mu = np.ones((M+1,N))*mu_0
sigma = np.ones((M+1,N))*0

delta_x = np.ones(M)*10**(-1)
delta_y = np.ones(N)*10**(-1)
delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
delta_y_matrix = np.array([delta_y for i in range(M)])
delta_t = 10**(-10)

observation_point_ez = np.array([[2, 2]])

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
    #assumes arrays
    #bx = bx_old + np.dot(E, ez_old)
    #return bx

    #assumes matrices
    bx = np.zeros((M, N))
    bx = bx_old[:,:-1] - (ez_old[:,1:] - ez_old[:,:-1])
    bx[:,-1] = bx_old[:,-1] - (ez_old[:,0] - ez_old[:,-1]) # add periodic boundary condition
    return bx

def update_implicit(ez_old, hy_old, bx, n):
    # to be changed
    [A, B] = def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, n)

    bx_term = -(delta_t/2)*np.divide(bx, np.multiply(mu, delta_y_matrix))
    C_term = np.roll(bx_term, -1, 0) + bx_term - np.roll(np.roll(bx_term, -1, 0), 1, 1) - np.roll(bx_term, 1, 1)
    
    # should be delta_y_star_matrix
    jz_n = -np.multiply(delta_y_matrix, jz[:,:,n])/4
    jz_nm1 = -np.multiply(delta_y_matrix, jz[:,:,n-1])/4
    D_term = -np.roll(jz_n, -1, 0) - jz_n - np.roll(jz_nm1, -1, 0) - jz_nm1

    new_values = np.dot(linalg.inv(A), (np.dot(B, np.concatenate((ez_old, hy_old))) + C_term + D_term))
    ez_new = new_values[:M,:]
    hy_new = new_values[M:,:]
    return [new_values[:M,:], new_values[M:,:]]

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
    A = np.zeros((2*M, 2*M))
    B = np.zeros((2*M, 2*M))

    for i in range(M):
        for j in range(M):
            #i = b // N
            #j = b % N
            b = N*i + j
            if i != M-1:
                A[i+1,j] = delta_y[j]/(2*delta_x[i])
                A[i, j] = -delta_y[j]/(2*delta_x[i])
                A[i+1,M+j] = -mu[i+1,j]/(2*delta_t)
                A[i,M+j] = -mu[i,j]/(2*delta_t)

                B[i+1,j] = -delta_y[j]/(2*delta_x[i])
                B[i,j] = delta_y[j]/(2*delta_x[i])
                B[i+1,M+j] = mu[i+1,j]/(2*delta_t)
                B[i,M+j] = mu[i,j]/(2*delta_t)

                # delta_y should be delta_y*
                A[M+i+1,j] = (epsilon[i+1,j]/(2*delta_t) + sigma[i+1,j]/4)*delta_y[j]
                A[M+i,j] = (epsilon[i,j]/(2*delta_t) + sigma[i,j]/4)*delta_y[j]
                A[M+i+1,M+j] = -1/(2*delta_x[i+1])
                A[M+i,M+j] = 1/(2*delta_x[i])

                B[M+i+1,j] = (epsilon[i+1,j]/(2*delta_t) - sigma[i+1,j]/4)*delta_y[j]
                B[M+i,j] = (epsilon[i,j]/(2*delta_t) - sigma[i,j]/4)*delta_y[j]
                B[M+i+1,M+j] = 1/(2*delta_x[i+1])
                B[M+i,M+j] = -1/(2*delta_x[i])

            else:
                A[0,j] = delta_y[j]/(2*delta_x[i])
                A[i, j] = -delta_y[j]/(2*delta_x[i])
                A[0,M+j] = -mu[0,j]/(2*delta_t)
                A[i,M+j] = -mu[i,j]/(2*delta_t)

                B[0,j] = -delta_y[j]/(2*delta_x[i])
                B[i,j] = delta_y[j]/(2*delta_x[i])
                B[0,M+j] = mu[0,j]/(2*delta_t)
                B[i,M+j] = mu[i,j]/(2*delta_t)

                # delta_y should be delta_y*
                A[M,j] = (epsilon[0,j]/(2*delta_t) + sigma[0,j]/4)*delta_y[j]
                A[M+i,j] = (epsilon[i,j]/(2*delta_t) + sigma[i,j]/4)*delta_y[j]
                A[M,M+j] = -1/(2*delta_x[0])
                A[M+i,M+j] = 1/(2*delta_x[i])

                B[M,j] = (epsilon[0,j]/(2*delta_t) - sigma[0,j]/4)*delta_y[j]
                B[M+i,j] = (epsilon[i,j]/(2*delta_t) - sigma[i,j]/4)*delta_y[j]
                B[M,M+j] = 1/(2*delta_x[0])
                B[M+i,M+j] = -1/(2*delta_x[i])

    return [A, B]


def run():
    ez = np.zeros((M,N))
    hy = np.zeros((M,N))
    bx = np.zeros((M,N))

    bx_list = np.zeros((M,N, 100))

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