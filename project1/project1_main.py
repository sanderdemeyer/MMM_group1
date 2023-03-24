import numpy as np
import numpy.linalg as linalg

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)

M = 15
N = 10

iterations = 10

modulated = True
J0 = 1
tc = 5
sigma_source = 5
omega_c = 20

epsilon = np.ones(M*N)*epsilon_0
mu = np.ones(M*N+1)*mu_0
sigma = np.ones(M*N)*0

delta_x = np.ones(M)*10**(-3)
delta_y = np.ones(N)*10**(-3)
delta_t = 10**(-3)

def def_jz(modulated):
    jz = np.zeros((M+1, N+1, iterations))
    if modulated:
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n*delta_t-tc)**2/(2*sigma_source**2))
    else:
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n*delta_t-tc)**2/(2*sigma_source**2))*np.sin(omega_c*n*delta_t)
    return jz
def update_bx(bx_old, ez_old, E):
    bx = bx_old + np.dot(E, ez_old)
    return bx
    #assumes matrices
#    bx = bx_old[:,:-1] - (delta_x/delta_t)*(ez_old[:,1:] - ez_old[:,:-1])
#    bx[:,-1] = 0 # add boundary condition

def update_implicit(ez_old, hy_old, bx, n):
    [A, B, C, D] = def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, n)

    new_values = np.dot(linalg.inv(A), (np.dot(B, np.concatenate((ez_old, hy_old))) + np.dot(C, np.concatenate((np.zeros(M*N), bx))) + D))
    return [new_values[:M*N], new_values[M*N:]]

def def_explicit_update_matrix():
    E = np.zeros((M*N, M*N))
    for b in range(M*N-1):
        i = b // N
        print(delta_x[i])
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

            A[b, M*N + b] = -mu[b]/(4*delta_t*delta_x[j])
            try:
                A[b, M*N + b+1] = -mu[b+1]/(4*delta_t*delta_x[j+1])
            except:
                pass
            try:
                A[b, M*N + b+N] = -mu[b+N]/(4*delta_t*delta_x[j])
            except:
                pass
            try:
                A[b, M*N + b+N+1] = -mu[b+N+1]/(4*delta_t*delta_x[j+1])
            except:
                pass

            B[b, b] = 1/4
            B[b, b+1] = 1/4
            B[b, b+N] = -1/4
            B[b, b+N+1] = -1/4

            B[b, M*N + b] = mu[b]/(4*delta_t*delta_x[j])
            try:
                B[b, M*N + b+1] = mu[b+1]/(4*delta_t*delta_x[j+1])
            except:
                pass
            try:
                B[b, M*N + b+N] = mu[b+N]/(4*delta_t*delta_x[j])
            except:
                pass
            try:
                B[b, M*N + b+N+1] = mu[b+N+1]/(4*delta_t*delta_x[j+1])
            except:
                pass

            # update equation (4)
            try:
                A[M*N + b, b] = -epsilon[b]*delta_y[j]*(2*delta_t) - sigma[b]*(delta_y[j] + delta_y[j+1])/8
            except:
                pass
            try:
                A[M*N + b, b+N] = -epsilon[b+N]*delta_y[j]*(2*delta_t) - sigma[b+N]*(delta_y[j+1] + delta_y[j+2])/8
            except:
                pass
            A[M*N + b, M*N + b] = -1/4
            try:
                A[M*N + b, M*N + b+N] = 1/4
            except:
                pass

            B[M*N + b, b] = -epsilon[b]*delta_y[j]/(2*delta_t)
            try:
                B[M*N + b, b+N] = -epsilon[b+N]*delta_y[j+1]/(2*delta_t)
            except:
                pass

            B[M*N + b, M*N + b] = 1/4
            try:
                B[M*N + b, M*N + b+N] = 1/4
            except:
                pass

            C[M*N + b, M*N + b] = 1
            C[M*N + b, M*N + b-1] = 1

            D[M*N + b] = (delta_y[j]*jz[i+1,j,n] + delta_y[j]*jz[i,j,n] + delta_y[j]*jz[i+1,j,n-1] + delta_y[j]*jz[i,j,n-1])/4

    return [A, B, C, D]


def run():
    ez = np.ones(M*N)
    hy = np.zeros(M*N)
    bx = np.zeros(M*N)

    bx_list = np.zeros((M*N, 100))

    E = def_explicit_update_matrix()

    for n in range(iterations):
        print(f'iteration {n+1}/{iterations} started')
        [ez, hy, bx] = step(ez, hy, bx, E, n)
        bx_list[:,n] = bx
    return bx_list

def step(ez_old, hy_old, bx_old, E, n):
    bx_new = update_bx(bx_old, ez_old, E)
    [ez_new, hy_new] = update_implicit(ez_old, hy_old, bx_new, n)
    return [ez_new, hy_new, bx_new]


jz = def_jz(modulated)

bx_list = run()