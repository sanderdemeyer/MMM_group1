import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8
M = 100
N = 100

iterations = 150

#source should be either 'sine', 'gaussian_modulated', or 'gaussian'


# last 'extra' element should be the same as the 0th.
epsilon = np.ones((M,N))*epsilon_0
mu = np.ones((M,N))*mu_0
sigma = np.ones((M,N))*0

delta_x = np.ones(M)*10/M
delta_y = np.ones(N)*10/N
delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
delta_y_matrix = np.array([delta_y for i in range(M)])

courant_number = 1
delta_t = np.max(delta_y)/(c)*courant_number
print(delta_t)

source = 'dirac'
J0 = 1
tc = 5
sigma_source = 1
period = 10
omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps

# period for the wave to go around = 
# T = delta_t*n = N delta_y / c
# n = N delta_y/(c*delta_t) = N delta_y/(c*courant*delta_y/c) = N/courant

observation_points_ez = [(15, 0), (0, 15), (15, 15)]

#observation_points_ez = [(0, i) for i in range(M)]
observation_points_ez = [(i, 0) for i in range(M)]

def def_jz(source):
    jz = np.zeros((M, N, iterations))
    if source == 'gaussian_modulated':
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    elif source == 'gaussian':
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.sin(omega_c*n*delta_t)*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    elif source == 'sine':
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.sin(omega_c*n*delta_t)*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    elif source == 'dirac':
        jz[M//3, N//4, 0] = 1/(delta_x[0]*delta_y[0])
    return jz

def update_bx(bx_old, ez_old):
    #assumes arrays
    #bx = bx_old + np.dot(E, ez_old)
    #return bx

    #assumes matrices
    bx = np.zeros((M, N))
    bx[:,:-1] = bx_old[:,:-1] - (ez_old[:,1:] - ez_old[:,:-1])
    bx[:,-1] = bx_old[:,-1] - (ez_old[:,0] - ez_old[:,-1]) # add periodic boundary condition
    return bx

def update_implicit(ez_old, hy_old, bx, n, A_inv, B):

    bx_term = -(delta_t/2)*np.divide(bx, np.multiply(mu, delta_y_matrix))
    C_term_base = np.roll(bx_term, -1, 0) + bx_term - np.roll(np.roll(bx_term, -1, 0), 1, 1) - np.roll(bx_term, 1, 1)
    C_term = np.concatenate((np.zeros((M, N)), C_term_base))
    # should be delta_y_star_matrix
    jz_n = -np.multiply(delta_y_matrix, jz[:,:,n])/4
    jz_nm1 = -np.multiply(delta_y_matrix, jz[:,:,n-1])/4
    D_term_base = np.roll(jz_n, -1, 0) + jz_n + np.roll(jz_nm1, -1, 0) + jz_nm1
    D_term = np.concatenate((np.zeros((M, N)), D_term_base))

    new_values = np.dot(A_inv, (np.dot(B, np.concatenate((ez_old, hy_old))) + C_term + D_term))
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

def def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t):
    A = np.zeros((2*M, 2*M))
    B = np.zeros((2*M, 2*M))

    for i in range(M):

        if i != M-1:
            A[i,i+1] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,i+1] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) + sigma[i+1,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M+i+1] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) - sigma[i+1,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M+i+1] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])

        else:
            A[i,0] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M] = -mu[0,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,0] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M] = -mu[0,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,0] = (epsilon[0,0]/(2*delta_t) + sigma[0,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,0] = (epsilon[0,0]/(2*delta_t) - sigma[0,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])
    return [A, B]


def run_UCHIE():
    ez = np.zeros((M,N))
    hy = np.zeros((M,N))
    bx = np.zeros((M,N))

    bx_list = np.zeros((M,N, iterations))

    ez_list = np.zeros((iterations, len(observation_points_ez)))

    [A, B] = def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t)
    A_inv = linalg.inv(A)
    for n in range(iterations):
        print(f'iteration {n+1}/{iterations} started')
        [ez, hy, bx] = step(ez, hy, bx, A_inv, B, n)
        bx_list[:,:,n] = bx

        for i, point in enumerate(observation_points_ez):
            ez_list[n, i] = ez[point]

    return bx_list, ez_list

def step(ez_old, hy_old, bx_old, A_inv, B, n):
    bx_new = update_bx(bx_old, ez_old)
    [ez_new, hy_new] = update_implicit(ez_old, hy_old, bx_new, n, A_inv, B)
    return [ez_new, hy_new, bx_new]


jz = def_jz(source)


[bx_list, ez_list] = run_UCHIE()


def hankel(x):
    return -(J0*omega_c*mu_0/4)*special.hankel2(0, (omega_c*x*delta_x[0]/c))

d_list = []
v_list = []
for i, point in enumerate(observation_points_ez):
    """
    plt.plot(range(iterations), ez_list[:,i])
    plt.xlabel('Time [s]')
    plt.ylabel('Ez')
    plt.title(f'Ez at {point}')
    plt.show()
    """
    fft_transform = fft.fft(ez_list[:,i])
    #d_list.append(point[1])
    d_list.append(point[0])
    v_list.append(fft_transform[iterations//period])

plt.plot(d_list[10:-10], [i/2 for i in v_list[10:-10]], label = 'computationally')
#plt.plot(d_list, [hankel(x+1) for x in d_list], label = 'exact solution')
plt.legend()
plt.show()


animation_speed = 1

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(bx_list[:,:,int(i*animation_speed)])
   ax.set_title(f'n = {int(i*animation_speed)}')


anim = FuncAnimation(fig, animate)
plt.show()
