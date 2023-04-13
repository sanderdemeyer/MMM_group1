import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
from functions import def_update_matrices, update_implicit, def_jz
import scipy.optimize as opt
#from Yee_students/Main.py import output_function


# THIS IS THE MAIN UCHIE FILE

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8
M = 100
N = 100

iterations = 60

#source should be either 'sine', 'gaussian_modulated', or 'gaussian'


# last 'extra' element should be the same as the 0th.
epsilon = np.ones((M,N))*epsilon_0
mu = np.ones((M,N))*mu_0
sigma = np.ones((M,N))*0

delta_x = np.ones(M)*10/M/10
delta_y = np.ones(N)*10/N/10

#delta_x = [((i+1)**(1/10))*10/M for i in range(M)]

delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
delta_y_matrix = np.array([delta_y for i in range(M)])

courant_number = 1
delta_t = np.min(delta_y)/(c)*courant_number
print(delta_t)

source = 'dirac'
x_source = 50
y_source = 50
J0 = 1
tc = 5
sigma_source = 1
period = 10
omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps

jz = def_jz(source, M, N, x_source, y_source, iterations, delta_t)

spectral_content = fft.fft(jz[x_source,y_source,:])[0]
jz = jz/spectral_content

# period for the wave to go around = 
# T = delta_t*n = N delta_y / c
# n = N delta_y/(c*delta_t) = N delta_y/(c*courant*delta_y/c) = N/courant

observation_points_ez = [(15, 0), (0, 15), (15, 15)]

#observation_points_ez = [(0, i) for i in range(M)]
observation_points_ez = [(x_source + i, y_source) for i in range(M//2)]

def def_jz_OLD(source):
    jz = np.zeros((M, N, iterations))
    if source == 'gaussian_modulated':
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.exp(-((i-x_point)**2 + j**2)/(2*sigma_source**2))
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

def update_implicit_OLD(ez_old, hy_old, bx, n, A_inv, B):

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

def def_update_matrices_OLD(epsilon, mu, sigma, delta_x, delta_y, delta_t):
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

def step_OLD(ez_old, hy_old, bx_old, A_inv, B, n):
    bx_new = update_bx(bx_old, ez_old)
    [ez_new, hy_new] = update_implicit(ez_old, hy_old, bx_new, n, A_inv, B, delta_t, delta_y_matrix, M, N, jz, mu)
    return [ez_new, hy_new, bx_new]

def step(ez_old, hy_old, bx_old, A_inv, B, n):
    [ez_new, hy_new] = update_implicit(ez_old, hy_old, bx_old, n, A_inv, B, delta_t, delta_y_matrix, M, N, jz, mu)
    bx_new = update_bx(bx_old, ez_new)
    return [ez_new, hy_new, bx_new]


def run_UCHIE():
    ez = np.zeros((M,N))
    hy = np.zeros((M,N))
    bx = np.zeros((M,N))

    bx_list = np.zeros((M,N, iterations))
    ez_list = np.zeros((M,N, iterations))
    hy_list = np.zeros((M,N, iterations))

    ez_list_observe = np.zeros((iterations, len(observation_points_ez)))

    [A, B] = def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, M)
    A_inv = linalg.inv(A)

    for n in range(iterations):
        print(f'iteration {n+1}/{iterations} started')
        [ez, hy, bx] = step(ez, hy, bx, A_inv, B, n)
        bx_list[:,:,n] = bx
        ez_list[:,:,n] = ez
        hy_list[:,:,n] = hy

        for i, point in enumerate(observation_points_ez):
            ez_list_observe[n, i] = ez[point]

    return bx_list, ez_list, hy_list, ez_list_observe



[bx_list, ez_list, hy_list, ez_list_observe] = run_UCHIE()


def hankel(x, omega, J0=1):
    return -(J0*omega*mu_0/4)*special.hankel2(0, (omega*x/c))

frequency_point = 20

fft_transform_r_values = [i*delta_x[0] for i in range(M//2)]
fft_list = []

plt.plot(range(iterations), ez_list_observe[:,40])
plt.show()

for i, point in enumerate(observation_points_ez):
    """
    plt.plot(range(iterations), ez_list_observe[:,i])
    plt.xlabel('Time [s]')
    plt.ylabel('Ez')
    plt.title(f'Ez at {point}')
    #plt.show()
    """
    fft_transform = fft.fft(ez_list_observe[:,i])
    #plt.plot(fft_transform)
    #plt.show()
    fft_list.append(fft_transform[frequency_point])

omega = 1/delta_t*frequency_point/delta_x[0]*1
print(omega)

omega = frequency_point/(iterations*delta_t)*6.25

print(omega)
plt.plot(fft_transform_r_values, fft_list, label = 'computational')
plt.plot(fft_transform_r_values, np.array([hankel(i, omega) for i in fft_transform_r_values])*10**(-4.5), label = 'analytical')
plt.legend()
plt.show()

plt.plot(fft_transform_r_values, fft_list, label = 'computational')
plt.plot(np.array([hankel(delta_x[0]*i, 1/delta_t*frequency_point) for i in range(len(observation_points_ez))])/(3*10**(12)), label = 'analytical')
plt.legend()
plt.show()


"""
plt.plot(d_list[10:-10], [i/2 for i in v_list[10:-10]], label = 'computationally')
#plt.plot(d_list, [hankel(x+1) for x in d_list], label = 'exact solution')
plt.legend()
plt.show()
"""
plt.show()

plt.plot(ez_list[x_source, y_source*3//2,:])
plt.show()

animation_speed = 1

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(ez_list[:,:,int(i*animation_speed)])
   ax.set_title(f'n = {int(i*animation_speed)}')


print(bx_list[:,:,10])
print(ez_list[:,:,10])
print(hy_list[:,:,10])

anim = FuncAnimation(fig, animate)
plt.show()
