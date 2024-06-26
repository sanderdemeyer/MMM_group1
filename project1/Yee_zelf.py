import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
import copy
from functions import def_jz

def def_jz_OLD(source, M, N, iterations, x_point, y_point):
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
        jz[x_point, y_point, 0] = 1/(delta_x[0]*delta_y[0])
    return jz

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8
M = 100
N = 100

x_source = 50
y_source = 50

observation_points_ez = [(i, 0) for i in range(M)]
observation_points_ez = [(x_source + i, y_source) for i in range(M//2)]
iterations = 150


epsilon = np.ones((M,N))*epsilon_0
mu = np.ones((M,N))*mu_0
sigma = np.zeros((M,N))

delta_x = np.ones(M)*10/M
delta_y = np.ones(N)*10/N
delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
delta_y_matrix = np.array([delta_y for i in range(M)])

courant_number = 1
delta_t = courant_number/(c*np.sqrt(1/(np.max(delta_x))**2 + 1/(np.max(delta_y))**2))
print(delta_t)

eps_sigma_plus = epsilon/delta_t + sigma/2
eps_sigma_min = epsilon/delta_t - sigma/2

eq2_matrix = np.divide(delta_t, np.multiply(mu, delta_x_matrix))


#source should be either 'sine', 'gaussian_modulated', or 'gaussian'
source = 'gaussian_modulated'
J0 = 1
tc = 5
sigma_source = 0.2
period = 10
omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps
omega_c = (2*np.pi)/(period*10**(-9)/3)
jz = def_jz(source, M, N, x_source, y_source, iterations, delta_t)


ez_new = np.zeros((M, N))
hy_new = np.zeros((M, N))
bx_new = np.zeros((M, N))


bx_list = np.zeros((M,N, iterations))
ez_list = np.zeros((M,N, iterations))
hy_list = np.zeros((M,N, iterations))

ez_observation_list = np.zeros((iterations, len(observation_points_ez)))
""" Correct
for n in range(iterations):
    print(f'iteration {n+1}/{iterations} started')
    ez_old = ez_new
    hy_old = hy_new
    bx_old = bx_new

    eq_2_term = np.multiply(eq2_matrix, ez_old)
    hy_new = hy_old + np.roll(eq_2_term, -1, 0) - eq_2_term

    eq_3_term = np.divide(ez_old*delta_t, delta_y_matrix)
    bx_new = bx_old - np.roll(eq_3_term, -1, 1) + eq_3_term
    
    eq_4_hy = np.divide(hy_new, delta_x_matrix)
    eq_4_bx = np.divide(bx_new, np.multiply(delta_y_matrix, mu))
    # with source averaging
    #eq_4_term = np.multiply(eps_sigma_min, ez_old) - (jz[:,:,n]+jz[:,:,n-1])/2 + eq_4_hy - np.roll(eq_4_hy, 1, 0) - eq_4_bx + np.roll(eq_4_bx, 1, 1)
    # without source averaging
    eq_4_term = np.multiply(eps_sigma_min, ez_old) - jz[:,:,n] + eq_4_hy - np.roll(eq_4_hy, 1, 0) - eq_4_bx + np.roll(eq_4_bx, 1, 1)
    ez_new = np.divide(eq_4_term, eps_sigma_plus)

    bx_list[:,:,n] = bx_new
    ez_list[:,:,n] = ez_new
    hy_list[:,:,n] = hy_new

    for i, point in enumerate(observation_points_ez):
        ez_observation_list[n, i] = ez_new[point]
"""
for n in range(iterations):
    print(f'iteration {n+1}/{iterations} started')
    """
    ez_old = copy.deepcopy(ez_new)
    hy_old = copy.deepcopy(hy_new)
    bx_old = copy.deepcopy(bx_new)
    """
    ez_old = ez_new
    hy_old = hy_new
    bx_old = bx_new

    eq_2_term = np.multiply(eq2_matrix, ez_old)
    hy_new = hy_old + np.roll(eq_2_term, -1, 0) - eq_2_term

    eq_3_term = np.divide(ez_old*delta_t, delta_y_matrix)
    bx_new = bx_old - np.roll(eq_3_term, -1, 1) + eq_3_term
    
    eq_4_hy = np.divide(hy_new, delta_x_matrix)
    eq_4_bx = np.divide(bx_new, np.multiply(delta_y_matrix, mu))
    # with source averaging
    eq_4_term = np.multiply(eps_sigma_min, ez_old) - (jz[:,:,n]+jz[:,:,n-1])/2 + eq_4_hy - np.roll(eq_4_hy, 1, 0) - eq_4_bx + np.roll(eq_4_bx, 1, 1)
    # without source averaging
    #eq_4_term = np.multiply(eps_sigma_min, ez_old) - jz[:,:,n] + eq_4_hy - np.roll(eq_4_hy, 1, 0) - eq_4_bx + np.roll(eq_4_bx, 1, 1)
    ez_new = np.divide(eq_4_term, eps_sigma_plus)

    bx_list[:,:,n] = bx_new
    ez_list[:,:,n] = ez_new
    hy_list[:,:,n] = hy_new

    for i, point in enumerate(observation_points_ez):
        ez_observation_list[n, i] = ez_new[point]

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

"""
fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(ez_list[:,:,int(i*animation_speed)])
   ax.set_title(f'n = {int(i*animation_speed)}')


anim = FuncAnimation(fig, animate)
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(hy_list[:,:,int(i*animation_speed)])
   ax.set_title(f'n = {int(i*animation_speed)}')


anim = FuncAnimation(fig, animate)
plt.show()


print(bx_list[:,:,50])
"""


fft_trafod = [fft.fft(ez_observation_list[:,i])[1] for i in range(len(observation_points_ez))]


plt.plot(ez_list[70, 50, :])
plt.show()