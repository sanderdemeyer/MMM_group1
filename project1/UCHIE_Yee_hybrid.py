import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation

from functions import def_jz, def_update_matrices
epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8

M = 100
N = 100
iterations = 150

epsilon = np.ones((M,N))*epsilon_0
mu = np.ones((M,N))*mu_0
sigma = np.ones((M,N))*0

delta_x = np.ones(M)*10/M
delta_y = np.ones(N)*10/N
delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
delta_y_matrix = np.array([delta_y for i in range(M)])

courant_number = 1
delta_t = np.max(delta_y)/(c)*courant_number

eps_sigma_plus = epsilon/delta_t + sigma/2
eps_sigma_min = epsilon/delta_t - sigma/2

eq2_matrix = np.divide(delta_t, np.multiply(mu, delta_x_matrix))

# Define UCHIE region
# Define the 4 edges of the region
U_x_left = 5
U_x_right = 8
U_y_top = 10
U_y_bottom = 20
# Define how many layers in the x-direction the UCHIE region has (and y-direction, specified by the above)
U_x_size = 30 
U_y_size = U_y_top - U_y_bottom

source = 'dirac'

ez_Yee_new = np.zeros((M, N))
hy_Yee_new = np.zeros((M, N))
bx_Yee_new = np.zeros((M, N))

ez_U_new = np.zeros((U_x_size, U_y_size))
hy_U_new = np.zeros((U_x_size, U_y_size))
bx_U_new = np.zeros((U_x_size, U_y_size))


jz = def_jz(source, M, N, iterations, 1/(delta_x[0]*delta_y[0]))

[A, B] = def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t)

for n in range(iterations):
    print(f'iteration {n+1}/{iterations} started')

    ez_Yee_old = ez_Yee_new
    hy_Yee_old = hy_Yee_new
    bx_Yee_old = bx_Yee_new

    ez_U_old = ez_U_new
    hy_U_old = hy_U_new
    bx_U_old = bx_U_new

    eq_4_hy = np.divide(hy_Yee_new, delta_x_matrix)
    eq_4_bx = np.divide(bx_Yee_new, np.multiply(delta_y_matrix, mu))
    eq_4_term = np.multiply(eps_sigma_min, ez_Yee_old) - (jz[:,:,n]+jz[:,:,n-1])/2 + eq_4_hy - np.roll(eq_4_hy, 1, 0) - eq_4_bx + np.roll(eq_4_bx, 1, 1)
    ez_new = np.divide(eq_4_term, eps_sigma_plus)






