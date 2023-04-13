import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
import copy
from functions import def_jz, def_update_matrices, def_update_matrices_hybrid, update_implicit, update_implicit_hybrid, def_update_matrices_hybrid_new, update_implicit_hybrid_new, update_implicit_hybrid_zeros

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8


M_Yee = 500
N_Yee = 500
iterations = 250


M_U_separate = [5 for i in range(M_Yee)]
M_U = sum(M_U_separate)
N_U = N_Yee

# Define the location of the source

source_X_U = M_U//3
source_Y_U = N_U//3
source_X_Yee = 5
source_Y_Yee = 20

assert len(M_U_separate) == M_Yee, 'Length of M_U should be the same number as the number of Yee cells incompassed in the UCHIE region'


epsilon_Yee = np.ones((M_Yee,N_Yee))*epsilon_0
mM_Yee = np.ones((M_Yee,N_Yee))*mu_0
sigma_Yee = np.ones((M_Yee,N_Yee))*0

epsilon_U = np.ones((M_U,N_U))*epsilon_0
mu_U = np.ones((M_U,N_U))*mu_0
sigma_U = np.ones((M_U,N_U))*0

delta_x_Yee = np.ones(M_Yee)*10/M_Yee
delta_y_Yee = np.ones(N_Yee)*10/N_Yee
delta_x_matrix_Yee = np.array([np.repeat(delta_x_Yee[i], N_Yee) for i in range(M_Yee)])
delta_y_matrix_Yee = np.array([delta_y_Yee for i in range(M_Yee)])

#delta_x_U_fractions = np.array([[delta_x_Yee[U_x_left + i]/M_U_separate[i] for j in range(M_U_separate[i])] for i in range(M_Yee)])
delta_x_U_fractions = np.array([[1/M_U_separate[i] for j in range(M_U_separate[i])] for i in range(M_Yee)])
for i in range(N_Yee):
    pass
 #   assert abs(sum(delta_x_U_fractions[i,:])-1) < 10**(-7), 'sum of all individual fractions should be 1 for each Yee-cell'

delta_x_U_fractions_cumsumlist = np.array([np.cumsum(el) for el in delta_x_U_fractions])
M_U_separate_cumsumlist = np.cumsum(M_U_separate)
M_U_separate_cumsumlist = np.insert(M_U_separate_cumsumlist, 0, 0)

interpolate_matrix = np.zeros((M_U, M_Yee+1))
for i in range(M_Yee):
    interpolate_matrix[M_U_separate_cumsumlist[i]:M_U_separate_cumsumlist[i+1],i:i+2] = np.transpose([1-delta_x_U_fractions_cumsumlist[i], delta_x_U_fractions_cumsumlist[i]])


#delta_x_U = np.array([delta_x_Yee[i]/M_U_separate[i] for i in range(M_Yee)]).flatten()
delta_x_U = np.array([delta_x_U_fractions[i,:]*delta_x_Yee[i] for i in range(M_Yee)]).flatten()
delta_y_U = delta_y_Yee

delta_x_matrix_U = np.array([np.repeat(delta_x_U[i], N_U) for i in range(M_U)])
delta_y_matrix_U = np.array([delta_y_U for i in range(M_U)])


courant_number = 0.9

delta_t = (courant_number/c)*min(np.max(delta_y_U), 1/(np.sqrt(1/(np.max(delta_x_Yee))**2 + 1/(np.max(delta_y_Yee))**2)))

eps_sigma_plus_U = epsilon_U/delta_t + sigma_U/2
eps_sigma_min_U = epsilon_U/delta_t - sigma_U/2

eps_sigma_plus_Yee = epsilon_Yee/delta_t + sigma_Yee/2
eps_sigma_min_Yee = epsilon_Yee/delta_t - sigma_Yee/2

eq2_matrix = np.divide(delta_t, np.multiply(mM_Yee, delta_x_matrix_Yee))


ez_Yee_new = np.zeros((M_Yee, N_Yee))
hy_Yee_new = np.zeros((M_Yee, N_Yee))
bx_Yee_new = np.zeros((M_Yee, N_Yee))

ez_U_new = np.zeros((M_U, N_U))
hy_U_new = np.zeros((M_U, N_U))
bx_U_new = np.zeros((M_U, N_U))

source = 'gaussian_modulated'
jz_U = def_jz(0, M_U, N_U, source_X_U, source_Y_U, iterations, 1/(delta_x_U[0]*delta_y_U[0]))
jz_Yee = def_jz(source, M_Yee, N_Yee, source_X_Yee, source_Y_Yee, iterations, 1/(delta_x_Yee[0]*delta_y_Yee[0]))

print(f'jz_U = {np.sum(jz_U)}')
print(f'jz_Yee = {np.sum(jz_Yee)}')

[A, B] = def_update_matrices(epsilon_U, mu_U, sigma_U, delta_x_U, delta_y_U, delta_t, M_U)
A_inv = linalg.inv(A)

print(linalg.norm(A_inv))


bx_Yee_list = np.zeros((M_Yee, N_Yee, iterations))
ez_Yee_list = np.zeros((M_Yee, N_Yee, iterations))
hy_Yee_list = np.zeros((M_Yee, N_Yee, iterations))

bx_U_list = np.zeros((M_U, N_U, iterations))
ez_U_list = np.zeros((M_U, N_U, iterations))
hy_U_list = np.zeros((M_U, N_U, iterations))


for n in range(iterations):
    print(f'iteration {n+1}/{iterations} started')

    ez_Yee_old = ez_Yee_new
    hy_Yee_old = hy_Yee_new
    bx_Yee_old = bx_Yee_new

    ez_U_old = ez_U_new
    hy_U_old = hy_U_new
    bx_U_old = bx_U_new

    # Update ez in Yee region
    eq_4_hy = np.divide(hy_Yee_old, delta_x_matrix_Yee)
    eq_4_bx = np.divide(bx_Yee_old, np.multiply(delta_y_matrix_Yee, mM_Yee))
    eq_4_term = np.multiply(eps_sigma_min_Yee, ez_Yee_old) - (jz_Yee[:,:,n]+jz_Yee[:,:,n-1])/2 + eq_4_hy - np.roll(eq_4_hy, 1, 0) - eq_4_bx + np.roll(eq_4_bx, 1, 1)
    ez_Yee_new = np.divide(eq_4_term, eps_sigma_plus_Yee)

    ez_Yee_new[:,0] = np.zeros(M_Yee)

    """
    # Change the edges of the Yee-region: equate the bx-values around the UCHIE box and the ez-values on the left and right side to the just updated UCHIE values.
    ez_Yee_new[U_x_left+1:U_x_right-1,U_y_bottom+1:U_y_top-1] = np.zeros((M_Yee-2, N_U-2))
    bx_Yee_old[U_x_left+1:U_x_right-1,U_y_bottom+1:U_y_top-1] = np.zeros((M_Yee-2, N_U-2))
    hy_Yee_old[U_x_left+1:U_x_right-1,U_y_bottom+1:U_y_top-1] = np.zeros((M_Yee-2, N_U-2))
    """

    # Ez and Hy implicitly updated in the UCHIE region
#    [ez_U_new, hy_U_new] = update_implicit_hybrid_new(ez_U_old, hy_U_old, bx_U_old, n, A_inv, B, delta_t, delta_y_matrix_U, M_U, N_U, jz_U, mu_U, delta_x_Yee_left, delta_x_Yee_right, Ez_left_new, Ez_left_old, Ez_right_new, Ez_right_old, Hy_left, Hy_right)
    [ez_U_new, hy_U_new] = update_implicit(ez_U_old, hy_U_old, bx_U_old, n, A_inv, B, delta_t, delta_y_matrix_U, M_U, N_U, jz_U, mu_U)

    ez_U_new[:,0] = np.zeros(M_U)
    hy_U_new[:,0] = np.zeros(M_U)

    # Bx explicitly updated in the UCHIE region (interpolation needed)

    bx_U_new = np.zeros((M_U, N_U))
    bx_U_new[:,1:-1] = bx_U_old[:,1:-1] - (ez_U_new[:,2:] - ez_U_new[:,1:-1])
    # top
    ez_top = list(ez_Yee_new[:,1])
    ez_top.append(ez_Yee_new[0,1])
    ez_top = np.array(ez_top)
    ez_bottom = list(ez_Yee_new[:,-1])
    ez_bottom.append(ez_Yee_new[0,-1])
    ez_bottom = np.array(ez_bottom)

    if n < 15:
        print('top')
        print(np.dot(interpolate_matrix, ez_top))
        print(ez_U_new[:,-1])
        print('bottom')
        print(np.dot(interpolate_matrix, ez_bottom))
        print(ez_U_new[:,1])
    bx_U_new[:,-1] = bx_U_old[:,-1] - (np.dot(interpolate_matrix, ez_top)*delta_t - ez_U_new[:,-1]) # add periodic boundary condition
    # ez_U[:,0] does not exist, or equivalently, is always zero.
    # bottom
    bx_U_new[:,0] = bx_U_old[:,0] - (ez_U_new[:,1] - np.dot(interpolate_matrix, ez_bottom)*delta_t) # add periodic boundary condition



    """ #Should be executed, but does not work yet.
    # Change the edges of the Yee-region: equate the bx-values around the UCHIE box and the ez-values on the left and right side to the just updated UCHIE values.

    bx_Yee_old[U_x_left,U_y_bottom:U_y_top] = bx_U_new[0,:]
    bx_Yee_old[U_x_right,U_y_bottom:U_y_top] = bx_U_new[-1,:]
    bx_Yee_old[U_x_right, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[-1]-1,0]
    bx_Yee_old[U_x_right, U_y_top] = bx_U_new[M_U_separate_cumsumlist[-1]-1,-1]
    """    

    # Bx is explicitly updated in the Yee region
    eq_3_term = np.divide(ez_Yee_new*delta_t, delta_y_matrix_Yee)
    bx_Yee_new = bx_Yee_old - np.roll(eq_3_term, -1, 1) + eq_3_term

    # Hy is explicitly updated in the Yee region
    eq_2_term = np.multiply(eq2_matrix, ez_Yee_new)
    hy_Yee_new = hy_Yee_old + np.roll(eq_2_term, -1, 0) - eq_2_term

    hy_Yee_new[:,0] = np.zeros(M_Yee)

    for i in range(M_Yee):
        bx_Yee_new[i, 0] = bx_U_new[M_U_separate_cumsumlist[i],-1]/delta_y_U[-1]
        bx_Yee_new[i, -1] = bx_U_new[M_U_separate_cumsumlist[i],0]/delta_y_U[0]



    bx_Yee_list[:,:,n] = bx_Yee_new
    ez_Yee_list[:,:,n] = ez_Yee_new
    hy_Yee_list[:,:,n] = hy_Yee_new

    bx_U_list[:,:,n] = bx_U_new
    ez_U_list[:,:,n] = ez_U_new
    hy_U_list[:,:,n] = hy_U_new

animation_speed = 1

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(np.transpose(ez_Yee_list[:,:,int(i*animation_speed)]))
   ax.set_title(f'n = {int(i*animation_speed)}')


anim = FuncAnimation(fig, animate)
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(np.transpose(ez_U_list[:,:,int(i*animation_speed)]))
   ax.set_title(f'n = {int(i*animation_speed)}')


anim = FuncAnimation(fig, animate)
plt.show()
