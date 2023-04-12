import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation

from functions import def_jz, def_update_matrices, def_update_matrices_hybrid, update_implicit, update_implicit_hybrid

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8

M_Yee = 100
N_Yee = 100
iterations = 150

# Define UCHIE region
# Define the 4 edges of the region
U_x_left = 20
U_x_right = 80
U_y_top = 80
U_y_bottom = 20
# Define how many layers in the x-direction the UCHIE region has (and y-direction, specified by the above)
U_Yee = U_x_right - U_x_left
M_U_separate = [5 for i in range(U_Yee)]
M_U = sum(M_U_separate)
N_U = U_y_top - U_y_bottom

# Define the location of the source

source_X_U = M_U//3
source_Y_U = N_U//3
source_X_Yee = M_Yee//2
source_Y_Yee = 5

assert len(M_U_separate) == U_Yee, 'Length of M_U should be the same number as the number of Yee cells incompassed in the UCHIE region'


epsilon_Yee = np.ones((M_Yee,N_Yee))*epsilon_0
mu_Yee = np.ones((M_Yee,N_Yee))*mu_0
sigma_Yee = np.ones((M_Yee,N_Yee))*0

epsilon_U = np.ones((M_U,N_U))*epsilon_0
mu_U = np.ones((M_U,N_U))*mu_0
sigma_U = np.ones((M_U,N_U))*0

delta_x_Yee = np.ones(M_Yee)*10/M_Yee
delta_y_Yee = np.ones(N_Yee)*10/N_Yee
delta_x_matrix_Yee = np.array([np.repeat(delta_x_Yee[i], N_Yee) for i in range(M_Yee)])
delta_y_matrix_Yee = np.array([delta_y_Yee for i in range(M_Yee)])

#delta_x_U_fractions = np.array([[delta_x_Yee[U_x_left + i]/M_U_separate[i] for j in range(M_U_separate[i])] for i in range(U_Yee)])
delta_x_U_fractions = np.array([[1/M_U_separate[i] for j in range(M_U_separate[i])] for i in range(U_Yee)])
for i in range(U_Yee):
    assert abs(sum(delta_x_U_fractions[i,:])-1) < 10**(-7), 'sum of all individual fractions should be 1 for each Yee-cell'

delta_x_U_fractions_cumsumlist = np.array([np.cumsum(el) for el in delta_x_U_fractions])
M_U_separate_cumsumlist = np.cumsum(M_U_separate)
M_U_separate_cumsumlist = np.insert(M_U_separate_cumsumlist, 0, 0)

interpolate_matrix = np.zeros((M_U, U_Yee+1))
for i in range(U_Yee):
    interpolate_matrix[M_U_separate_cumsumlist[i]:M_U_separate_cumsumlist[i+1],i:i+2] = np.transpose([1-delta_x_U_fractions_cumsumlist[i], delta_x_U_fractions_cumsumlist[i]])


#delta_x_U = np.array([delta_x_Yee[i]/M_U_separate[i] for i in range(U_Yee)]).flatten()
delta_x_U = np.array([delta_x_U_fractions[i,:]*delta_x_Yee[U_x_left+i] for i in range(U_Yee)]).flatten()
delta_y_U = delta_y_Yee[U_y_bottom:U_y_top]

delta_x_matrix_U = np.array([np.repeat(delta_x_U[i], N_U) for i in range(M_U)])
delta_y_matrix_U = np.array([delta_y_U for i in range(M_U)])


courant_number = 0.9

delta_t = (courant_number/c)*min(np.max(delta_y_U), 1/(np.sqrt(1/(np.max(delta_x_Yee))**2 + 1/(np.max(delta_y_Yee))**2)))

eps_sigma_plus_U = epsilon_U/delta_t + sigma_U/2
eps_sigma_min_U = epsilon_U/delta_t - sigma_U/2

eps_sigma_plus_Yee = epsilon_Yee/delta_t + sigma_Yee/2
eps_sigma_min_Yee = epsilon_Yee/delta_t - sigma_Yee/2

eq2_matrix = np.divide(delta_t, np.multiply(mu_Yee, delta_x_matrix_Yee))


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
delta_x_Yee_left = delta_x_Yee[U_x_left]
delta_x_Yee_right = delta_x_Yee[U_x_right]
#[A, B] = def_update_matrices_hybrid(epsilon_U, mu_U, sigma_U, delta_x_U, delta_y_U, delta_t, M_U, delta_x_Yee_left, delta_x_Yee_right)
A_inv = linalg.inv(A)

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
    eq_4_bx = np.divide(bx_Yee_old, np.multiply(delta_y_matrix_Yee, mu_Yee))
    eq_4_term = np.multiply(eps_sigma_min_Yee, ez_Yee_old) - (jz_Yee[:,:,n]+jz_Yee[:,:,n-1])/2 + eq_4_hy - np.roll(eq_4_hy, 1, 0) - eq_4_bx + np.roll(eq_4_bx, 1, 1)
    ez_Yee_new = np.divide(eq_4_term, eps_sigma_plus_Yee)

    """
    ez_U_old[0,:] = ez_Yee_new[U_x_left,U_y_bottom:U_y_top]
    ez_U_old[-1,:] = ez_Yee_new[U_x_right,U_y_bottom:U_y_top]

    bx_U_old[0,:] = bx_Yee_new[U_x_left,U_y_bottom:U_y_top]
    bx_U_old[-1,:] = bx_Yee_new[U_x_right,U_y_bottom:U_y_top]

    hy_U_old[0,:] = hy_Yee_new[U_x_left,U_y_bottom:U_y_top]
    hy_U_old[-1,:] = hy_Yee_new[U_x_right,U_y_bottom:U_y_top]
    """

    # Ez and Hy implicitly updated in the UCHIE region
    [ez_U_new, hy_U_new] = update_implicit(ez_U_old, hy_U_old, bx_U_old, n, A_inv, B, delta_t, delta_y_matrix_U, M_U, N_U, jz_U, mu_U)
    Hy_Yee_left = hy_Yee_new[U_x_left,U_y_bottom:U_y_top]
    Hy_Yee_right = hy_Yee_new[U_x_right,U_y_bottom:U_y_top]
    Ez_Yee_left = ez_Yee_new[U_x_left,U_y_bottom:U_y_top]
    Ez_Yee_right = ez_Yee_new[U_x_right,U_y_bottom:U_y_top]


    #[ez_U_new, hy_U_new] = update_implicit_hybrid(ez_U_old, hy_U_old, bx_U_old, n, A_inv, B, delta_t, delta_y_matrix_U, M_U, N_U, jz_U, mu_U, delta_x_Yee_left, delta_x_Yee_right, Hy_Yee_left, Hy_Yee_right, Ez_Yee_left, Ez_Yee_right)

    # Bx explicitly updated in the UCHIE region (interpolation needed)

    bx_U_new = np.zeros((M_U, N_U))
    bx_U_new[:,1:-1] = bx_U_old[:,1:-1] - (ez_U_new[:,2:] - ez_U_new[:,1:-1])
    bx_U_new[:,-1] = bx_U_old[:,-1] - (np.dot(interpolate_matrix, ez_Yee_new[U_x_left:U_x_right+1,U_y_top+1]) - ez_U_new[:,-1]) # add periodic boundary condition
    # ez_U[:,0] does not exist, or equivalently, is always zero.
    bx_U_new[:,0] = bx_U_old[:,0] - (ez_U_new[:,1] - np.dot(interpolate_matrix, ez_Yee_new[U_x_left:U_x_right+1,U_y_bottom])) # add periodic boundary condition

    """    #Should be executed, but does not work yet.
    # Change the edges of the Yee-region: equate the bx-values around the UCHIE box and the ez-values on the left and right side to the just updated UCHIE values.
    ez_Yee_new[U_x_left,U_y_bottom:U_y_top] = ez_U_new[0,:]
    ez_Yee_new[U_x_right,U_y_bottom:U_y_top] = ez_U_new[-1,:]

    bx_Yee_old[U_x_left,U_y_bottom:U_y_top] = bx_U_new[0,:]
    bx_Yee_old[U_x_right,U_y_bottom:U_y_top] = bx_U_new[-1,:]
    for i in range(U_Yee):
        bx_Yee_old[U_x_left+i, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[i],0]
        bx_Yee_old[U_x_left+i, U_y_top] = bx_U_new[M_U_separate_cumsumlist[i],-1]
   # bx_Yee_old[U_x_right, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[-1]-1,0]
   # bx_Yee_old[U_x_right, U_y_top] = bx_U_new[M_U_separate_cumsumlist[-1]-1,-1]
    """


    """ wrong, I think
    bx_Yee_new[U_x_left,U_y_bottom:U_y_top] = bx_U_new[0,:]
    bx_Yee_new[U_x_right,U_y_bottom:U_y_top] = bx_U_new[-1,:]
    for i in range(U_Yee):
        bx_Yee_new[U_x_left+i, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[i],0]
        bx_Yee_new[U_x_left+i, U_y_top] = bx_U_new[M_U_separate_cumsumlist[i],-1]
    bx_Yee_new[U_x_right, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[-1]-1,0]
    bx_Yee_new[U_x_right, U_y_top] = bx_U_new[M_U_separate_cumsumlist[-1]-1,-1]
    """

    # Bx is explicitly updated in the Yee region
    eq_3_term = np.divide(ez_Yee_new*delta_t, delta_y_matrix_Yee)
    bx_Yee_new = bx_Yee_old - np.roll(eq_3_term, -1, 1) + eq_3_term

    # Hy is explicitly updated in the Yee region
    eq_2_term = np.multiply(eq2_matrix, ez_Yee_new)
    hy_Yee_new = hy_Yee_old + np.roll(eq_2_term, -1, 0) - eq_2_term





    """ weird shit. Probably wrong
    term2_plus = delta_x_Yee[U_x_right]*(np.divide(bx_U_new[-1,U_y_bottom+1:U_y_top], 2*np.multiply(mu_Yee[U_x_right,U_y_bottom+1:U_y_top], delta_y_Yee[U_y_bottom+1:U_y_top])))
    term2_min = -delta_x_Yee[U_x_right]*(np.divide(bx_U_new[-1,U_y_bottom:U_y_top-1], 2*np.multiply(mu_Yee[U_x_right,U_y_bottom:U_y_top-1], delta_y_Yee[U_y_bottom:U_y_top-1])))
    term3 = np.multiply(delta_x_Yee[U_x_right]/2, np.multiply(epsilon_Yee[U_x_right,U_y_bottom+1:U_y_top]/delta_t + sigma_Yee[U_x_right,U_y_bottom+1:U_y_top]/2, ez_U_new[-1,U_y_bottom+1:U_y_top]))
    term4 = -np.multiply(delta_x_Yee[U_x_right]/2, np.multiply(epsilon_Yee[U_x_right,U_y_bottom+1:U_y_top]/delta_t - sigma_Yee[U_x_right,U_y_bottom+1:U_y_top]/2, ez_U_old[-1,U_y_bottom+1:U_y_top]))
    term5 = np.multiply(delta_x_Yee[U_x_right]/4, jz_U[-1,U_y_bottom+1:U_y_top,n] + jz_U[-1,U_y_bottom+1:U_y_top,n-1])
    """
    """
    term1 = (hy_U_new[-1,1:] + hy_U_old[-1,1:])/2
    term2_plus = delta_x_Yee[U_x_right]*(np.divide(bx_U_new[-1,1:], 2*np.multiply(mu_Yee[U_x_right,U_y_bottom+1:U_y_top], delta_y_Yee[U_y_bottom+1:U_y_top])))
    term2_min = -delta_x_Yee[U_x_right]*(np.divide(bx_U_new[-1,:-1], 2*np.multiply(mu_Yee[U_x_right,U_y_bottom:U_y_top-1], delta_y_Yee[U_y_bottom:U_y_top-1])))
    term3 = np.multiply(delta_x_Yee[U_x_right]/2, np.multiply(epsilon_Yee[U_x_right,U_y_bottom+1:U_y_top]/delta_t + sigma_Yee[U_x_right,U_y_bottom+1:U_y_top]/2, ez_U_new[-1,1:]))
    term4 = -np.multiply(delta_x_Yee[U_x_right]/2, np.multiply(epsilon_Yee[U_x_right,U_y_bottom+1:U_y_top]/delta_t - sigma_Yee[U_x_right,U_y_bottom+1:U_y_top]/2, ez_U_old[-1,1:]))
    term5 = np.multiply(delta_x_Yee[U_x_right]/4, jz_U[-1,1:,n] + jz_U[-1,1:,n-1])

    hy_Yee_new[U_x_right,U_y_bottom+1:U_y_top] = term1 + term2_plus + term2_min + term3 + term4 + term5

    term1 = (hy_U_new[0,1:] + hy_U_old[0,1:])/2
    term2_plus = -delta_x_Yee[U_x_left]*(np.divide(bx_U_new[0,1:], 2*np.multiply(mu_Yee[U_x_left,U_y_bottom+1:U_y_top], delta_y_Yee[U_y_bottom+1:U_y_top])))
    term2_min = delta_x_Yee[U_x_left]*(np.divide(bx_U_new[0,:-1], 2*np.multiply(mu_Yee[U_x_left,U_y_bottom:U_y_top-1], delta_y_Yee[U_y_bottom:U_y_top-1])))
    term3 = -np.multiply(delta_x_Yee[U_x_left]/2, np.multiply(epsilon_Yee[U_x_left,U_y_bottom+1:U_y_top]/delta_t + sigma_Yee[U_x_left,U_y_bottom+1:U_y_top]/2, ez_U_new[0,1:]))
    term4 = np.multiply(delta_x_Yee[U_x_left]/2, np.multiply(epsilon_Yee[U_x_left,U_y_bottom+1:U_y_top]/delta_t - sigma_Yee[U_x_left,U_y_bottom+1:U_y_top]/2, ez_U_old[0,1:]))
    term5 = -np.multiply(delta_x_Yee[U_x_left]/4, jz_U[0,1:,n] + jz_U[0,1:,n-1])

    hy_Yee_new[U_x_left-1,U_y_bottom+1:U_y_top] = term1 + term2_plus + term2_min + term3 + term4 + term5

    hy_Yee_new[U_x_left:U_x_right,U_y_bottom:U_y_top] = np.zeros((U_Yee, N_U))

    hy_Yee_new[U_x_left,U_y_bottom:U_y_top] = hy_U_new[0,:]
    hy_Yee_new[U_x_right,U_y_bottom:U_y_top] = hy_U_new[-1,:]
    for i in range(U_Yee):
        hy_Yee_new[U_x_left+i, U_y_bottom] = hy_U_new[M_U_separate_cumsumlist[i],0]
        hy_Yee_new[U_x_left+i, U_y_top] = hy_U_new[M_U_separate_cumsumlist[i],-1]
    hy_Yee_new[U_x_right, U_y_bottom] = hy_U_new[M_U_separate_cumsumlist[-1]-1,0]
    hy_Yee_new[U_x_right, U_y_top] = hy_U_new[M_U_separate_cumsumlist[-1]-1,-1]

    ez_Yee_new[U_x_left,U_y_bottom:U_y_top] = ez_U_new[0,:]
    ez_Yee_new[U_x_right,U_y_bottom:U_y_top] = ez_U_new[-1,:]
    for i in range(U_Yee):
        ez_Yee_new[U_x_left+i, U_y_bottom] = ez_U_new[M_U_separate_cumsumlist[i],0]
        ez_Yee_new[U_x_left+i, U_y_top] = ez_U_new[M_U_separate_cumsumlist[i],-1]
    ez_Yee_new[U_x_right, U_y_bottom] = ez_U_new[M_U_separate_cumsumlist[-1]-1,0]
    ez_Yee_new[U_x_right, U_y_top] = ez_U_new[M_U_separate_cumsumlist[-1]-1,-1]

    bx_Yee_new[U_x_left,U_y_bottom:U_y_top] = bx_U_new[0,:]
    bx_Yee_new[U_x_right,U_y_bottom:U_y_top] = bx_U_new[-1,:]
    for i in range(U_Yee):
        bx_Yee_new[U_x_left+i, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[i],0]
        bx_Yee_new[U_x_left+i, U_y_top] = bx_U_new[M_U_separate_cumsumlist[i],-1]
    bx_Yee_new[U_x_right, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[-1]-1,0]
    bx_Yee_new[U_x_right, U_y_top] = bx_U_new[M_U_separate_cumsumlist[-1]-1,-1]

    """
    bx_Yee_list[:,:,n] = bx_Yee_new
    ez_Yee_list[:,:,n] = ez_Yee_new
    hy_Yee_list[:,:,n] = hy_Yee_new

    bx_U_list[:,:,n] = bx_U_new
    ez_U_list[:,:,n] = ez_U_new
    hy_U_list[:,:,n] = hy_U_new

animation_speed = 1

fig, ax = plt.subplots()
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(bx_Yee_list[:,:,int(i*animation_speed)])
   ax.set_title(f'n = {int(i*animation_speed)}')


anim = FuncAnimation(fig, animate)
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(bx_U_list[:,:,int(i*animation_speed)])
   ax.set_title(f'n = {int(i*animation_speed)}')


anim = FuncAnimation(fig, animate)
plt.show()
