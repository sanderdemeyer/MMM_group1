import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from time import perf_counter
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
import copy
from functions import def_jz, def_update_matrices, def_update_matrices_hybrid, update_implicit, update_implicit_hybrid, def_update_matrices_hybrid_new, update_implicit_hybrid_new, update_implicit_hybrid_zeros
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as ssalg

Lx = 1 # Length in the x-direction in units m
Ly = 1 # Length in the x-direction in units m

M_Yee = 400 # Number of cells in the x-direction
N_Yee = 400 # Number of cells in the y-direction
partition = 'uniform' # delta_x_Yee and delta_y_Yee are then constants. If partition != uniform, these should be specified as arrays.

M_U_separate = [5 for i in range(M_Yee)] # For each Yee-cell, this denotes the number of UCHIE cells it is subdivided in.
M_U = sum(M_U_separate) # The total number of UCHIE cells in the x-direction
N_U = N_Yee # The total number of UCHIE cells in the y-direction. This 

iterations = 100 # Number of iterations. The total time length that is simulated is then equal to iterations * delta_t


### Definitions of physical constants
epsilon_0 = 8.85*10**(-12)  # in units F/m
mu_0 = 1.25663706*10**(-6) # in units N/A^2
c = 3*10**8 # in units m/s

### Definition of the material properties. These properties should not depend on the y-coordinate.
epsilon_Yee = np.ones((M_Yee,N_Yee))*epsilon_0
mM_Yee = np.ones((M_Yee,N_Yee))*mu_0
sigma_Yee = np.ones((M_Yee,N_Yee))*0 # in units kg m^3 s^-3 A^-2 = V m^2 A^-1

epsilon_U = np.ones((M_U,N_U))*epsilon_0
mu_U = np.ones((M_U,N_U))*mu_0
sigma_U = np.ones((M_U,N_U))*0 # in units kg m^3 s^-3 A^-2 = V m^2 A^-1

if partition == 'uniform':
    delta_x_Yee = np.ones(M_Yee)*Lx/M_Yee
    delta_y_Yee = np.ones(N_Yee)*Ly/N_Yee
else:
    delta_x_Yee = 0 # specify explicitly
    delta_y_Yee = 0 # specify explicitly

### These matrices contain the delta_x and delta_y values at a given vertex. 
delta_x_matrix_Yee = np.array([np.repeat(delta_x_Yee[i], N_Yee) for i in range(M_Yee)])
delta_y_matrix_Yee = np.array([delta_y_Yee for i in range(M_Yee)])

# Corresponding delta_x and delta_y matrices for UCHIE part. This is completely determined by the previous settings.
delta_x_U_fractions = np.array([[1/M_U_separate[i] for j in range(M_U_separate[i])] for i in range(M_Yee)])
"""
for i in range(N_Yee):
    assert abs(sum(delta_x_U_fractions[i,:])-1) < 10**(-7), 'sum of all individual fractions should be 1 for each Yee-cell'
"""
delta_x_U_fractions_cumsumlist = np.array([np.cumsum(el) for el in delta_x_U_fractions])
M_U_separate_cumsumlist = np.cumsum(M_U_separate)
M_U_separate_cumsumlist = np.insert(M_U_separate_cumsumlist, 0, 0)

interpolate_matrix = np.zeros((M_U, M_Yee+1))
for i in range(M_Yee):
    interpolate_matrix[M_U_separate_cumsumlist[i]:M_U_separate_cumsumlist[i+1],i:i+2] = np.transpose([1-delta_x_U_fractions_cumsumlist[i], delta_x_U_fractions_cumsumlist[i]])

delta_x_U = np.array([delta_x_U_fractions[i,:]*delta_x_Yee[i] for i in range(M_Yee)]).flatten()
delta_y_U = delta_y_Yee

delta_x_matrix_U = np.array([np.repeat(delta_x_U[i], N_U) for i in range(M_U)])
delta_y_matrix_U = np.array([delta_y_U for i in range(M_U)])

### Definition of the courant number and the corresponding delta_t.
# This is based on the minimal time step that needs to be taken to satisfy the courant limit of both the UCHIE and Yee part.
# The smallest time step of both is chosen. This delta_t is used in both the UCHIE and Yee part.
courant_number = 1
delta_t = (courant_number/c)*min(np.max(delta_y_U), 1/(np.sqrt(1/(np.max(delta_x_Yee))**2 + 1/(np.max(delta_y_Yee))**2)))

### Definition of the sources. It is recommended to only set 1 of the 2 sources to non-zero.
# The source type should be either dirac, gaussian, or gaussian_modulated

# Yee source
source_Yee = 'gaussian_modulated' # type of the source
source_X_Yee = 50 # x-coordinate of the source. Make sure this is within bounds.
source_Y_Yee = 90 # y-coordinate of the source. Make sure this is within bounds.
J0_Yee = 1 # amplitude of the source in units V^2 m A^-1
tc_Yee = 5 # tc*delta_t is the time the source peaks
sigma_source_Yee = 1 # spread of the source in the case of gaussian or gaussian_modulated source
period_Yee = 10 # period of the source in number of time steps in the case of gaussian or gaussian_modulated source
omega_c_Yee = (2*np.pi)/(period_Yee*delta_t) # angular frequency of the source in the case of gaussian or gaussian_modulated source

# UCHIE source
source_U = 'dirac' # type of the source
source_X_U = 50 # x-coordinate of the source. Make sure this is within bounds.
source_Y_U = 50 # y-coordinate of the source. Make sure this is within bounds.
J0_U = 0 # amplitude of the source in units V^2 m A^-1
tc_U = 5 # tc*delta_t is the time the source peaks
sigma_source_U = 1 # spread of the source in the case of gaussian or gaussian_modulated source
period_U = 10 # period of the source in number of time steps in the case of gaussian or gaussian_modulated source
omega_c_U = (2*np.pi)/(period_U*delta_t) # angular frequency of the source in the case of gaussian or gaussian_modulated source

assert len(M_U_separate) == M_Yee, 'Length of M_U should be the same number as the number of Yee cells incompassed in the UCHIE region'

### Definition of some matrices that are useful later on.
eps_sigma_plus_U = epsilon_U/delta_t + sigma_U/2
eps_sigma_min_U = epsilon_U/delta_t - sigma_U/2

eps_sigma_plus_Yee = epsilon_Yee/delta_t + sigma_Yee/2
eps_sigma_min_Yee = epsilon_Yee/delta_t - sigma_Yee/2

eq2_matrix = np.divide(delta_t, np.multiply(mM_Yee, delta_x_matrix_Yee))

### Initialization of all the fields.
ez_Yee_new = np.zeros((M_Yee, N_Yee))
hy_Yee_new = np.zeros((M_Yee, N_Yee))
bx_Yee_new = np.zeros((M_Yee, N_Yee))

ez_U_new = np.zeros((M_U, N_U))
hy_U_new = np.zeros((M_U, N_U))
bx_U_new = np.zeros((M_U, N_U))

### Value of the sources based on the definitions above.
jz_U = def_jz(J0_U, source_U, M_U, N_U, source_X_U, source_Y_U, iterations, 1/(delta_x_U[0]*delta_y_U[0]), tc_U, sigma_source_U, period_U)
jz_Yee = def_jz(J0_Yee, source_Yee, M_Yee, N_Yee, source_X_Yee, source_Y_Yee, iterations, 1/(delta_x_Yee[0]*delta_y_Yee[0]), tc_Yee, sigma_source_Yee, period_Yee)

### Definition of the UCHIE implicit update matrices.
[A, B] = def_update_matrices(epsilon_U, mu_U, sigma_U, delta_x_U, delta_y_U, delta_t, M_U)
# Determining the type of inversion that is used.
# Options are numpy_nonsparse, numpy_sparse, schur
inversion_method = 'numpy_schur'

t1 = perf_counter()
# Taking the inverse
if inversion_method == 'numpy_nonsparse':
    A_inv = linalg.inv(A)
elif inversion_method == 'numpy_sparse':
    A_csc = csc_matrix(A)
    A_inv_csc = ssalg.inv(A_csc)
    A_inv = A_inv_csc.toarray()
elif inversion_method == 'numpy_schur':
    M11 = A[:M_U+1,:M_U+1]
    M12 = A[:M_U+1,M_U+1:]
    M21 = A[M_U+1:,:M_U+1]
    M22 = A[M_U+1:,M_U+1:]
    M22_inv = linalg.inv(M22)
    S = M11 - np.dot(M12,np.dot(M22_inv,M21))
    S_inv = linalg.inv(S)
    A_inv = np.zeros((2*M_U,2*M_U))
    Deel1 = S_inv
    Deel2 =  -np.dot(S_inv,np.dot(M12,M22_inv))
    Deel3 = -np.dot(M22_inv,np.dot(M21,S_inv))
    Deel4 =  M22_inv + np.dot(M22_inv,np.dot(M21,np.dot(S_inv,np.dot(M12,M22_inv))))
    A_inv[:M_U+1,:M_U+1] = Deel1
    A_inv[:M_U+1,M_U+1:] = Deel2
    A_inv[M_U+1:,:M_U+1] = Deel3
    A_inv[M_U+1:,M_U+1:] = Deel4
elif inversion_method == 'numpy_sparse_schur':
    M11 = csc_matrix(A[:M_U+1,:M_U+1])
    M12 = csc_matrix(A[:M_U+1,M_U+1:])
    M21 = csc_matrix(A[M_U+1:,:M_U+1])
    M22 = csc_matrix(A[M_U+1:,M_U+1:])
    M22_inv = ssalg.inv(M22)
    S = M11 - M12*M22_inv*M21
    S_inv = ssalg.inv(S)
    M_inv = np.zeros((2*M_U,2*M_U))
    Deel1 = S_inv
    Deel2 =  -S_inv*M12*M22_inv
    Deel3 = -M22_inv*M21*S_inv
    Deel4 =  M22_inv + M22_inv*M21*S_inv*M12*M22_inv
    M_inv[:M_U+1,:M_U+1] = Deel1.toarray()
    M_inv[:M_U+1,M_U+1:] = Deel2.toarray()
    M_inv[M_U+1:,:M_U+1] = Deel3.toarray()
    M_inv[M_U+1:,M_U+1:] = Deel4.toarray()
    A_inv = M_inv
else:
    print('Invalid inversion method')
t2 = perf_counter()
print(f'Inversion took {t2-t1} seconds.')

# initialization of the list of fields
bx_Yee_list = np.zeros((M_Yee, N_Yee, iterations))
ez_Yee_list = np.zeros((M_Yee, N_Yee, iterations))
hy_Yee_list = np.zeros((M_Yee, N_Yee, iterations))

bx_U_list = np.zeros((M_U, N_U, iterations))
ez_U_list = np.zeros((M_U, N_U, iterations))
hy_U_list = np.zeros((M_U, N_U, iterations))

### Starting the run

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
