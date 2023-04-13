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

M_Yee = 100
N_Yee = 100
iterations = 60

# Define UCHIE region
# Define the 4 edges of the region
U_x_left = 40
U_x_right = 60
U_y_top = 60
U_y_bottom = 40
# Define how many layers in the x-direction the UCHIE region has (and y-direction, specified by the above)
U_Yee = U_x_right - U_x_left
M_U_separate = [5 for i in range(U_Yee)]
M_U = sum(M_U_separate)
N_U = U_y_top - U_y_bottom

# Define the location of the source

source_X_U = M_U//3
source_Y_U = N_U//3
source_X_Yee = 50
source_Y_Yee = 10

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
A_inv = linalg.inv(A)

print(linalg.norm(A_inv))

delta_x_Yee_left = delta_x_Yee[U_x_left]
delta_x_Yee_right = delta_x_Yee[U_x_right]
[A_new, B_new] = def_update_matrices_hybrid_new(epsilon_U, mu_U, sigma_U, delta_x_U, delta_y_U, delta_t, M_U, delta_x_Yee_left, delta_x_Yee_right)


A_changed = copy.deepcopy(A)
A_changed[M_U-1, 0] = 0
A_changed[2*M_U-1, 0] = 0
A_changed[M_U-1, M_U] = 0
A_changed[2*M_U-1, M_U] = 0
A_changed_inv = linalg.inv(A_changed)


"""
A[0, M_U] = 0
A[M_U-1, 2*M_U-1] = 0
A[M_U, 0] = 0
A[2*M_U-1, M_U-1] = 0
"""
"""
A[0,:] = np.zeros(2*M_U)
A[M_U-1,:] = np.zeros(2*M_U)
A[M_U,:] = np.zeros(2*M_U)
A[2*M_U-1,:] = np.zeros(2*M_U)
A[:,0] = np.zeros(2*M_U)
A[:,M_U-1] = np.zeros(2*M_U)
A[:,M_U] = np.zeros(2*M_U)
A[:,2*M_U-1] = np.zeros(2*M_U)

A[0,0] = 1
A[M_U-1,M_U-1] = 1
A[M_U,M_U] = 1
A[2*M_U-1,2*M_U-1] = 1

"""
print('ok')


A_new[0,1] = -5
A_new[M_U,1] = 0.04
A_new[0,M_U+1] = -6000
A_new[M_U,M_U+1] = -10

A_new[0,0] = -0.5
A_new[0, 1] = 0.5
A_new[0, 5] = -2961.92
A_new[0, 6] = -2961.92
A_new[5, 0] = 0.002085
A_new[5, 1] = 0.002085
A_new[5, 5] = 5
A_new[5, 6] = -5

A_new[4, 3] = 5
A_new[4, 8] = A_new[4, 9]
A_new[9, 3] = -A_new[9, 4]
A_new[9, 8] = -10


A_new_inv = linalg.inv(A_new)

#A[M_U-1, M_U-2] = 10

# A[0,M_U+1] = -6000
# A[M_U-1,2*M_U-2] = 6000

# A[2*M_U-1,M_U-2] = 0.04

#A[M_U,M_U+1] = 10
#A[2*M_U-1,2*M_U-2] = -10

"""
A[0,0:2] = -A[0,0:2]
A[M_U-1,M_U-2] = -A[M_U-1,M_U-2]

A[M_U-1,2*M_U-2:2*M_U] = -A[M_U-1,2*M_U-2:2*M_U] 
A[M_U,M_U:M_U+2] = -A[M_U,M_U:M_U+2]
A[2*M_U-1,2*M_U-2:2*M_U] = -A[2*M_U-1,2*M_U-2:2*M_U]
"""


A_inv = linalg.inv(A)


print(linalg.norm(A_inv))
print(linalg.norm(A_changed_inv))
print(linalg.norm(A_new_inv))

print('check')
print(linalg.norm(A))
print(linalg.norm(A_inv))
print(np.min(A))
print(np.max(A))

"""
A[[1, M_U-1]] = A[[M_U-1, 1]]
A[:,[1, M_U-1]] = A[:,[M_U-1, 1]]

A[[2, M_U]] = A[[M_U, 2]]
A[:,[2, M_U]] = A[:,[M_U, 2]]

A[[3, 2*M_U-1]] = A[[2*M_U-1, 3]]
A[:,[3, 2*M_U-1]] = A[:,[2*M_U-1, 3]]
"""


[u, s, v] = linalg.svd(A)

print('svd')

print(s)

print(u[:,-1])
print(v[-1,:])

print('done')

print(linalg.norm(A_inv))


print(A_inv)
print(np.max(A_inv))
print(np.min(A))

print(linalg.det(A))

[w, v] = linalg.eig(A)

print(np.sort([[abs(i) for i in w]]))

A_lb = np.zeros((2, 2))
B_lb = np.zeros((2, 2))
A_rb = np.zeros((2, 2))
B_rb = np.zeros((2, 2))
#  C_term = np.zeros((2, N_U))

A_lb[0, 0] = epsilon_U[0,0]/delta_t + sigma_U[0,0]/2
A_lb[0, 1] = -1/delta_x_Yee_left
A_lb[1, 0] = 1/(2*delta_x_Yee_left)
A_lb[1, 1] = -mu_U[0,0]/delta_t

B_lb[0, 0] = epsilon_U[0,0]/delta_t - sigma_U[0,0]/2
B_lb[0, 0] = 1/delta_x_Yee_left
B_lb[1, 0] = -1/(2*delta_x_Yee_left)
B_lb[1, 1] = -mu_U[0,0]/delta_t

A_rb[0, 0] = epsilon_U[-1,0]/delta_t + sigma_U[-1,0]/2
A_rb[0, 1] = 1/delta_x_Yee_right
A_rb[1, 0] = 1/(2*delta_x_Yee_right)
A_rb[1, 1] = mu_U[-1,0]/delta_t

B_rb[0, 0] = epsilon_U[-1,0]/delta_t - sigma_U[-1,0]/2
B_rb[0, 0] = -1/delta_x_Yee_right
B_rb[1, 0] = -1/(2*delta_x_Yee_right)
B_rb[1, 1] = mu_U[-1,0]/delta_t

A_lb_inv = linalg.inv(A_lb)
A_rb_inv = linalg.inv(A_rb)

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
    # Change the edges of the Yee-region: equate the bx-values around the UCHIE box and the ez-values on the left and right side to the just updated UCHIE values.
    ez_Yee_new[U_x_left+1:U_x_right-1,U_y_bottom+1:U_y_top-1] = np.zeros((U_Yee-2, N_U-2))
    bx_Yee_old[U_x_left+1:U_x_right-1,U_y_bottom+1:U_y_top-1] = np.zeros((U_Yee-2, N_U-2))
    hy_Yee_old[U_x_left+1:U_x_right-1,U_y_bottom+1:U_y_top-1] = np.zeros((U_Yee-2, N_U-2))
    """

    Ez_left_new = ez_Yee_new[U_x_left-1,U_y_bottom:U_y_top]*delta_t
    Ez_left_old = ez_Yee_old[U_x_left-1,U_y_bottom:U_y_top]*delta_t
    Ez_right_new = ez_Yee_new[U_x_right+1,U_y_bottom:U_y_top]*delta_t
    Ez_right_old = ez_Yee_old[U_x_right+1,U_y_bottom:U_y_top]*delta_t
    Hy_left = hy_Yee_old[U_x_left-1,U_y_bottom:U_y_top]*delta_t*delta_y_matrix_Yee[U_x_left-1,0]
    Hy_right = hy_Yee_old[U_x_right,U_y_bottom:U_y_top]*delta_t*delta_y_matrix_Yee[U_x_right,0]

    # Ez and Hy implicitly updated in the UCHIE region
#    [ez_U_new, hy_U_new] = update_implicit_hybrid_new(ez_U_old, hy_U_old, bx_U_old, n, A_inv, B, delta_t, delta_y_matrix_U, M_U, N_U, jz_U, mu_U, delta_x_Yee_left, delta_x_Yee_right, Ez_left_new, Ez_left_old, Ez_right_new, Ez_right_old, Hy_left, Hy_right)
    [ez_U_new, hy_U_new] = update_implicit(ez_U_old, hy_U_old, bx_U_old, n, A_inv, B, delta_t, delta_y_matrix_U, M_U, N_U, jz_U, mu_U)

    # left and right interface

    C_term_left = np.zeros((2, N_U))
    C_term_right = np.zeros((2, N_U))

    if n == 40:
        print('start')
        bx_term_left = np.divide(bx_U_new[0,:], np.multiply(mu_U[0,:], delta_y_matrix_U[0,:]))
        bx_term_right = np.divide(bx_U_new[-1,:], np.multiply(mu_U[-1,:], delta_y_matrix_U[-1,:]))

        print((Ez_left_new + Ez_left_old)/(2*delta_x_Yee_left))
        print((Ez_right_new + Ez_right_old)/(2*delta_x_Yee_left))
        print('1')
        print(-bx_term_left + np.roll(bx_term_left, 1))
        print(jz_U[0,:,n])
        print('2')
        print(2*Hy_left/delta_x_Yee_left)
        print(-bx_term_right + np.roll(bx_term_right, 1))
        print('3')
        print(jz_U[0,:,n])
        print(2*Hy_right/delta_x_Yee_right)
        print('done')
    bx_term_left = np.divide(bx_U_new[0,:], np.multiply(mu_U[0,:], delta_y_matrix_U[0,:]))
    bx_term_right = np.divide(bx_U_new[-1,:], np.multiply(mu_U[-1,:], delta_y_matrix_U[-1,:]))
    
    C_term_left[0,:] = (Ez_left_new + Ez_left_old)
    C_term_right[0,:] = (Ez_right_new + Ez_right_old)

    bx_term_left = np.divide(bx_U_new[0,:], np.multiply(mu_U[0,:], delta_y_matrix_U[0,:]))
    C_term_left[1,:] = -bx_term_left + np.roll(bx_term_left, 1) - jz_U[0,:,n] - 2*Hy_left/delta_x_Yee_left
    bx_term_right = np.divide(bx_U_new[-1,:], np.multiply(mu_U[-1,:], delta_y_matrix_U[-1,:]))
    C_term_right[1,:] = -bx_term_right + np.roll(bx_term_right, 1) - jz_U[0,:,n] + 2*Hy_right/delta_x_Yee_right

    values_left_old = np.array([ez_U_old[0,:], hy_U_old[0,:]])
    values_left_new = np.dot(A_lb_inv, np.dot(B_lb, values_left_old) + C_term_left)
    ez_U_new[0,:] = values_left_new[0,:]
    hy_U_new[0,:] = values_left_new[1,:]

    values_right_old = np.array([ez_U_old[-1,:], hy_U_old[-1,:]])
    values_right_new = np.dot(A_rb_inv, np.dot(B_rb, values_right_old) + C_term_right)
    ez_U_new[-1,:] = values_right_new[0,:]
    hy_U_new[-1,:] = values_right_new[1,:]

    """
    ez_U_new[:,0] = np.zeros(M_U)
    ez_U_new[:,-1] = np.zeros(M_U)
    hy_U_new[:,0] = np.zeros(M_U)
    hy_U_new[:,-1] = np.zeros(M_U)
    """
    if n == 40:
        print(ez_U_new)
        print(ez_Yee_new)
        print('ok')

#    hy_U_new[0,:] = hy_U_old[0,:] - delta_t*np.divide(0, np.multiply(mu_U, delta_x_U))

    """
    Hy_Yee_left = hy_Yee_new[U_x_left,U_y_bottom:U_y_top]
    Hy_Yee_right = hy_Yee_new[U_x_right,U_y_bottom:U_y_top]
    Ez_Yee_left = ez_Yee_new[U_x_left,U_y_bottom:U_y_top]
    Ez_Yee_right = ez_Yee_new[U_x_right,U_y_bottom:U_y_top]
    """

   # [ez_U_new, hy_U_new] = update_implicit_hybrid_zeros(ez_U_old, hy_U_old, bx_U_old, n, A_inv, B, delta_t, delta_y_matrix_U, M_U, N_U, jz_U, mu_U, delta_x_Yee_left, delta_x_Yee_right, 0, 0, Ez_Yee_left, Ez_Yee_right, Hy_Yee_left, Hy_Yee_right)

    # Bx explicitly updated in the UCHIE region (interpolation needed)


    bx_U_new = np.zeros((M_U, N_U))
    bx_U_new[:,1:-1] = bx_U_old[:,1:-1] - (ez_U_new[:,2:] - ez_U_new[:,1:-1])
    bx_U_new[:,-1] = bx_U_old[:,-1] - (np.dot(interpolate_matrix, ez_Yee_new[U_x_left:U_x_right+1,U_y_top+1])*delta_t - ez_U_new[:,-1]) # add periodic boundary condition
    # ez_U[:,0] does not exist, or equivalently, is always zero.
    bx_U_new[:,0] = bx_U_old[:,0] - (ez_U_new[:,1] - np.dot(interpolate_matrix, ez_Yee_new[U_x_left:U_x_right+1,U_y_bottom])*delta_t) # add periodic boundary condition


    """ #Should be executed, but does not work yet.
    # Change the edges of the Yee-region: equate the bx-values around the UCHIE box and the ez-values on the left and right side to the just updated UCHIE values.

    bx_Yee_old[U_x_left,U_y_bottom:U_y_top] = bx_U_new[0,:]
    bx_Yee_old[U_x_right,U_y_bottom:U_y_top] = bx_U_new[-1,:]
    for i in range(U_Yee):
        bx_Yee_old[U_x_left+i, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[i],0]
        bx_Yee_old[U_x_left+i, U_y_top] = bx_U_new[M_U_separate_cumsumlist[i],-1]
    bx_Yee_old[U_x_right, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[-1]-1,0]
    bx_Yee_old[U_x_right, U_y_top] = bx_U_new[M_U_separate_cumsumlist[-1]-1,-1]
    """

    ez_Yee_new[U_x_left,U_y_bottom+1:U_y_top] = ez_U_new[0,1:]
    ez_Yee_new[U_x_right,U_y_bottom+1:U_y_top] = ez_U_new[-1,1:]
    

    # Bx is explicitly updated in the Yee region
    eq_3_term = np.divide(ez_Yee_new*delta_t, delta_y_matrix_Yee)
    bx_Yee_new = bx_Yee_old - np.roll(eq_3_term, -1, 1) + eq_3_term

    # Hy is explicitly updated in the Yee region
    eq_2_term = np.multiply(eq2_matrix, ez_Yee_new)
    hy_Yee_new = hy_Yee_old + np.roll(eq_2_term, -1, 0) - eq_2_term


    if n == 10:
        print('check differencesss')
        for i in range(U_Yee):
            print(i)
            print(ez_Yee_new[U_x_left+i, U_y_bottom])
            print(ez_U_new[M_U_separate_cumsumlist[i],0])
            print(ez_Yee_new[U_x_left+i, U_y_top])
            print(ez_U_new[M_U_separate_cumsumlist[i],-1])
        print('jfdkmlqsjfklm')
    """
    print('bxs are ')
    print(np.max(bx_Yee_new))
    print(np.max(bx_U_new))
    for i in range(U_Yee):
        bx_Yee_new[U_x_left+i, U_y_bottom] = bx_U_new[M_U_separate_cumsumlist[i],0]
        bx_Yee_new[U_x_left+i, U_y_top] = bx_U_new[M_U_separate_cumsumlist[i],-1]
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
