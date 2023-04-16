import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
from functions import def_update_matrices, update_implicit_faster, def_jz
from  material_properties import Material, Material_grid
import matplotlib.patches as patches
from inversions import inversion

### This file serves to be able to stack 2 UCHIE regions, both themselves homogeneous in the y-direction, on top of each other in the y-direction.
### Relevant paramters to be changed are:
### Lx, Ly_t, Ly_b, M_t, M_b, iterations, material_list_t, material_list_b (or grid), the source parameters, and the observation points.

Lx = 10 # Length in the x-direction in units m
Ly_t = 10 # Length of the top region in the y-direction in units m
Ly_b = 10 # Length of the bottom region in the y-direction in units m. This UCHIE region can be more refined than the top region.

M_t = 200 # Number of cells in the x-direction
M_b_separate = [1 for i in range(M_t)] # For each top-cell, this denotes the number of UCHIE cells it is subdivided in in the bottom region.
M_b = sum(M_b_separate) # The total number of UCHIE cells in the x-direction of bottom region.

N_t = 200 # Number of cells in the y-direction of top region
N_b = 200 # Number of cells in the y-direction of bottom region
partition = 'uniform' # delta_x and delta_y are then constants. If partition != uniform, delta_x and delta_y should be specified as arrays.

iterations = 250 # Number of iterations. The total time length that is simulated is then equal to iterations * delta_t
simulation_time = None # If set to None, the variable iterations is unaltered. 
# If a value is given, this is the simulation time in units of c, thus making iterations equal to int(simulation_time*c/delta_t)


Si = Material('Silicon')
Cu = Material('Copper')
SiO2 = Material('Silica')
Mat3 = Material(['epsilon_r_3', 3, 1, 0])
Mat3p2 = Material(['epsilon_r_3.2', 3.2, 1, 0])

grid = 'vacuum'

if grid == 'vacuum':
    material_list_b = []
    material_list_t = []
elif grid == 'dielectric':
    left = 100
    right = 140

    material_list_b = [[Mat3, left, right, 'blue']]
    material_list_t = [[Mat3, left, right, 'blue']]
elif grid == 'MIS':
    Si_left = 100
    Si_right = 120
    SiO2_right = 125
    Cu_right = 130

    material_list = [[Si, Si_left, Si_right, 'blue'], 
                     [SiO2, Si_right, SiO2_right, 'yellow'],
                     [Cu, SiO2_right, Cu_right, 'red']
                     ]
    material_list_b = material_list
    material_list_t = material_list
elif grid == 'microstrip':
    Lx = 324*10**(-6)
    Ly_t = 900*10**(-6)
    Ly_b = 150*10**(-6)

    M_t = 162
    N_t = 450
    N_b = 75

    M_b_separate = [1 for i in range(M_t)] # For each top-cell, this denotes the number of UCHIE cells it is subdivided in in the bottom region.
    M_b = sum(M_b_separate) # The total number of UCHIE cells in the x-direction of bottom region.

    material_list_t = [[Mat3p2, 89, 125, 'yellow'],
                       [Cu, 125, 134, 'red']
                        ]

    material_list_b = [[Cu, 80, 89, 'red'],
                       [Mat3p2, 89, 125, 'yellow'],
                       [Cu, 125, 134, 'red']
                        ]

elif grid == 'hole_in_wall':
    Lx = 1.2
    Ly_t = 1.18
    Ly_b = 0.02

    M_t = 600
    N_t = 590
    N_b = 10

    M_b_separate = [1 for i in range(M_t)] # For each top-cell, this denotes the number of UCHIE cells it is subdivided in in the bottom region.
    M_b = sum(M_b_separate) # The total number of UCHIE cells in the x-direction of bottom region.

    material_list_t = [[Cu, 250, 252, 'red']]
    material_list_b = []


### Definitions of physical constants
epsilon_0 = 8.85*10**(-12)  # in units F/m
mu_0 = 1.25663706*10**(-6) # in units N/A^2
c = 3*10**8 # in units m/s

### Definition of the material properties. These properties should not depend on the y-coordinate.
epsilon_t = np.ones((M_t,N_t))*epsilon_0
mu_t = np.ones((M_t,N_t))*mu_0
sigma_t = np.ones((M_t,N_t))*0 # in units kg m^3 s^-3 A^-2 = V m^2 A^-1

epsilon_b = np.ones((M_b,N_b))*epsilon_0
mu_b = np.ones((M_b,N_b))*mu_0
sigma_b = np.ones((M_b,N_b))*0 # in units kg m^3 s^-3 A^-2 = V m^2 A^-1

material_grid_t = Material_grid(material_list_t)
material_grid_b = Material_grid(material_list_b)

[epsilon_t, mu_t, sigma_t] = material_grid_t.set_properties(epsilon_t, mu_t, sigma_t)
[epsilon_b, mu_b, sigma_b] = material_grid_b.set_properties(epsilon_b, mu_b, sigma_b)

# epsilon[60:90,:] = np.ones((30,N))*3*epsilon_0

if partition == 'uniform':
    delta_x_t = np.ones(M_t)*Lx/M_t
    delta_y_t = np.ones(N_t)*Ly_t/N_t
    delta_y_b = np.ones(N_b)*Ly_b/N_b

   # delta_x[Cu_left:Cu_right] = delta_x[Cu_left:Cu_right]/100
   # delta_x[Si_left:Si_right] = delta_x[Si_left:Si_right]/10
else:
    delta_x_t = 0 # specify explicitly
    delta_y_t = 0 # specify explicitly
    delta_y_b = 0 # specify explicitly

#delta_x = [((i+1)**(1/10))*10/M for i in range(M)]

### These matrices contain the delta_x and delta_y values at a given vertex. 
delta_x_matrix_t = np.array([np.repeat(delta_x_t[i], N_t) for i in range(M_t)])
delta_y_matrix_t = np.array([delta_y_t for i in range(M_t)])

# Corresponding delta_x and delta_y matrices for UCHIE part. This is completely determined by the previous settings.
delta_x_b_fractions = np.array([[1/M_b_separate[i] for j in range(M_b_separate[i])] for i in range(M_t)])

delta_x_b_fractions_cumsumlist = np.array([np.cumsum(el) for el in delta_x_b_fractions])
M_b_separate_cumsumlist = np.cumsum(M_b_separate)
M_b_separate_cumsumlist = np.insert(M_b_separate_cumsumlist, 0, 0)

interpolate_matrix = np.zeros((M_b, M_t+1))
for i in range(M_t):
    interpolate_matrix[M_b_separate_cumsumlist[i]:M_b_separate_cumsumlist[i+1],i:i+2] = np.transpose([1-delta_x_b_fractions_cumsumlist[i], delta_x_b_fractions_cumsumlist[i]])

delta_x_b = np.array([delta_x_b_fractions[i,:]*delta_x_t[i] for i in range(M_t)]).flatten()

delta_x_matrix_b = np.array([np.repeat(delta_x_b[i], N_b) for i in range(M_b)])
delta_y_matrix_b = np.array([delta_y_b for i in range(M_b)])

### Definition of the courant number and the corresponding delta_t.
courant_number = 1
delta_t = min(np.min(delta_y_t), np.min(delta_y_b))/(c)*courant_number # in units s

if simulation_time is not None:
    iterations = int(simulation_time*c/delta_t)

print(f'duration is {delta_t*iterations} seconds')

### Definition of the source 
# The source type should be either dirac, gaussian, gaussian_modulated, or gaussian_modulated_dirac
source = 'gaussian_modulated' # type of the source
x_source = 60 # x-coordinate of the source. Make sure this is within bounds.
y_source = 100 # y-coordinate of the source. Make sure this is within bounds.
J0 = 0 # amplitude of the source in units V^2 m A^-1
tc = 5 # tc*delta_t is the time the source peaks
sigma_source = 2.2 # spread of the source in the case of gaussian or gaussian_modulated source
period = 10 # period of the source in number of time steps in the case of gaussian or gaussian_modulated source
omega_c = (2*np.pi)/(period*delta_t) # angular frequency of the source in the case of gaussian or gaussian_modulated source

jz_t = def_jz(J0, source, M_t, N_t, x_source, y_source, iterations, delta_t, tc, sigma_source, period, 1/(delta_x_t[x_source]*delta_y_t[y_source]))


source = 'gaussian_modulated' # type of the source
x_source = 100 # x-coordinate of the source. Make sure this is within bounds.
y_source = 100 # y-coordinate of the source. Make sure this is within bounds.
J0 = 1 # amplitude of the source in units V^2 m A^-1
tc = 5 # tc*delta_t is the time the source peaks
sigma_source = 2.2 # spread of the source in the case of gaussian or gaussian_modulated source
period = 10 # period of the source in number of time steps in the case of gaussian or gaussian_modulated source
omega_c = (2*np.pi)/(period*delta_t) # angular frequency of the source in the case of gaussian or gaussian_modulated source

jz_b = def_jz(J0, source, M_b, N_b, x_source, y_source, iterations, delta_t, tc, sigma_source, period, 1/(delta_x_b[x_source]*delta_y_b[y_source]))

observation_points_ez_t = [(100, 100)]
ez_t_list_observe = np.zeros((iterations, len(observation_points_ez_t)))

observation_points_ez_b = [(100, 100)]
ez_b_list_observe = np.zeros((iterations, len(observation_points_ez_b)))



[A_t, B_t] = def_update_matrices(epsilon_t, mu_t, sigma_t, delta_x_t, delta_y_t, delta_t, M_t)

[A_b, B_b] = def_update_matrices(epsilon_b, mu_b, sigma_b, delta_x_b, delta_y_b, delta_t, M_b)

inversion_method = 'numpy_nonsparse'

A_t_inv = inversion(A_t, M_t, inversion_method)
A_b_inv = inversion(A_b, M_b, inversion_method)

A_invB_t = np.dot(A_t_inv, B_t)
A_invB_b = np.dot(A_b_inv, B_b)


### Definition of some matrices that are useful later on.
eps_sigma_plus_t = epsilon_t/delta_t + sigma_t/2
eps_sigma_min_t = epsilon_t/delta_t - sigma_t/2

eps_sigma_plus_b = epsilon_b/delta_t + sigma_b/2
eps_sigma_min_b = epsilon_b/delta_t - sigma_b/2

bx_t_list = np.zeros((M_t, N_t, iterations))
ez_t_list = np.zeros((M_t, N_t, iterations))
hy_t_list = np.zeros((M_t, N_t, iterations))

bx_b_list = np.zeros((M_b, N_b, iterations))
ez_b_list = np.zeros((M_b, N_b, iterations))
hy_b_list = np.zeros((M_b, N_b, iterations))

ez_t_new = np.zeros((M_t, N_t))
hy_t_new = np.zeros((M_t, N_t))
bx_t_new = np.zeros((M_t, N_t))

ez_b_new = np.zeros((M_b, N_b))
hy_b_new = np.zeros((M_b, N_b))
bx_b_new = np.zeros((M_b, N_b))

for n in range(iterations):
    print(f'iteration {n+1}/{iterations} started')

    ez_t_old = ez_t_new
    hy_t_old = hy_t_new
    bx_t_old = bx_t_new

    ez_b_old = ez_b_new
    hy_b_old = hy_b_new
    bx_b_old = bx_b_new

    # Ez and Hy implicitly updated in the UCHIE region
    [ez_t_new, hy_t_new] = update_implicit_faster(ez_t_old, hy_t_old, bx_t_old, n, A_t_inv, A_invB_t, delta_t, delta_y_matrix_t, M_t, N_t, jz_t, mu_t)
    [ez_b_new, hy_b_new] = update_implicit_faster(ez_b_old, hy_b_old, bx_b_old, n, A_b_inv, A_invB_b, delta_t, delta_y_matrix_b, M_b, N_b, jz_b, mu_b)

    ez_t_new[:,0] = np.zeros(M_t)
    hy_t_new[:,0] = np.zeros(M_t)
    
    ez_b_new[:,0] = np.zeros(M_t)
    hy_b_new[:,0] = np.zeros(M_t)
    
    # Bx explicitly updated in the top region (interpolation needed)
    bx_t_new = np.zeros((M_t, N_t))
    bx_t_new[:,1:-1] = bx_t_old[:,1:-1] - (ez_t_new[:,2:] - ez_t_new[:,1:-1])

    bx_b_new = np.zeros((M_b, N_b))
    bx_b_new[:,1:-1] = bx_b_old[:,1:-1] - (ez_b_new[:,2:] - ez_b_new[:,1:-1])

    # top and bottom here refer to the top and bottom of the bottom region.
    ez_top = list(ez_t_new[:,1])
    ez_top.append(ez_t_new[0,1])
    ez_top = np.array(ez_top)
    ez_bottom = list(ez_t_new[:,-1])
    ez_bottom.append(ez_t_new[0,-1])
    ez_bottom = np.array(ez_bottom)

    # top
    bx_b_new[:,-1] = bx_b_old[:,-1] - (np.dot(interpolate_matrix, ez_top)*delta_t - ez_b_new[:,-1]) # add periodic boundary condition
    # ez_U[:,0] does not exist, or equivalently, is always zero.
    # bottom
    bx_b_new[:,0] = bx_b_old[:,0] - (ez_b_new[:,1] - np.dot(interpolate_matrix, ez_bottom)*delta_t) # add periodic boundary condition

    for i in range(M_t):
        bx_t_new[i, 0] = bx_b_new[M_b_separate_cumsumlist[i],-1]
        bx_t_new[i, -1] = bx_b_new[M_b_separate_cumsumlist[i],0]

    bx_t_list[:,:,n] = bx_t_new
    ez_t_list[:,:,n] = ez_t_new
    hy_t_list[:,:,n] = hy_t_new

    bx_b_list[:,:,n] = bx_b_new
    ez_b_list[:,:,n] = ez_b_new
    hy_b_list[:,:,n] = hy_b_new

    for i, point in enumerate(observation_points_ez_t):
        ez_t_list_observe[n, i] = ez_t_new[point]

    for i, point in enumerate(observation_points_ez_b):
        ez_b_list_observe[n, i] = ez_b_new[point]

for i, point in enumerate(observation_points_ez_b):
    plt.plot(range(iterations), ez_b_list_observe[:,i])
    plt.xlabel('Iteration')
    plt.ylabel('Ez')
    plt.title(f'Ez at {point} in bottom UCHIE region')
    plt.show()

for i, point in enumerate(observation_points_ez_t):
    plt.plot(range(iterations), ez_t_list_observe[:,i])
    plt.xlabel('Iteration')
    plt.ylabel('Ez')
    plt.title(f'Ez at {point} in top UCHIE region')
    plt.show()


animation_speed = 10

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')
for mat in material_list_b:
    rect = patches.Rectangle((mat[1], 0), mat[2]-mat[1], N_b-1, edgecolor = mat[3], linewidth=1, facecolor="none", label = mat[0].name)
    ax.add_patch(rect)

def animate(i):
    ax.pcolormesh(np.transpose(ez_b_list[:,:,int(i*animation_speed)]))
    ax.set_title(f'bottom UCHIE region. n = {int(i*animation_speed)}')
    for mat in material_list_b:
        rect = patches.Rectangle((mat[1], 0), mat[2]-mat[1]-1, N_b-1, edgecolor = mat[3], linewidth=1, facecolor="none", label = mat[0].name)
        ax.add_patch(rect)


anim = FuncAnimation(fig, animate)
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')
for mat in material_list_t:
    rect = patches.Rectangle((mat[1], 0), mat[2]-mat[1], N_t-1, edgecolor = mat[3], linewidth=1, facecolor="none", label = mat[0].name)
    ax.add_patch(rect)

def animate(i):
    ax.pcolormesh(np.transpose(ez_t_list[:,:,int(i*animation_speed)]))
    ax.set_title(f'top UCHIE region. n = {int(i*animation_speed)}')
    for mat in material_list_t:
        rect = patches.Rectangle((mat[1], 0), mat[2]-mat[1]-1, N_t-1, edgecolor = mat[3], linewidth=1, facecolor="none", label = mat[0].name)
        ax.add_patch(rect)

anim = FuncAnimation(fig, animate)
plt.legend()
plt.show()