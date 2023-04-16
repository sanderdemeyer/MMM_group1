import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
from functions import def_jz
from  material_properties import Material, Material_grid, Material_grid_y_dependent
import matplotlib.patches as patches

### File that implements the Yee-code. Relevant paramters to be changed are:
### Lx, Ly_Yee, M_Yee, N_Yee, iterations, material_list_Yee (or grid), the source parameters, and the observation points.


Lx = 5 # Length in the x-direction in units m
Ly_Yee = 5 # Length in the x-direction in units m

M_Yee = 200 # Number of cells in the x-direction of the Yee region
N_Yee = 200 # Number of cells in the y-direction of the Yee region
partition = 'uniform' # delta_x_Yee and delta_y_Yee are then constants. If partition != uniform, these should be specified as arrays.

iterations = 250 # Number of iterations. The total time length that is simulated is then equal to iterations * delta_t



Si = Material('Silicon')
Cu = Material('Copper')
SiO2 = Material('Silica')
Mat3 = Material(['epsilon_r_3', 3, 1, 0])
Mat3p2 = Material(['epsilon_r_3.2', 3.2, 1, 0])

grid = 'vacuum'

if grid == 'vacuum':
    y_dependent = False
    material_list_Yee = []
elif grid == 'block':
    y_dependent = False

    left = 100
    right = 105

    material_list_Yee = [[Cu, left, right, 'blue']]
elif grid == 'MIS':
    y_dependent = False

    Si_left = 100
    Si_right = 120
    SiO2_right = 125
    Cu_right = 130

    material_list_Yee = [[Si, Si_left, Si_right, 'blue'], 
                     [SiO2, Si_right, SiO2_right, 'yellow'],
                     [Cu, SiO2_right, Cu_right, 'red']
                     ]
elif grid == 'microstrip':
    # Microstrip based on figure 2.28 of the syllabus
    y_dependent = True

    Lx = 324*10**(-6)
    Ly_Yee = 900*10**(-6)

    M_Yee = 162
    N_Yee = 525

    material_list_Yee = [[Cu, 80, 89, 225, 300, 'red'],
                    [Mat3p2, 89, 125, 0, N_Yee-1, 'yellow'],
                    [Cu, 125, 134, 0, N_Yee-1, 'red']
                     ]
elif grid == 'hole_in_wall':
    Lx = 1.2
    Ly_Yee = 1.2
    M_Yee = 600
    N_Yee = 600

    y_dependent = True
    material_list_Yee = [[Cu, 250, 252, 0, 295, 'red'],
                         [Cu, 250, 252, 305, 599, 'red']
                         ]

### Definitions of physical constants
epsilon_0 = 8.85*10**(-12)  # in units F/m
mu_0 = 1.25663706*10**(-6) # in units N/A^2
c = 3*10**8 # in units m/s

### Definition of the material properties. These properties should not depend on the y-coordinate.
epsilon_Yee = np.ones((M_Yee,N_Yee))*epsilon_0
mu_Yee = np.ones((M_Yee,N_Yee))*mu_0
sigma_Yee = np.ones((M_Yee,N_Yee))*0 # in units kg m^3 s^-3 A^-2 = V m^2 A^-1

if y_dependent:
    material_grid_Yee = Material_grid_y_dependent(material_list_Yee)
else:
    material_grid_Yee = Material_grid(material_list_Yee)

[epsilon_Yee, mu_Yee, sigma_Yee] = material_grid_Yee.set_properties(epsilon_Yee, mu_Yee, sigma_Yee)

if partition == 'uniform':
    delta_x_Yee = np.ones(M_Yee)*Lx/M_Yee
    delta_y_Yee = np.ones(N_Yee)*Ly_Yee/N_Yee
else:
    delta_x_Yee = 0 # specify explicitly
    delta_y_Yee = 0 # specify explicitly

### These matrices contain the delta_x and delta_y values at a given vertex. 
delta_x_matrix_Yee = np.array([np.repeat(delta_x_Yee[i], N_Yee) for i in range(M_Yee)])
delta_y_matrix_Yee = np.array([delta_y_Yee for i in range(M_Yee)])


### Definition of the courant number and the corresponding delta_t.
# This is based on the minimal time step that needs to be taken to satisfy the courant limit of both the UCHIE and Yee part.
# The smallest time step of both is chosen. This delta_t is used in both the UCHIE and Yee part.
courant_number = 1
delta_t = (courant_number/c)*(1/(np.sqrt(1/(np.max(delta_x_Yee))**2 + 1/(np.max(delta_y_Yee))**2)))

### Definition of the sources. It is recommended to only set 1 of the 2 sources to non-zero.
# The source type should be either dirac, gaussian, or gaussian_modulated

# Yee source
source_Yee = 'gaussian_modulated' # type of the source
source_X_Yee = 100 # x-coordinate of the source. Make sure this is within bounds.
source_Y_Yee = 100 # y-coordinate of the source. Make sure this is within bounds.
J0_Yee = 1 # amplitude of the source in units V^2 m A^-1
tc_Yee = 5 # tc*delta_t is the time the source peaks
sigma_source_Yee = 2.2 # spread of the source in the case of gaussian or gaussian_modulated source
period_Yee = 10 # period of the source in number of time steps in the case of gaussian or gaussian_modulated source
omega_c_Yee = (2*np.pi)/(period_Yee*delta_t) # angular frequency of the source in the case of gaussian or gaussian_modulated source



### Definition of some matrices that are useful later on.

eps_sigma_plus_Yee = epsilon_Yee/delta_t + sigma_Yee/2
eps_sigma_min_Yee = epsilon_Yee/delta_t - sigma_Yee/2

eq2_matrix = np.divide(delta_t, np.multiply(mu_Yee, delta_x_matrix_Yee))

### Initialization of all the fields.
ez_Yee_new = np.zeros((M_Yee, N_Yee))
hy_Yee_new = np.zeros((M_Yee, N_Yee))
bx_Yee_new = np.zeros((M_Yee, N_Yee))


### Value of the sources based on the definitions above.
jz_Yee = def_jz(J0_Yee, source_Yee, M_Yee, N_Yee, source_X_Yee, source_Y_Yee, iterations, delta_t, tc_Yee, sigma_source_Yee, period_Yee, 1/(delta_x_Yee[0]*delta_y_Yee[0]))

observation_points_ez_Yee = [(150, 100)]
ez_Yee_list_observe = np.zeros((iterations, len(observation_points_ez_Yee)))

# initialization of the list of fields
bx_Yee_list = np.zeros((M_Yee, N_Yee, iterations))
ez_Yee_list = np.zeros((M_Yee, N_Yee, iterations))
hy_Yee_list = np.zeros((M_Yee, N_Yee, iterations))

### Starting the run

for n in range(iterations):
    print(f'iteration {n+1}/{iterations} started')

    ez_Yee_old = ez_Yee_new
    hy_Yee_old = hy_Yee_new
    bx_Yee_old = bx_Yee_new


    # Update ez in Yee region
    eq_4_hy = np.divide(hy_Yee_old, delta_x_matrix_Yee)
    eq_4_bx = np.divide(bx_Yee_old, np.multiply(delta_y_matrix_Yee, mu_Yee))
    eq_4_term = np.multiply(eps_sigma_min_Yee, ez_Yee_old) - (jz_Yee[:,:,n]+jz_Yee[:,:,n-1])/2 + eq_4_hy - np.roll(eq_4_hy, 1, 0) - eq_4_bx + np.roll(eq_4_bx, 1, 1)
    ez_Yee_new = np.divide(eq_4_term, eps_sigma_plus_Yee)

    ez_Yee_new[:,0] = np.zeros(M_Yee)

    # Bx is explicitly updated in the Yee region
    eq_3_term = np.divide(ez_Yee_new*delta_t, delta_y_matrix_Yee)
    bx_Yee_new = bx_Yee_old - np.roll(eq_3_term, -1, 1) + eq_3_term

    # Hy is explicitly updated in the Yee region
    eq_2_term = np.multiply(eq2_matrix, ez_Yee_new)
    hy_Yee_new = hy_Yee_old + np.roll(eq_2_term, -1, 0) - eq_2_term

    bx_Yee_list[:,:,n] = bx_Yee_new
    ez_Yee_list[:,:,n] = ez_Yee_new
    hy_Yee_list[:,:,n] = hy_Yee_new


    for i, point in enumerate(observation_points_ez_Yee):
        ez_Yee_list_observe[n, i] = ez_Yee_new[point]

for i, point in enumerate(observation_points_ez_Yee):
    plt.plot(range(iterations), ez_Yee_list_observe[:,i])
    plt.xlabel('Iteration')
    plt.ylabel('Ez in Yee region')
    plt.title(f'Ez at {point}')
    plt.show()


animation_speed = 5
if y_dependent:
    fig, ax = plt.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    for mat in material_list_Yee:
        rect = patches.Rectangle((mat[1], mat[3]), mat[2]-mat[1], mat[4]-mat[3], edgecolor = mat[5], linewidth=1, facecolor="none", label = mat[0].name)
        ax.add_patch(rect)

    def animate(i):
        ax.pcolormesh(np.transpose(ez_Yee_list[:,:,int(i*animation_speed)]))
        ax.set_title(f'n = {int(i*animation_speed)}')
        for mat in material_list_Yee:
            rect = patches.Rectangle((mat[1], mat[3]), mat[2]-mat[1], mat[4]-mat[3], edgecolor = mat[5], linewidth=1, facecolor="none", label = mat[0].name)
            ax.add_patch(rect)

    anim = FuncAnimation(fig, animate)
    plt.legend()
    plt.show()
else:
    fig, ax = plt.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    for mat in material_list_Yee:
        rect = patches.Rectangle((mat[1], 0), mat[2]-mat[1], N_Yee-1, edgecolor = mat[3], linewidth=1, facecolor="none", label = mat[0].name)
        ax.add_patch(rect)

    def animate(i):
        ax.pcolormesh(np.transpose(ez_Yee_list[:,:,int(i*animation_speed)]))
        ax.set_title(f'n = {int(i*animation_speed)}')
        for mat in material_list_Yee:
            rect = patches.Rectangle((mat[1], 0), mat[2]-mat[1]-1, N_Yee-1, edgecolor = mat[3], linewidth=1, facecolor="none", label = mat[0].name)
            ax.add_patch(rect)

    anim = FuncAnimation(fig, animate)
    plt.legend()
    plt.show()