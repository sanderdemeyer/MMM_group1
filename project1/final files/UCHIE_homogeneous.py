import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
from functions import def_update_matrices, update_implicit_faster, def_jz
from  material_properties import Material, Material_grid
import matplotlib.patches as patches

### File that implements the UCHIE-code. Relevant paramters to be changed are:
### Lx, Ly, M, N, iterations, material_list (or grid), the source parameters, and the observation points.

Lx = 5 # Length in the x-direction in units m
Ly = 5 # Length in the y-direction in units m

M = 200 # Number of cells in the x-direction
N = 200 # Number of cells in the y-direction
partition = 'uniform' # delta_x and delta_y are then constants. If partition != uniform, delta_x and delta_y should be specified as arrays.


iterations = 250 # Number of iterations. The total time length that is simulated is then equal to iterations * delta_t
simulation_time = None # If set to None, the variable iterations is unaltered. 
# If a value is given, this is the simulation time in units of c, thus making iterations equal to int(simulation_time*c/delta_t)

### Definitions of physical constants
epsilon_0 = 8.85*10**(-12)  # in units F/m
mu_0 = 1.25663706*10**(-6) # in units N/A^2
c = 3*10**8 # in units m/s

### Definition of the material properties. These properties should not depend on the y-coordinate.
epsilon = np.ones((M,N))*epsilon_0
mu = np.ones((M,N))*mu_0
sigma = np.ones((M,N))*0 # in units kg m^3 s^-3 A^-2 = V m^2 A^-1

Si = Material('Silicon')
Cu = Material('Copper')
SiO2 = Material('Silica')
Mat3 = Material(['epsilon_r_3', 3, 1, 0])
Lossy = Material(['lossy', 3, 1, 0.1])

grid = 'vacuum'

if grid == 'vacuum':
    material_list = []
elif grid == 'dielectric':
    left = 100
    right = 140

    material_list = [[Mat3, left, right, 'blue']]
elif grid == 'lossy':
    material_list = [[Lossy, 100, 140, 'blue']]
elif grid == 'MIS':
    Si_left = 100
    Si_right = 120
    SiO2_right = 125
    Cu_right = 130

    material_list = [[Si, Si_left, Si_right, 'blue'], 
                     [SiO2, Si_right, SiO2_right, 'yellow'],
                     [Cu, SiO2_right, Cu_right, 'red']
                     ]
else:
    print('Invalid value of variable grid')

material_grid = Material_grid(material_list)
[epsilon, mu, sigma] = material_grid.set_properties(epsilon, mu, sigma)

# epsilon[60:90,:] = np.ones((30,N))*3*epsilon_0

if partition == 'uniform':
    delta_x = np.ones(M)*Lx/M
    delta_y = np.ones(N)*Ly/N

   # delta_x[Cu_left:Cu_right] = delta_x[Cu_left:Cu_right]/100
   # delta_x[Si_left:Si_right] = delta_x[Si_left:Si_right]/10
else:
    delta_x = 0 # specify explicitly
    delta_y = 0 # specify explicitly

#delta_x = [((i+1)**(1/10))*10/M for i in range(M)]

### These matrices contain the delta_x and delta_y values at a given vertex. 
delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
delta_y_matrix = np.array([delta_y for i in range(M)])

### Definition of the courant number and the corresponding delta_t.
courant_number = 1
delta_t = np.min(delta_y)/(c)*courant_number # in units s

if simulation_time is not None:
    iterations = int(simulation_time/delta_t)

print(f'duration is {delta_t*iterations} seconds')

### Definition of the source 
# The source type should be either dirac, gaussian, gaussian_modulated, or gaussian_modulated_dirac.
# gaussian_modulated_dirac is a modulated gaussian in time, defined only on the point of the source. The source for other spatial points is zero.
source = 'gaussian_modulated_dirac' # type of the source
x_source = 100 # x-coordinate of the source. Make sure this is within bounds.
y_source = 100 # y-coordinate of the source. Make sure this is within bounds.
J0 = 1 # amplitude of the source in units V^2 m A^-1
tc = 5 # tc*delta_t is the time the source peaks
sigma_source = 2.2 # spread of the source in the case of gaussian or gaussian_modulated source
period = 10 # period of the source in number of time steps in the case of gaussian or gaussian_modulated source
omega_c = (2*np.pi)/(period*delta_t) # angular frequency of the source in the case of gaussian or gaussian_modulated source

jz = def_jz(J0, source, M, N, x_source, y_source, iterations, delta_t, tc, sigma_source, period, 1/(delta_x[x_source]*delta_y[y_source]))

print('source defined')
#spectral_content = fft.fft(jz[x_source,y_source,:])[0]
#print(f'spec cont is {spectral_content}')
#jz = jz/spectral_content


observation_point = ((80, 100))
observation_points_ez = [observation_point]

def update_bx(bx_old, ez_old):
    bx = np.zeros((M, N))
    bx[:,:-1] = bx_old[:,:-1] - (ez_old[:,1:] - ez_old[:,:-1])
    bx[:,-1] = bx_old[:,-1] - (ez_old[:,0] - ez_old[:,-1]) # add periodic boundary condition
    return bx


def def_explicit_update_matrix():
    E = np.zeros((M*N, M*N))
    for b in range(M*N-1):
        i = b // N
        E[b, b+1] = -delta_x[i]/delta_t
        E[b, b] = delta_x[i]/delta_t
    return E

def step(ez_old, hy_old, bx_old, A_inv, A_invB, n):
    [ez_new, hy_new] = update_implicit_faster(ez_old, hy_old, bx_old, n, A_inv, A_invB, delta_t, delta_y_matrix, M, N, jz, mu)
    bx_new = update_bx(bx_old, ez_new)
    return [ez_new, hy_new, bx_new]


def run_UCHIE():
    # initialization of the fields
    ez = np.zeros((M,N))
    hy = np.zeros((M,N))
    bx = np.zeros((M,N))

    # initialization of the list of fields
    bx_list = np.zeros((M,N, iterations))
    ez_list = np.zeros((M,N, iterations))
    hy_list = np.zeros((M,N, iterations))

    # initialization of the list of e_z values at the observation points.
    ez_list_observe = np.zeros((iterations, len(observation_points_ez)))

    # Definition of the UCHIE implicit update matrices.
    print('started with things')
    [A, B] = def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, M)
    A_inv = linalg.inv(A)
    A_invB = np.dot(A_inv, B)

    check_eigenvalues = False

    if check_eigenvalues:
        Eigenvalues = np.array(linalg.eigvals(np.dot(A_inv, B)))
        abs_eigenvalues = np.abs(Eigenvalues)
        print(f'maximal eigenvalue has magnitude {np.max(abs_eigenvalues)}')
        print(f'minimal eigenvalue has magnitude {np.min(abs_eigenvalues)}')
        print(Eigenvalues)
        figure, axes = plt.subplots()
        Drawing_uncolored_circle = plt.Circle( (0, 0 ),
                                            1 ,
                                            fill = False )
        axes.add_artist( Drawing_uncolored_circle )
        axes.scatter(Eigenvalues.real,Eigenvalues.imag, s = 0.5, c = 'red')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        axes.set_aspect(1)
        plt.xlabel(r'Re($ \lambda $)', fontsize = 10)
        plt.ylabel(r'Im($ \lambda $)', fontsize = 10)
        plt.title(r'Eigenvalues of $ A_{inv} B $ for the ' + grid + ' system')
        plt.show()

    for n in range(iterations):
        print(f'iteration {n+1}/{iterations} started')
        [ez, hy, bx] = step(ez, hy, bx, A_inv, A_invB, n)
        bx_list[:,:,n] = bx
        ez_list[:,:,n] = ez
        hy_list[:,:,n] = hy

        for i, point in enumerate(observation_points_ez):
            ez_list_observe[n, i] = ez[point]

    return bx_list, ez_list, hy_list, ez_list_observe

[bx_list, ez_list, hy_list, ez_list_observe] = run_UCHIE()


animation_speed = 3

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')
for mat in material_list:
    rect = patches.Rectangle((mat[1], 0), mat[2]-mat[1], N-1, edgecolor = mat[3], linewidth=1, facecolor="none", label = mat[0].name)
    ax.add_patch(rect)

def animate(i):
    ax.pcolormesh(np.transpose(ez_list[:,:,int(i*animation_speed)]))
    ax.set_title(f'n = {int(i*animation_speed)}')
    for mat in material_list:
        rect = patches.Rectangle((mat[1], 0), mat[2]-mat[1]-1, N-1, edgecolor = mat[3], linewidth=1, facecolor="none", label = mat[0].name)
        ax.add_patch(rect)
anim = FuncAnimation(fig, animate)
plt.legend()
plt.show()
