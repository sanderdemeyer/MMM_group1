import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
from functions import def_update_matrices, update_implicit, def_jz
from  material_properties import Material, Material_grid
import matplotlib.patches as patches
import pickle

Lx = 10 # Length in the x-direction in units m
Ly = 10 # Length in the x-direction in units m

M = 800 # Number of cells in the x-direction
N = 800 # Number of cells in the y-direction
partition = 'uniform' # delta_x and delta_y are then constants. If partition != uniform, these should be specified as arrays.

iterations = 100 # Number of iterations. The total time length that is simulated is then equal to iterations * delta_t


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

grid = 'MIS'

if grid == 'dielectric':
    left = 100
    right = 140

    material_list = [[Mat3, left, right, 'blue']]
elif grid == 'MIS':
    Si_left = 100
    Si_right = 120
    SiO2_right = 125
    Cu_right = 130

    material_list = [[Si, Si_left, Si_right, 'blue'], 
                     [SiO2, Si_right, SiO2_right, 'yellow'],
                     [Cu, SiO2_right, Cu_right, 'red']
                     ]
elif grid == 'vacuum':
    material_list = []

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
courant_number = 1.1
delta_t = np.min(delta_y)/(c)*courant_number # in units s

print(f'duration is {delta_t*iterations} seconds')

### Definition of the source 
# The source type should be either dirac, gaussian, gaussian_modulated, or gaussian_modulated_dirac
source = 'gaussian_modulated_dirac' # type of the source
x_source = 60 # x-coordinate of the source. Make sure this is within bounds.
y_source = 100 # y-coordinate of the source. Make sure this is within bounds.
J0 = 1 # amplitude of the source in units V^2 m A^-1
tc = 5 # tc*delta_t is the time the source peaks
sigma_source = 2.2 # spread of the source in the case of gaussian or gaussian_modulated source
period = 10 # period of the source in number of time steps in the case of gaussian or gaussian_modulated source
omega_c = (2*np.pi)/(period*delta_t) # angular frequency of the source in the case of gaussian or gaussian_modulated source

jz = def_jz(J0, source, M, N, x_source, y_source, iterations, delta_t, tc, sigma_source, period, 1/(delta_x[x_source]*delta_y[y_source]))

spectral_content = fft.fft(jz[x_source,y_source,:])[0]


observation_points_ez = [(x_source + i, y_source) for i in range(M//2)] # observation points for the electric field

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

def step(ez_old, hy_old, bx_old, A_inv, B, n):
    [ez_new, hy_new] = update_implicit(ez_old, hy_old, bx_old, n, A_inv, B, delta_t, delta_y_matrix, M, N, jz, mu)
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
    [A, B] = def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, M)
    
    A_inv = linalg.inv(A)
    Eigenvalues = np.array(linalg.eigvals(np.dot(A_inv,B)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None')
    ax.add_patch(circ)
    plt.scatter(Eigenvalues.real,Eigenvalues.imag, s=2, c = 'r', marker='o')
    plt.show()
    if all(abs(i) <= 1.000000005 for i in Eigenvalues):
        print('Eigenvalues lie inside of unit circle')

    else:
        print('At least one eigenvalue outside of unit circle')
    test = 0
    for i in range(len(Eigenvalues)):
        if abs(Eigenvalues[i]) > 1.00000000:
            print(Eigenvalues[i])
        if abs(Eigenvalues[i]) > test:
            test = abs(Eigenvalues[i])
    print(test)
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


animation_speed = 3

fig, ax = plt.subplots()
ax.set_xlabel('Y')
ax.set_ylabel('X')
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

def hankel(x, f, J0=1):
    omega = 2*np.pi*f
    return -(J0*omega*mu_0/4)*special.hankel2(0, (omega*x/c))

frequency_point = 20

fft_transform_r_values = [i*delta_x[0] for i in range(M//2)]
fft_list = []



#plt.plot(range(iterations), ez_list_observe[:,40])
#plt.show()

with open('Yee_epsr_3.pkl', 'rb') as f:
    [ez_Yee_list_observe, iterations_Yee, times_Yee] = pickle.load(f)

times = [i*delta_t for i in range(iterations)]

for i, point in enumerate(observation_points_ez):
    plt.plot(times_Yee, ez_Yee_list_observe[:,i], label = 'Yee')

    plt.plot(times, ez_list_observe[:,i], label = 'UCHIE')
    plt.xlabel('Time [s]')
    plt.ylabel('Ez')
    plt.title(f'Ez at {point}')
    plt.legend()
    plt.show()

