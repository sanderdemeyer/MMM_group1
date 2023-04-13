import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
from functions import def_update_matrices, update_implicit, def_jz
import scipy.optimize as opt

Lx = 1 # Length in the x-direction in units m
Ly = 1 # Length in the x-direction in units m

M = 100 # Number of cells in the x-direction
N = 100 # Number of cells in the y-direction
partition = 'uniform' # delta_x and delta_y are then constants. If partition != uniform, these should be specified as arrays.

iterations = 60 # Number of iterations. The total time length that is simulated is then equal to iterations * delta_t


### Definitions of physical constants
epsilon_0 = 8.85*10**(-12)  # in units F/m
mu_0 = 1.25663706*10**(-6) # in units N/A^2
c = 3*10**8 # in units m/s

### Definition of the material properties. These properties should not depend on the y-coordinate.
epsilon = np.ones((M,N))*epsilon_0
mu = np.ones((M,N))*mu_0
sigma = np.ones((M,N))*0 # in units kg m^3 s^-3 A^-2 = V m^2 A^-1

if partition == 'uniform':
    delta_x = np.ones(M)*Lx/M
    delta_y = np.ones(N)*Ly/N
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

### Definition of the source 
# The source type should be either dirac, gaussian, or gaussian_modulated
source = 'dirac' # type of the source
x_source = 50 # x-coordinate of the source. Make sure this is within bounds.
y_source = 50 # y-coordinate of the source. Make sure this is within bounds.
J0 = 1 # amplitude of the source in units V^2 m A^-1
tc = 5 # tc*delta_t is the time the source peaks
sigma_source = 1 # spread of the source in the case of gaussian or gaussian_modulated source
period = 10 # period of the source in number of time steps in the case of gaussian or gaussian_modulated source
omega_c = (2*np.pi)/(period*delta_t) # angular frequency of the source in the case of gaussian or gaussian_modulated source

jz = def_jz(J0, source, M, N, x_source, y_source, iterations, delta_t, tc, sigma_source, period)

spectral_content = fft.fft(jz[x_source,y_source,:])[0]
jz = jz/spectral_content

observation_points_ez = [(x_source + i, y_source) for i in range(M//2)] # observation points for the electric field


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


def hankel(x, omega, J0=1):
    return -(J0*omega*mu_0/4)*special.hankel2(0, (omega*x/c))

frequency_point = 20

fft_transform_r_values = [i*delta_x[0] for i in range(M//2)]
fft_list = []

plt.plot(range(iterations), ez_list_observe[:,40])
plt.show()

for i, point in enumerate(observation_points_ez):
    """
    plt.plot(range(iterations), ez_list_observe[:,i])
    plt.xlabel('Time [s]')
    plt.ylabel('Ez')
    plt.title(f'Ez at {point}')
    #plt.show()
    """
    fft_transform = fft.fft(ez_list_observe[:,i])
    #plt.plot(fft_transform)
    #plt.show()
    fft_list.append(fft_transform[frequency_point])

omega = 1/delta_t*frequency_point/delta_x[0]*1
print(omega)

omega = frequency_point/(iterations*delta_t)*6.25

print(omega)
plt.plot(fft_transform_r_values, fft_list, label = 'computational')
plt.plot(fft_transform_r_values, np.array([hankel(i, omega) for i in fft_transform_r_values])*10**(-4.5), label = 'analytical')
plt.legend()
plt.show()

plt.plot(fft_transform_r_values, fft_list, label = 'computational')
plt.plot(np.array([hankel(delta_x[0]*i, 1/delta_t*frequency_point) for i in range(len(observation_points_ez))])/(3*10**(12)), label = 'analytical')
plt.legend()
plt.show()


"""
plt.plot(d_list[10:-10], [i/2 for i in v_list[10:-10]], label = 'computationally')
#plt.plot(d_list, [hankel(x+1) for x in d_list], label = 'exact solution')
plt.legend()
plt.show()
"""
plt.show()

plt.plot(ez_list[x_source, y_source*3//2,:])
plt.show()

animation_speed = 1

fig, ax = plt.subplots()
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_aspect('equal', adjustable='box')

def animate(i):
   ax.pcolormesh(np.transpose(ez_list[:,:,int(i*animation_speed)]))
   ax.set_title(f'n = {int(i*animation_speed)}')


print(bx_list[:,:,10])
print(ez_list[:,:,10])
print(hy_list[:,:,10])

anim = FuncAnimation(fig, animate)
plt.show()
