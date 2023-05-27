import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix
from scipy.integrate import quad
"""
#define (fundamental) constants : hbar,massas,lenghts
mu = constants.mu_0
epsilon = constants.epsilon_0
c = constants.c
print(c)
print(1/np.sqrt(mu*epsilon))
L_x = 50*10**(-9)
delta_x = 0.05*10**(-9) # grid size in the x-direction in meter
n_x = int(L_x/delta_x) + 1 # Number of y grid cells
#provide location of structure through boundary of y-domain
Courant = 1 # Courant number
delta_t = Courant*delta_x/c # time step based on the Courant number
n_t = 1000 # Total number of time steps
print(n_t)
def run():
    ey = np.zeros((n_x, n_t))
    hz = np.zeros((n_x, n_t))

    ey_old = np.zeros(n_x)
    hz_old = np.zeros(n_x)

    for i in range(1,n_t):

        hz_new = hz_old - delta_t/(mu*delta_x)*(ey_old - np.roll(ey_old, 1))
       
        Jy = np.zeros(n_x)
        if i < 100 and i > 50:
            Jy[n_x//2] = np.exp(-(i-75)**2/20)
        ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy

        ey[:,i] = ey_new
        hz[:,i] = hz_new
        ey_old = ey_new
        hz_old = hz_new

        if i%1000 == 0:
            print(f'Done iteration {i} of {n_t}')
    return ey, hz

ey,hz = run()

print( delta_t/(mu*delta_x))
print(delta_t/(epsilon*delta_x))
animation_speed = 5

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('ey')
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.plot(ey[:,int(i*animation_speed)], c = 'black')
    ax.set_title(f'n = {int(i*animation_speed)}')

plt.legend()
anim = FuncAnimation(fig, animate)
plt.show()

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix
from scipy.integrate import quad
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse import identity as sparse_identity
#define (fundamental) constants : hbar,massas,lenghts
hbar = constants.hbar
m = 0.15*constants.m_e 
c = constants.c
q = -constants.e
mu = constants.mu_0
epsilon = constants.epsilon_0

L_x = 1000*10**(-6) # Length in the x-direction in meter. Not to be confused by Lx as defined in the assignment, which is the width of a quantum dot.
L_y = 100*10**(-9) # Length in the y-direction in meter
L_x_size_quantum_dot = 0.5*10**(-9)
N = 9*10**(27)
omega_HO = 10*10**(12) # frequency of the HO
Eg_0 = 5*10**6 # Amplitude of the PW-pulse if this is Gaussian
Es_0 = 1*10**5 # Amplitude of the PW-pulse if this is a sine wave
sigma_t = 10*10**(-15) # Width of the gaussian pulse
t0 = 20*10**(-15) # Center of the gaussian pulse
alpha = 0.95 #should be between 0.9 and 1.1
omega_EM = alpha*omega_HO
#omega_EM = 7.5398*10**(15) # This makes the period equal to 500 time steps
delta_x = 1*10**(-6) # grid size in the x-direction in meter
delta_y = 0.5*10**(-9) # grid size in the y-direction in meter
n_y = int(L_y/delta_y) + 1 # Number of y grid cells
n_x = int(L_x/delta_x) + 1 # Number of x grid cells
t_sim = 10**(-12) # Total simulated time in seconds
#provide location of structure through boundary of y-domain
y_start = -L_y/2
Courant = 1 # Courant number
delta_t = 1/c*Courant*delta_y # time step based on the Courant number
n_t = int(t_sim/delta_t) # Total number of time steps
y_axis = np.linspace(y_start,y_start + (n_y-1)*delta_y,n_y)
x_axis = np.linspace(0, (n_x-1)*delta_x,n_x)
#initialize both real and imaginary parts of the wave function psi. In case alpha is not real, the initialization needs to be adapted.
alpha_y = 0

t0_gauss = t0//delta_t
sigma_gauss = sigma_t//delta_t
print(t0_gauss)
print(sigma_gauss)
"""
t0_gauss = 12000
sigma_gauss = 5000
"""
source = 'sine' # should be either 'gaussian' or 'sine'

x_place_qd = L_x/4 # place of the quantum dot
x_qd = int(x_place_qd/delta_x) # y-coordinate of the quantum dot

print(f'Zero-point energy is {omega_HO*hbar/2}')

coupling = True
back_coupling = False
gauge = 'length'

norm_every_step = False

def update_matrix():
    A = np.zeros((n_y,n_y))
    for i in range(n_y):
        A[i,i] = -30
        """
        if i-2 >=0:
            A[i,i-2] = -1
        if i-1 >=0:
            A[i,i-1] = 16
        if i+1 <= n_y-1:
            A[i,i+1] = 16
        if i+2 <= n_y-1:
            A[i,i+2] = -1
        """
        if i == n_y - 2:
            A[i,i-2] = -1
            A[i,i-1] = 16
            A[i,i+1] = 16
            A[i,0] = -1
        elif i == n_y - 1:
            A[i,i-2] = -1
            A[i,i-1] = 16
            A[i,0] = 16
            A[i,1] = -1
        else:
            A[i,i-2] = -1
            A[i,i-1] = 16
            A[i,i+1] = 16
            A[i,i+2] = -1

    A = csr_matrix(A)
    A.eliminate_zeros()
    return A


def harmonic_potential_and_length():
    V = np.zeros((n_y,n_y))
    H_int = np.zeros((n_y,n_y))
    for i in range(n_y):
        y = y_start + i*delta_y
        V[i,i] = (m/2)*(omega_HO**2)*(y**2)
        H_int[i,i] = -q*y
    V = csr_matrix(V)
    V.eliminate_zeros()
    H_int = csr_matrix(H_int)
    H_int.eliminate_zeros()
    return V,H_int

def potential_diag():
    V = np.zeros(n_y)
    for i in range(n_y):
        y = y_start + i*delta_y
        V[i] = (m/2)*(omega_HO**2)*(y**2)
    return V

def get_H_int(a):
    H_int = np.zeros((n_y,n_y))
    for i in range(n_y):
        y = y_start + i*delta_y
        H_int[i,i] = q**2/(2*m)*a**2
        H_int[i,i+1] = 0
    H_int = csr_matrix(H_int)
    H_int.eliminate_zeros()
    return H_int

def ABC():
    print('to do')

def run(coupling):
    psi_r = np.zeros((n_y,n_t))
    psi_im = np.zeros((n_y,n_t))
    psi_squared = np.zeros((n_y, n_t))
    psi_squared_cut = np.zeros((n_y, n_t//1000+1))
    Norm = np.zeros(n_t)

    ey = np.zeros((n_x, n_t))
    hz = np.zeros((n_x, n_t))

    A = update_matrix()
    V,H_int = harmonic_potential_and_length()

    ey_new = np.zeros(n_x)
    hz_new = np.zeros(n_x)

    """
    for i in range(n_y):
        y = y_start + delta_y*i
        psi_r[i,0] = (m*omega_HO/constants.pi/hbar)**(1/4)*np.exp(-m*omega_HO/2/hbar*(y-(2*hbar/m/omega_HO)**(1/2)*alpha_y)**2)
        Norm[i,0] = psi_r[i,0]**2
    """

    for i in range(1,n_t):
        ey_old = ey_new
        hz_old = hz_new

        hz_new = hz_old - delta_t/(mu*delta_x)*(ey_old - np.roll(ey_old, 1))

        sigma = 250
        Jy = np.zeros(n_x)
        """
        if i < 1000:
            J0 = 10**5*np.exp(-(i-500)**2/(2*sigma**2))
        else:
            J0 = 0
        """
        J0 = 10**(17)*1.3
        sigma_ramping = 100000

        """
        if source == 'gaussian':
            if i > t0_gauss - 5*sigma_gauss and i < t0_gauss + 5*sigma_gauss:
                Jy[n_x//2] = J0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
        elif source == 'sine':
            J0 = 10**(15)
            Jy[n_x//2] = J0*np.sin(omega_EM*i*delta_t)*np.tanh((i-t0)/sigma_ramping)
        else:
            print('wrong source')
        """
        #Jy[3*n_x//4] = J0
        j_q = 0
        #ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_old, -1) - hz_old) - (delta_t/epsilon)*Jy
        

        if source == 'gaussian':
            if i > t0_gauss - 5*sigma_gauss and i < t0_gauss + 5*sigma_gauss:
                ey_new[n_x//3] = Eg_0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
                
        elif source == 'sine':
            J0 = 10**5
            t0 = 30000
            sigma_ramping = 5000
            # ey_new[n_x//3] += Es_0*np.sin(omega_EM*i*delta_t)
            ey_new[n_x//3] = J0*np.sin(omega_EM*i*delta_t)*np.tanh((i*delta_t-t0)/sigma_t)

        else:
            print('wrong source')

        ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy - delta_t*L_x_size_quantum_dot/(epsilon*delta_x)*j_q

        S = c*delta_t/delta_x
        ey_new[0] = ey_old[1] + (1-S)/(1+S)*(ey_old[0]-ey_new[1])
        ey_new[-1] = ey_old[-2] + (1-S)/(1+S)*(ey_old[-1]-ey_new[-2])

        ey[:,i] = ey_new
        hz[:,i] = hz_new

        if i%1000 == 0:
            print(f'Done iteration {i} of {n_t}')
    return ey, hz

def f(t):
    '''
    Choose a ramping function
    '''
    return np.sin(t)

def Exciting_PW(arg,i):
    if arg == 'Gaussian_pulse':
        return Eg_0*np.exp((i*delta_t-t0)**2/(2*sigma_t**2))
    elif arg == 'Monochromatic_sine_wave':
        return Es_0*np.sin(omega_EM*i*delta_t)*f(i*delta_t)
    else:
        print('Invalid source')


ey,hz = run(coupling)



"""
exp_pos = expectation_value_position(psi_r, psi_im, y_axis)
plt.plot([i for i in range(n_t)], exp_pos)
plt.title('expectation value of the position')
plt.show()

print(exp_pos.imag)

exp_mom =  expectation_value_momentum(psi_r, psi_im)
plt.plot([i for i in range(n_t)], exp_mom)
plt.title('expectation value of the momentum')
plt.show()
"""


#error = check_continuity_equation(psi_r, psi_im)
#print(error)

"""
for i in range(n_t):
    for j in range(n_y):
        Norm[j,i] = (psi_r[j,i]**2 + psi_im[j,i]**2)
"""

"""
fig,ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.set_xlabel('Y')
    ax.set_ylabel('Prob.Ampl.')
    ax.set_title('With EM excitation')
    ax.set_ylim([0,10*10**8])
    ax.plot(y_axis,Norm)
    ax.set_title(f'n = {int(i*1000)}')
anim = FuncAnimation(fig,animate)
plt.legend()
plt.show()
"""


animation_speed = 1500

fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('ey')
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.plot(x_axis, ey[:,int(i*animation_speed)], c = 'black')
    ax.axvline(x = x_place_qd, c = 'red', label = 'quantum dot')
    ax.set_title(f'n = {int(i*animation_speed)}')

plt.legend()
anim = FuncAnimation(fig, animate)
plt.show()