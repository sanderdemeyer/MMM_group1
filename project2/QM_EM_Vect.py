import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix
from scipy.integrate import quad

#define (fundamental) constants : hbar,massas,lenghts
hbar = constants.hbar
m = 0.15*constants.m_e 
c = constants.c
q = -constants.e

L_x = 0.5*10**(-9) # Length in the x-direction in meter
L_y = 100*10**(-9) # Length in the y-direction in meter
N = 9*10**(27)
omega_HO = 10*10**(12) # frequency of the HO
Eg_0 = 5*10**6 # Amplitude of the PW-pulse if this is Gaussian
Es_0 = 1*10**5 # Amplitude of the PW-pulse if this is a sine wave
sigma_t = 10*10**(-15) # Width of the gaussian pulse
t0 = 20*10**(-15) # Center of the gaussian pulse
alpha = 0.9 #should be between 0.9 and 1.1
omega_EM = alpha*omega_HO
delta_x = 1.0*10**(-6) # grid size in the x-direction in meter
delta_y = 0.5*10**(-9) # grid size in the y-direction in meter
n_y = int(L_y/delta_y) + 1 # Number of y grid cells
t_sim = 1*10**(-12) # Total simulated time in seconds
#provide location of structure through boundary of y-domain
y_start = -L_y/2
Courant = 1 # Courant number
delta_t = 1/c*Courant*delta_y # time step based on the Courant number
n_t = int(t_sim/delta_t) # Total number of time steps
y_axis = np.linspace(y_start,y_start + (n_y-1)*delta_y,n_y)
#initialize both real and imaginary parts of the wave function psi. In case alpha is not real, the initialization needs to be adapted.
alpha_y = 0

coupling = False

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
        V[i,i] = (1/2*m)*(omega_HO**2)*(y**2)
        H_int[i,i] = -q*y
    V = csr_matrix(V)
    V.eliminate_zeros()
    H_int = csr_matrix(H_int)
    H_int.eliminate_zeros()
    return V,H_int

def ABC():
    print('to do')


def run(coupling):
    psi_r = np.zeros((n_y,n_t))
    psi_im = np.zeros((n_y,n_t))
    Norm = np.zeros(n_t)

    A = update_matrix()
    V,H_int = harmonic_potential_and_length()

    psi_r_old = (m*omega_HO/constants.pi/hbar)**(1/4)*np.exp(-m*omega_HO/2/hbar*(y_axis-(2*hbar/m/omega_HO)**(1/2)*alpha_y)**2)
    psi_r_old = np.roll(psi_r_old, n_t//8)
    psi_r[:,0] = psi_r_old
    psi_im_old = np.zeros(n_y)
    Norm[0] = (np.sum(psi_r_old**2) + np.sum(psi_im_old**2))*delta_y
    """
    for i in range(n_y):
        y = y_start + delta_y*i
        psi_r[i,0] = (m*omega_HO/constants.pi/hbar)**(1/4)*np.exp(-m*omega_HO/2/hbar*(y-(2*hbar/m/omega_HO)**(1/2)*alpha_y)**2)
        Norm[i,0] = psi_r[i,0]**2
    """

    for i in range(1,n_t):
        if coupling == False:
            psi_r_new = psi_r_old - hbar*delta_t*(A @ psi_im_old)/(24*m*(delta_y**2)) + delta_t*(V @ psi_im_old)/hbar
            psi_im_new = psi_im_old + hbar*delta_t/24/(m*delta_y**2)*(A @ psi_r_new) - delta_t/hbar*(V @ psi_r_new)
        else:
            H_int = H_int*Exciting_PW('Gaussian_pulse',i)
            psi_r_new = psi_r_old - hbar*delta_t/24/(m*delta_y**2)*(A*psi_im_old) + delta_t/hbar*(V*psi_im_old + H_int*psi_im_old)
            psi_im_new = psi_im_old + hbar*delta_t/24/(m*delta_y**2)*(A*psi_r_new) - delta_t/hbar*(V*psi_r_new + H_int*psi_r_new)
       
        psi_r[:,i] = psi_r_new
        psi_im[:,i] = psi_im_new 
        psi_r_old = psi_r_new
        psi_im_old = psi_im_new

        Norm[i] = (np.sum(psi_r_old**2) + np.sum(psi_im_old**2))*delta_y

        if i%1000 == 0:
            print(f'Done iteration {i} of {n_t}')
    return psi_r,psi_im, Norm

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

psi_r,psi_im,norm = run(coupling)

print(norm)

plt.plot(norm)
plt.title('norm')
plt.show()

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

animation_speed = 10000

fig, ax = plt.subplots()
ax.set_xlabel('Y')
ax.set_ylabel('psi')
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.plot(y_axis, psi_r[:,int(i*animation_speed)]**2 + psi_im[:,int(i*animation_speed)]**2, c = 'black')
    ax.set_title(f'n = {int(i*animation_speed)}')
anim = FuncAnimation(fig, animate)
plt.legend()
plt.show()
