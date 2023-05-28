import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.animation import FuncAnimation, ArtistAnimation
from scipy.sparse import csr_matrix
from scipy.integrate import quad
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity as sparse_identity
import time

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
Eg_0 = 5*10**6 # Amplitude of the PW-pulse if it is Gaussian
Es_0 = 1*10**5 # Amplitude of the PW-pulse if it is a sine wave
f = 2 # factor by which the 'normal' width of the gaussian pulse is normalized
sigma_t = 10*10**(-15)*f # Width of the gaussian pulse
sigma_ramping = 30*10**(-15)
t0 = 20*10**(-15)*5 # Center of the gaussian pulse

alpha = 1 #should be between 0.9 and 1.1
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
delta_t = 1/c*Courant*delta_y # time step based on the Courant number. With the current definitions, this means that delta_t = 1.6667*10**(-18)
n_t = int(t_sim/delta_t) # Total number of time steps
y_axis = np.linspace(y_start,y_start + (n_y-1)*delta_y,n_y)
x_axis = np.linspace(0, (n_x-1)*delta_x,n_x)
#initialize both real and imaginary parts of the wave function psi. In case alpha is not real, the initialization needs to be adapted.
alpha_y = 0

t0_gauss = t0//delta_t
sigma_gauss = sigma_t//delta_t

safe_frequency = 1000
safe_points = n_t//safe_frequency+1

source = 'sine' # should be either 'gaussian' or 'sine'

"""
if source == 'gaussian':
    J0 = Eg_0*epsilon/np.sqrt(2*np.pi*sigma_t**2)*150
    sigma_gauss = sigma_t//delta_t
    t0_gauss = t0//delta_t
"""
x_place_qd = L_x/4 # place of the quantum dot
x_qd = int(x_place_qd/delta_x) # y-coordinate of the quantum dot



def run():
    ey = np.zeros((n_x, safe_points))
    hz = np.zeros((n_x, safe_points))

    ey_new = np.zeros(n_x)
    hz_new = np.zeros(n_x)

    for i in range(1,n_t):
        ey_old = ey_new
        hz_old = hz_new

        hz_new = hz_old - delta_t/(mu*delta_x)*(ey_old - np.roll(ey_old, 1))
            
        Jy = np.zeros(n_x)
        if source == 'gaussian':
            if i > t0_gauss - 5*sigma_gauss and i < t0_gauss + 5*sigma_gauss:
                #ey_new[n_x//3] = Eg_0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
                Jy[n_x//3] = -1110.6833660953741*np.sqrt(1/f)*Eg_0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
                #Jy[n_x//2] = -1110.6833660953741*np.sqrt(3/f)*Eg_0/np.sqrt(f)*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
            else:
                pass
        elif source == 'sine':
            Jy[n_x//3] = -5308.993524411968*Es_0*np.sin(omega_EM*i*delta_t)*(1+np.tanh((i*delta_t-t0)/sigma_ramping))/2
            Jy[n_x//2] = -5308.993524411968*Es_0*np.sin(omega_EM*i*delta_t)*(1+np.tanh((i*delta_t-t0)/sigma_ramping))/2
            #ey_new[n_x//3] = Es_0*np.sin(omega_EM*i*delta_t)*np.tanh((i*delta_t-t0)/sigma_ramping)
        else:
            #print('wrong source')
            pass        

        ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy

        S = c*delta_t/delta_x
        ey_new[0] = ey_old[1] + (1-S)/(1+S)*(ey_old[0]-ey_new[1])
        ey_new[-1] = ey_old[-2] + (1-S)/(1+S)*(ey_old[-1]-ey_new[-2])

        if i % safe_frequency == 0:
            ey[:,i//safe_frequency] = ey_new
            hz[:,i//safe_frequency] = hz_new


        if i%1000 == 0:
            print(f'Done iteration {i} of {n_t}')
    return ey, hz


ey,hz = run()

animation_speed = 7500//safe_frequency

animation_speed = 7500//safe_frequency

maximum = np.max(np.abs(ey[:,280]))
print(f'max is {maximum}')
print(f'should be {1123.0356172173870176477172221528*10**6/maximum}')

fig, ax = plt.subplots()
ax.set_xlabel('x position [m]')
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.set_ylabel('ey [V/m]')
    ax.plot(x_axis, ey[:,int(i*animation_speed)], c = 'black')
    ax.axvline(x = x_place_qd, c = 'red', label = 'quantum dot')
    ax.set_title(f'n = {int(i*animation_speed)}')

    #ax2 = ax.twinx()
    #ax2.set_ylabel('y')
    #ax2.scatter([x_place_qd for i in range(len(y_axis))], y_axis, c = psi_r[:,int(i*animation_speed)]**2 + psi_im[:,int(i*animation_speed)]**2)

anim = FuncAnimation(fig, animate)
plt.show()
