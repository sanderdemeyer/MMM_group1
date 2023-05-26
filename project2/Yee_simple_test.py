import numpy as np
import matplotlib.pyplot as plt
from scipy import constants


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
sigma_t = 10*10**(-15)*5/2 # Width of the gaussian pulse
t0 = 20*10**(-15)*5 # Center of the gaussian pulse
alpha = 1 #should be between 0.9 and 1.1
omega_EM = alpha*omega_HO
# omega_EM = 7.5398*10**(15) # This makes the period equal to 500 time steps
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

J0 = 10**5
t0 = 30000
sigma_ramping = 50000

i = np.linspace(0, 600000, 600000)

y = J0*np.sin(omega_EM*i*delta_t)*np.tanh((i-t0)/sigma_ramping)

print(omega_EM)
print(delta_t)
print(2*np.pi/(omega_EM*delta_t))
plt.plot(i, y)
plt.show()
