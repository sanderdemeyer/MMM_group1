import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix
from scipy.integrate import quad

#define (fundamental) constants : hbar,massas,lenghts
mu = constants.mu_0
epsilon = constants.epsilon_0
c = constants.c
print(c)
print(1/np.sqrt(mu*epsilon))
L_x = 1
delta_x = 1*10**(-2) # grid size in the x-direction in meter
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
        if i == 2:
            Jy[n_x//2] = 1
        ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_old, -1) - hz_old) - (delta_t/epsilon)*Jy

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
animation_speed = 50

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