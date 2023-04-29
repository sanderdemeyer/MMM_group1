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
L_x = 0.5*10**(-9)
L_y = 100*10**(-9)
N = 9*10**(27)
q = -constants.e
omega_HO = 10*10**(12)
Eg_0 = 5*10**6
Es_0 = 1*10**5
sigma_t = 10*10**(-15)
t0 = 20*10**(-15)
alpha = 0.9 #should be between 0.9 and 1.1
omega_EM = alpha*omega_HO
delta_x = 1.0*10**(-6)
delta_y = 0.5*10**(-9)
n_y = int(L_y/delta_y)
t_sim = 1*10**(-13)
#provide location of structure through boundary of y-domain
y_start = -L_y/2
Courant = 1
delta_t = 1/c*Courant*delta_y
n_t = int(t_sim/delta_t)
y_axis = np.linspace(y_start,y_start + n_y*delta_y,n_y)
#initialize both real and imaginary parts of the wave fucntion psi. In case alpha is not real, the initialization needs to be adapted.
alpha_y = 0

coupling = True

Norm = np.zeros((n_y,n_t))


def update_matrix():
    A = np.zeros((n_y,n_y))
    for i in range(n_y):
        A[i,i] = -30
        if i-2 >=0:
            A[i,i-2] = -1
        if i-1 >=0:
            A[i,i-1] = 16
        if i+1 <= n_y-1:
            A[i,i+1] = 16
        if i+2 <= n_y-1:
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
    for i in range(n_y):
        y = y_start + delta_y*i
        psi_r[i,0] = (m*omega_HO/constants.pi/hbar)**(1/4)*np.exp(-m*omega_HO/2/hbar*(y-(2*hbar/m/omega_HO)**(1/2)*alpha_y)**2)
        Norm[i,0] = psi_r[i,0]**2
    psi_r_old = psi_r[:,0]
    psi_im_old = psi_im[:,0]
    A = update_matrix()
    V,H_int = harmonic_potential_and_length()
    for i in range(1,n_t):
        if coupling == False:
            psi_r_new = psi_r_old - hbar*delta_t*(A*psi_im_old)/(24*m*(delta_y**2)) + delta_t*(V*psi_im_old)/hbar
            psi_im_new = psi_im_old + hbar*delta_t/24/(m*delta_y**2)*(A*psi_r_new) - delta_t/hbar*(V*psi_r_new)
        else:
            H_int = H_int*Exciting_PW('Gaussian_pulse',i)
            psi_r_new = psi_r_old - hbar*delta_t/24/(m*delta_y**2)*(A*psi_im_old) + delta_t/hbar*(V*psi_im_old + H_int*psi_im_old)
            psi_im_new = psi_im_old + hbar*delta_t/24/(m*delta_y**2)*(A*psi_r_new) - delta_t/hbar*(V*psi_r_new + H_int*psi_r_new)
       
        psi_r[:,i] = psi_r_new
        psi_im[:,i] = psi_im_new 
        psi_r_old = psi_r_new
        psi_im_old = psi_im_new
        if i%1000 == 0:
            print('Done iteration %d'%(i))
    return psi_r,psi_im

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
psi_r,psi_im = run(coupling)

Norm = np.zeros((n_y,n_t))
for i in range(n_t):
    for j in range(n_y):
        Norm[j,i] = (psi_r[j,i]**2 + psi_im[j,i]**2)
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