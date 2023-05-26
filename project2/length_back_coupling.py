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
sigma_t = 10*10**(-15)*5/2 # Width of the gaussian pulse
t0 = 20*10**(-15)*5 # Center of the gaussian pulse
alpha = 0.95 #should be between 0.9 and 1.1
omega_EM = alpha*omega_HO
omega_EM = 7.5398*10**(15) # This makes the period equal to 500 time steps
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

source = 'gaussian' # should be either 'gaussian' or 'sine'

if source == 'gaussian':
    J0 = Eg_0*epsilon/np.sqrt(2*np.pi*sigma_t**2)*150
    sigma_gauss = sigma_t//delta_t
    t0_gauss = t0//delta_t

x_place_qd = L_x/4 # place of the quantum dot
x_qd = int(x_place_qd/delta_x) # y-coordinate of the quantum dot

print(f'Zero-point energy is {omega_HO*hbar/2}')

gauge = 'length'

animation_speed = 2500
n_snapshots = n_t//animation_speed

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

def update_matrix_velocity():
    A = np.zeros((n_y,n_y))
    B = np.zeros((n_y,n_y))
    for i in range(n_y):
        if i == n_y-1:
            A[i,i] = -30
            A[i,i-1] = 16
            A[i,0] = 16
            A[i,i-2] = -1
            A[i,1] = -1

            B[i,i-1] = -8
            B[i,0] = 8
            B[i,i-2] = 1
            B[i,1] = -1
        elif i == n_y-2:
            A[i,i] = -30
            A[i,i-1] = 16
            A[i,i+1] = 16
            A[i,i-2] = -1
            A[i,0] = -1

            B[i,i-1] = -8
            B[i,i+1] = 8
            B[i,i-2] = 1
            B[i,0] = -1
        else:
            A[i,i] = -30
            A[i,i-1] = 16
            A[i,i+1] = 16
            A[i,i-2] = -1
            A[i,i+2] = -1

            B[i,i-1] = -8
            B[i,i+1] = 8
            B[i,i-2] = 1
            B[i,i+2] = -1
    A = csr_matrix(A)
    A.eliminate_zeros()
    B = csr_matrix(B)
    B.eliminate_zeros()
    return A, B

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

def run():
    psi_r = np.zeros((n_y,n_snapshots+1))
    psi_im = np.zeros((n_y,n_snapshots+1))
    psi_squared = np.zeros((n_y, n_snapshots+1))
    Norm = np.zeros(n_snapshots+1)

    ey = np.zeros((n_x, n_snapshots+1))
    hz = np.zeros((n_x, n_snapshots+1))
    ey_x_qd = np.zeros(n_t)

    A = update_matrix()
    V,H_int = harmonic_potential_and_length()
    V_diag = potential_diag()

    print('hooray')

    starting_n_point = n_y//4
    starting_n_point = 0
    # psi_r_old = (m*omega_HO/(constants.pi*hbar))**(1/4)*np.exp(-m*omega_HO/2/hbar*(y_axis-starting_point*(2*hbar/m/omega_HO)**(1/2)*alpha_y)**2)
    psi_r_old = (m*omega_HO/(constants.pi*hbar))**(1/4)*np.exp(-m*omega_HO/2/hbar*(y_axis - starting_n_point*delta_y)**2)
    psi_r[:,0] = psi_r_old
    psi_im_old = np.zeros(n_y)
    Norm[0] = (np.sum(psi_r_old**2) + np.sum(psi_im_old**2))*delta_y

    ey_old = np.zeros(n_x)
    hz_old = np.zeros(n_x)
    Jy = np.zeros(n_x)

    for i in range(1,n_t):
        hz_new = hz_old - delta_t/(mu*delta_x)*(ey_old - np.roll(ey_old, 1))

        psi_r_new = psi_r_old - hbar*delta_t*(
            -np.roll(psi_im_old,2)+16*np.roll(psi_im_old,1)-30*psi_im_old+16*np.roll(psi_im_old,-1)-np.roll(psi_im_old,-2)
            )/(24*m*(delta_y**2)) + delta_t*(V_diag*psi_im_old - q*y_axis*ey_old[x_qd]*psi_im_old)/hbar
        psi_im_new = psi_im_old + hbar*delta_t/24/(m*delta_y**2)*(
            -np.roll(psi_r_new,2)+16*np.roll(psi_r_new,1)-30*psi_r_new+16*np.roll(psi_r_new,-1)-np.roll(psi_r_new,-2)
            ) - delta_t/hbar*(V_diag*psi_r_new - q*y_axis*ey_old[x_qd]*psi_r_new)
        
        j_q = np.zeros(n_x)
        j_q[x_qd] = q*hbar*N/(2*m)*np.sum(psi_r_new*np.roll(psi_im_new+psi_im_old,-1) - np.roll(psi_r_new,-1)*(psi_im_new+psi_im_old))
        ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy - delta_t*L_x_size_quantum_dot/(epsilon*delta_x)*j_q
        
        if i > t0_gauss - 5*sigma_gauss and i < t0_gauss + 5*sigma_gauss:
            if source == 'gaussian':
                ey_new[n_x//3] = Eg_0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
            elif source == 'sine':
                ey_new[n_x//2] = Es_0*np.sin(omega_EM*i*delta_t)*np.tanh((i-t0)/sigma_t)
            else:
                print('wrong source')
        
        S = c*delta_t/delta_x
        ey_new[0] = ey_old[1] + (1-S)/(1+S)*(ey_old[0]-ey_new[1])
        ey_new[-1] = ey_old[-2] + (1-S)/(1+S)*(ey_old[-1]-ey_new[-2])

        if i % animation_speed == 0:
            t_now = i//animation_speed
            psi_r[:,t_now] = psi_r_new
            psi_im[:,t_now] = psi_im_new 
            psi_squared[:,t_now] = (psi_r_new)**2 + (psi_im_new)**2
            ey[:,t_now] = ey_new
            hz[:,t_now] = hz_new
            Norm[t_now] = (np.sum(psi_r_old**2) + np.sum(psi_im_old**2))*delta_y

        ey_x_qd[i] = ey_old[x_qd]

        psi_r_old = psi_r_new
        psi_im_old = psi_im_new

        ey_old = ey_new
        hz_old = hz_new

        if i%1000 == 0:
            print(f'Done iteration {i} of {n_t}')
    return psi_r,psi_im, Norm, ey, hz, psi_squared, ey_x_qd

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

def expectation_value_position(psi_r, psi_im, y_axis):
    exp_pos = np.sum((psi_r**2 + psi_im**2) * y_axis * delta_y,0)
    """
    exp_pos = np.zeros(n_t)
    for i in range(n_t):
        exp_pos[i] = np.sum((psi_r[:,i]**2 + psi_im[:,i]**2) * y_axis * delta_y)
    """
    return exp_pos

def expectation_value_momentum(psi_r, psi_im):
    exp_mom = np.sum(-1j*hbar*(psi_r - 1j*psi_im)*(np.roll(psi_r, -1,1) - psi_r + 1j*(np.roll(psi_im, -1,1) - psi_im)),0)
    """
    exp_mom = np.zeros(n_t)
    for i in range(n_t):
        psi_r_i = psi_r[:,i]
        psi_im_i = psi_im[:,i]
        exp_mom[i] = np.sum(-1j*hbar*(psi_r_i - 1j*psi_im_i)*(np.roll(psi_r_i, -1) - psi_r_i + 1j*(np.roll(psi_im_i, -1) - psi_im_i)))
    """
    return exp_mom

def expectation_value_kinetic_energy(psi_r, psi_im):
    #exp_kin = -hbar**2/(2*m)*np.sum((psi_r - 1j*psi_im)*(np.roll(psi_r,1,1)+np.roll(psi_r,-1,1)-2*psi_r + 1j*(np.roll(psi_im,1,1)+np.roll(psi_im,-1,1)-2*psi_im)),0)/delta_y
    exp_kin = -hbar**2/(2*m)*np.sum((psi_r - 1j*psi_im)*(np.roll(psi_r,1,0)+np.roll(psi_r,-1,0)-2*psi_r + 1j*(np.roll(psi_im,1,0)+np.roll(psi_im,-1,0)-2*psi_im)),0)/delta_y
    """
    exp_kin = np.zeros(n_t)
    for i in range(n_t):
        psi_r_i = psi_r[:,i]
        psi_im_i = psi_im[:,i]
        exp_kin[i] = -hbar**2/(2*m)*np.sum((psi_r_i - 1j*psi_im_i)*(np.roll(psi_r_i,1)+np.roll(psi_r_i,-1)-2*psi_r_i + 1j*(np.roll(psi_im_i,1)+np.roll(psi_im_i,-1)-2*psi_im_i)))/delta_y
    """
    return exp_kin

def expectation_value_potential_energy(psi_r, psi_im):
    V = potential_diag()
    V = np.transpose(np.array([V for t in range(n_snapshots+1)]))
    exp_pot = np.sum(((psi_r)**2 + (psi_im)**2)*V,0)*delta_y

    """
    exp_pot = np.zeros(n_t)
    V = potential_diag()
    for i in range(n_t):
        exp_pot[i] = np.sum(((psi_r[:,i])**2 + (psi_im[:,i])**2)*V)*delta_y
    """
    return exp_pot

def expectation_value_energy(psi_r, psi_im):
    exp_energy = np.sum(1j*hbar*(psi_r - 1j*psi_im)*(psi_r - np.roll(psi_r,1) + 1j*(psi_im - np.roll(psi_im,1))),0)*delta_y/(animation_speed*delta_t)
    """
    exp_energy = np.zeros(n_t-2)
    for i in range(1, n_t-1):
        exp_energy[i-1] = np.sum(1j*hbar*(psi_r[:,i] - 1j*psi_im[:,i])*(psi_r[:,i] - psi_r[:,i-1] + 1j*(psi_im[:,i] - psi_im[:,i-1])))*delta_y/delta_t
    """
    return exp_energy[1:-1]

def check_continuity_equation(psi_r, psi_im):
    error = np.zeros(n_t-1)
    
    for i in range(1,n_t):
        dPdt = ((psi_r[:,i]**2 + psi_im[:,i]**2) - (psi_r[:,i-1]**2 + psi_im[:,i-1]**2))/delta_t

        term1 = psi_r[:,i]*(np.roll(psi_im[:,i], 1) + np.roll(psi_im[:,i], -1) - 2*psi_im[:,i])/delta_y**2
        term2 = psi_im[:,i]*(np.roll(psi_r[:,i], 1) + np.roll(psi_r[:,i], -1) - 2*psi_r[:,i])/delta_y**2
        term1_with_const = hbar/(m)*term1
        term2_with_const = hbar/(m)*term2
        """
        print(np.sum(psi_r[:,i]**2 + psi_im[:,i]**2)*delta_y)
        print(np.sum(dPdt)*delta_y)
        print(np.sum(dPdt)*delta_y*delta_t)
        print(np.sum(term1_with_const)*delta_y)
        print(np.sum(term2_with_const)*delta_y)
        print(np.sum(term1_with_const - term2_with_const)*delta_y)
        print('hey1')
        print(term1_with_const[100:120])
        print('hey2')
        print(term2_with_const[100:120])
        print('hey3')
        print(term1_with_const[100:120] - term2_with_const[100:120])
        print('hey4')
        print(dPdt[100:120])
        print('hey5')
        if i > 1000:
            print('ok')
        print(np.divide((term1_with_const[100:120] - term2_with_const[100:120]), dPdt[100:120]))
        """
        error = dPdt + hbar/(m*delta_y**2)*(term1 - term2)
        error_tot = np.sum(error)
        error_min = dPdt - hbar/(m*delta_y**2)*(term1 - term2)
        error_min_tot = np.sum(error_min)


psi_r,psi_im,norm,ey,hz,psi_squared, ey_x_qd = run()


plt.plot(norm)
plt.title('norm')
plt.show()

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

exp_energy =  expectation_value_energy(psi_r, psi_im)
exp_pot =  expectation_value_potential_energy(psi_r, psi_im)
exp_kin =  expectation_value_kinetic_energy(psi_r, psi_im)


"""
print(exp_energy[200:210])
print(exp_pot[200:210])
print(exp_kin[200:210])
"""
plt.plot(range(len(exp_energy)), exp_energy, label = 'energy')
plt.plot(range(len(exp_pot)), exp_pot, label = 'potential energy')
plt.plot(range(len(exp_kin)), exp_kin, label = 'kinetic energy')
plt.title('expectation value of the energy')
plt.legend()
plt.show()


plt.plot(range(n_t), ey_x_qd)
plt.title('ey at quantum dot')
plt.show()

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

"""
animation_speed = 10

fig = plt.figure()

ims = []
for i in range(5994//animation_speed):
    ims.append([plt.plot(x_axis, ey[:,int(i*animation_speed)], c = 'black')])

anim = ArtistAnimation(fig, ims, interval=20)
plt.show()
"""

animation_speed = 2500

fig, ax = plt.subplots()
ax.set_xlabel('x')
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.set_ylabel('ey')
    ax.plot(x_axis, ey[:,int(i)], c = 'black')
    ax.axvline(x = x_place_qd, c = 'red', label = 'quantum dot')
    ax.set_title(f'n = {int(i*animation_speed)}')

    #ax2 = ax.twinx()
    #ax2.set_ylabel('y')
    #ax2.scatter([x_place_qd for i in range(len(y_axis))], y_axis, c = psi_r[:,int(i*animation_speed)]**2 + psi_im[:,int(i*animation_speed)]**2)

anim = FuncAnimation(fig, animate)
plt.show()


animation_speed = 2500

fig, ax = plt.subplots()
ax.set_xlabel('Y')
ax.set_ylabel('psi')
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.plot(y_axis, psi_r[:,int(i)]**2 + psi_im[:,int(i)]**2, c = 'black')
    ax.set_title(f'n = {int(i*animation_speed)}')
anim = FuncAnimation(fig, animate)
plt.show()