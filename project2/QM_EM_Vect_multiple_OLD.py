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
sigma_t = 10*10**(-15) # Width of the gaussian pulse
t0 = 20*10**(-15)*5 # Center of the gaussian pulse


alpha = 1 #should be between 0.9 and 1.1
omega_EM = alpha*omega_HO
#omega_EM = 7.5398*10**(15) # This makes the period equal to 500 time steps
delta_x = 1*10**(-6) # grid size in the x-direction in meter
delta_y = 0.5*10**(-9) # grid size in the y-direction in meter
n_y = int(L_y/delta_y) + 1 # Number of y grid cells
n_x = int(L_x/delta_x) + 1 # Number of x grid cells
t_sim = 10**(-12)/10 # Total simulated time in seconds
#provide location of structure through boundary of y-domain
y_start = -L_y/2
Courant = 1 # Courant number
delta_t = 1/c*Courant*delta_y # time step based on the Courant number. With the current definitions, this means that delta_t = 1.6667*10**(-18)
n_t = int(t_sim/delta_t) # Total number of time steps
y_axis = np.linspace(y_start,y_start + (n_y-1)*delta_y,n_y)
x_axis = np.linspace(0, (n_x-1)*delta_x,n_x)
#initialize both real and imaginary parts of the wave function psi. In case alpha is not real, the initialization needs to be adapted.
alpha_y = 0

n_sheets = 2

t0_gauss = t0//delta_t
sigma_gauss = sigma_t//delta_t

safe_frequency = 1000
safe_points = n_t//safe_frequency+1

source = 'gaussian' # should be either 'gaussian' or 'sine'

"""
if source == 'gaussian':
    J0 = Eg_0*epsilon/np.sqrt(2*np.pi*sigma_t**2)*150
    sigma_gauss = sigma_t//delta_t
    t0_gauss = t0//delta_t
"""
x_place_qd = [L_x/4, L_x*3/4] # place of the quantum dot
assert(len(x_place_qd) == n_sheets, 'Length of x_place_qd should be equal to n_sheets')
x_qd = [int(x_place_qd[sheet]/delta_x) for sheet in range(n_sheets)] # y-coordinate of the quantum dot

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

def run(coupling):
    psi_r = np.zeros((n_y,safe_points,n_sheets))
    psi_im = np.zeros((n_y,safe_points,n_sheets))
    psi_squared = np.zeros((n_y, safe_points,n_sheets))
    psi_squared_cut = np.zeros((n_y, safe_points,n_sheets))
    Norm = np.zeros((n_t, n_sheets))

    ey = np.zeros((n_x, safe_points))
    hz = np.zeros((n_x, safe_points))

    A = update_matrix()
    V,H_int = harmonic_potential_and_length()
    V_diag = potential_diag()
    
    a_list = np.zeros((safe_points, n_sheets))
    a = np.zeros(n_sheets)
    if gauge == 'velocity':
        A_vel, B_vel = update_matrix_velocity()
        E0 = 1
        """
        B_plus = sparse_identity(n_y) + q*E0*delta_t/(24*m*omega_EM*delta_y)*B_vel
        B_min = sparse_identity(n_y) - q*E0*delta_t/(24*m*omega_EM*delta_y)*B_vel
        A_plus = -hbar*delta_t/(24*m*delta_y**2)*A_vel + V*delta_t/hbar
        B_plus_inv = sparse_inv(B_plus)
        B_min_inv = sparse_inv(B_min)
        """
        A_plus = -hbar*delta_t/(24*m*delta_y**2)*A_vel + V*delta_t/hbar
        B_matrix = q*delta_t/(24*m*delta_y)*B_vel
    print('hooray')

    starting_n_point = [n_y//4, n_y//8]
    #starting_n_point = [0, 0]

    psi_r_new = np.zeros((n_y, n_sheets))
    psi_im_new = np.zeros((n_y, n_sheets))
    # psi_r_old = (m*omega_HO/(constants.pi*hbar))**(1/4)*np.exp(-m*omega_HO/2/hbar*(y_axis-starting_point*(2*hbar/m/omega_HO)**(1/2)*alpha_y)**2)
    psi_r_new[:,0] = (m*omega_HO/(constants.pi*hbar))**(1/4)*np.exp(-m*omega_HO/2/hbar*(y_axis - starting_n_point[0]*delta_y)**2)
    psi_r_new[:,1] = (m*omega_HO/(constants.pi*hbar))**(1/4)*np.exp(-m*omega_HO/2/hbar*(y_axis - starting_n_point[1]*delta_y)**2)
    psi_r[:,0,:] = psi_r_new
    psi_im[:,0,:] = psi_im_new
    Norm[0,:] = (np.sum(psi_r_new**2) + np.sum(psi_im_new**2))*delta_y

    if gauge == 'velocity':
        a_squared_int = np.zeros(n_sheets)
        #a_squared_integral_list = np.zeros(safe_points)

    ey_new = np.zeros(n_x)
    hz_new = np.zeros(n_x)
    Jy = np.zeros(n_x)

    for i in range(1,n_t):
        ey_old = ey_new
        hz_old = hz_new
        psi_r_old = psi_r_new
        psi_im_old = psi_im_new

        hz_new = hz_old - delta_t/(mu*delta_x)*(ey_old - np.roll(ey_old, 1))

        if source == 'gaussian':
            if i > t0_gauss - 5*sigma_gauss and i < t0_gauss + 5*sigma_gauss:
                ey_new[n_x//3] = Eg_0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
                #ey_new[int(n_x*1.3/3)] = Eg_0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
            else:
                pass
                #ey_new[n_x//3] = 0
        elif source == 'sine':
            ey_new[n_x//3] = Es_0*np.sin(omega_EM*i*delta_t)*np.tanh((i*delta_t-t0)/sigma_t)
            #ey_new[int(n_x*1.3/3)] = 0
        else:
            #print('wrong source')
            pass


        if gauge == 'length':
            if coupling == False:
                psi_r_new = psi_r_old - hbar*delta_t*(A @ psi_im_old)/(24*m*(delta_y**2)) + delta_t*(V @ psi_im_old)/hbar
                psi_im_new = psi_im_old + hbar*delta_t/24/(m*delta_y**2)*(A @ psi_r_new) - delta_t/hbar*(V @ psi_r_new)
                ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy
            else:
                if back_coupling == False:
                    #ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_old, -1) - hz_old) - (delta_t/epsilon)*Jy
                    """
                    psi_r_new = psi_r_old - hbar*delta_t*(A @ psi_im_old)/(24*m*(delta_y**2)) + delta_t*(V @ psi_im_old - q*y_axis*ey_old[x_qd]*psi_im_old)/hbar
                    psi_im_new = psi_im_old + hbar*delta_t/24/(m*delta_y**2)*(A @ psi_r_new) - delta_t/hbar*(V @ psi_r_new - q*y_axis*ey_old[x_qd]*psi_r_new)
                    """
                    for sheet in range(n_sheets):
                        psi_r_new[:,sheet] = psi_r_old[:,sheet] - hbar*delta_t*(
                            -np.roll(psi_im_old[:,sheet],2)+16*np.roll(psi_im_old[:,sheet],1)-30*psi_im_old[:,sheet]+16*np.roll(psi_im_old[:,sheet],-1)-np.roll(psi_im_old[:,sheet],-2)
                            )/(24*m*(delta_y**2)) + delta_t*(V_diag*psi_im_old[:,sheet] - q*y_axis*ey_old[x_qd[sheet]]*psi_im_old[:,sheet])/hbar
                        psi_im_new[:,sheet] = psi_im_old[:,sheet] + hbar*delta_t/24/(m*delta_y**2)*(
                            -np.roll(psi_r_new[:,sheet],2)+16*np.roll(psi_r_new[:,sheet],1)-30*psi_r_new[:,sheet]+16*np.roll(psi_r_new[:,sheet],-1)-np.roll(psi_r_new[:,sheet],-2)
                            ) - delta_t/hbar*(V_diag*psi_r_new[:,sheet] - q*y_axis/2*(ey_old[x_qd[sheet]]+ey_new[x_qd[sheet]])*psi_r_new[:,sheet])
                        ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy
                else:
                    j_q = np.zeros(n_x)
                    for sheet in range(n_sheets):
                        psi_r_new[:,sheet] = psi_r_old[:,sheet] - hbar*delta_t*(
                            -np.roll(psi_im_old[:,sheet],2)+16*np.roll(psi_im_old[:,sheet],1)-30*psi_im_old[:,sheet]+16*np.roll(psi_im_old[:,sheet],-1)-np.roll(psi_im_old[:,sheet],-2)
                            )/(24*m*(delta_y**2)) + delta_t*(V_diag*psi_im_old[:,sheet] - q*y_axis*ey_old[x_qd[sheet]]*psi_im_old[:,sheet])/hbar
                        psi_im_new[:,sheet] = psi_im_old[:,sheet] + hbar*delta_t/24/(m*delta_y**2)*(
                            -np.roll(psi_r_new[:,sheet],2)+16*np.roll(psi_r_new[:,sheet],1)-30*psi_r_new[:,sheet]+16*np.roll(psi_r_new[:,sheet],-1)-np.roll(psi_r_new[:,sheet],-2)
                            ) - delta_t/hbar*(V_diag*psi_r_new[:,sheet] - q*y_axis*ey_old[x_qd[sheet]]*psi_r_new[:,sheet])
                    
                        j_q[x_qd[sheet]] = q*hbar*N/(2*m)*np.sum(psi_r_new[:,sheet]*np.roll(psi_im_new[:,sheet]+psi_im_old[:,sheet],-1) - np.roll(psi_r_new[:,sheet],-1)*(psi_im_new[:,sheet]+psi_im_old[:,sheet]))
    
                    ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy - delta_t*L_x_size_quantum_dot/(epsilon*delta_x)*j_q

        if gauge == 'velocity':
            if coupling == False:
                pass
            else:
                if back_coupling == False:
                    ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy
                    for sheet in range(n_sheets):
                        a[sheet] += -ey_old[x_qd[sheet]]*delta_t

                    """
                    B_plus = sparse_identity(n_y) - q*a*delta_t/(24*m*delta_y)*B_vel
                    B_min = sparse_identity(n_y) + q*a*delta_t/(24*m*delta_y)*B_vel
                    B_plus_inv = sparse_inv(B_plus)
                    B_min_inv = sparse_inv(B_min)

                    psi_r_new = B_plus_inv @ (A_plus @ psi_im_old + B_min @ psi_r_old)
                    psi_im_new = B_min_inv @ (-A_plus @ psi_r_new + B_min @ psi_im_old)
                    """
                    B_plus = sparse_identity(n_y) + a*B_matrix
                    B_min = sparse_identity(n_y) - a*B_matrix

                    psi_r_new = spsolve(B_plus, A_plus @ psi_im_old + B_min @ psi_r_old)
                    #psi_im_new = spsolve(B_min, -A_plus @ psi_r_new + B_min @ psi_im_old)
                    psi_im_new = spsolve(B_plus, -A_plus @ psi_r_new + B_min @ psi_im_old)
                else:
                    B_plus = sparse_identity(n_y) + q*a*delta_t/(24*m*delta_y)*B_vel
                    B_min = sparse_identity(n_y) - q*a*delta_t/(24*m*delta_y)*B_vel

                    psi_r_new = spsolve(B_plus, A_plus @ psi_im_old + B_min @ psi_r_old)
                    #psi_im_new = spsolve(B_min, -A_plus @ psi_r_new + B_min @ psi_im_old)
                    psi_im_new = spsolve(B_plus, -A_plus @ psi_r_new + B_min @ psi_im_old)

                    j_q = np.zeros(n_x)
                    j_q[x_qd] = q*hbar*N/(2*m)*np.sum(psi_r_new*np.roll(psi_im_new+psi_im_old,-1) - np.roll(psi_r_new,-1)*(psi_im_new+psi_im_old))
                    ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy - delta_t*L_x_size_quantum_dot/(epsilon*delta_x)*j_q

                    a += -ey_new[x_qd]*delta_t
            a_squared_int += a**2*delta_t

        if norm_every_step:
            norm_new = np.sqrt(np.sum(psi_r_old**2+psi_im_old**2)*delta_y)
            psi_r_new /= norm_new
            psi_im_new /= norm_new

        """
        sigma = 250
        Jy = np.zeros(n_x)
        J0 = 0
        if source == 'gaussian':
            if i > t0_gauss - 5*sigma_gauss and i < t0_gauss + 5*sigma_gauss:
                Jy[n_x//3] = J0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
                
        elif source == 'sine':
            Jy[n_x//2] = J0*np.sin(omega_EM*i*delta_t)*np.tanh((i-t0)/sigma_t)
        else:
            print('wrong source')
        #Jy[3*n_x//4] = J0
        """

            
        S = c*delta_t/delta_x
        ey_new[0] = ey_old[1] + (1-S)/(1+S)*(ey_old[0]-ey_new[1])
        ey_new[-1] = ey_old[-2] + (1-S)/(1+S)*(ey_old[-1]-ey_new[-2])

        if i % safe_frequency == 0:
            ey[:,i//safe_frequency] = ey_new
            hz[:,i//safe_frequency] = hz_new
            if gauge == 'velocity':
                psi_L = np.exp(-1j*q**2/(2*m*hbar)*a_squared_int)*(psi_r_new + 1j*psi_im_new)
                psi_r[:,i//safe_frequency] = psi_L.real
                psi_im[:,i//safe_frequency] = psi_L.imag
                a_list[i//safe_frequency] = a
                #a_squared_integral_list[i//safe_frequency] = a_squared_int
            else:
                psi_r[:,i//safe_frequency] = psi_r_new
                psi_im[:,i//safe_frequency] = psi_im_new 
                a_list[i//safe_frequency] = a
            psi_squared[:,i//safe_frequency] = (psi_r_new)**2 + (psi_im_new)**2

        if i % 1000 == 0:
            psi_squared_cut[:,i//1000] = (psi_r_new)**2 + (psi_im_new)**2

        Norm[i] = (np.sum(psi_r_old**2) + np.sum(psi_im_old**2))*delta_y

        if i%1000 == 0:
            print(f'Done iteration {i} of {n_t}')
    return psi_r,psi_im, Norm, ey, hz, psi_squared, psi_squared_cut, a_list

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
    print('pos')
    print(psi_r)
    np.shape(psi_im)
    np.shape(y_axis)
    print(delta_y)
    y_axis = np.transpose(np.array([[y_axis for t in range(safe_points)] for n in range(n_sheets)]))
    exp_pos = np.sum((psi_r**2 + psi_im**2) * y_axis * delta_y,0)
    """
    exp_pos = np.zeros(n_t)
    for i in range(n_t):
        exp_pos[i] = np.sum((psi_r[:,i]**2 + psi_im[:,i]**2) * y_axis * delta_y)
    """
    return exp_pos

def expectation_value_momentum(psi_r, psi_im, a):
    exp_mom = np.sum(-1j*hbar*(psi_r - 1j*psi_im)*(np.roll(psi_r, -1,0) - psi_r + 1j*(np.roll(psi_im, -1,0) - psi_im)),0)
    exp_mom_2 = q*a
    """
    exp_mom = np.zeros(n_t)
    for i in range(n_t):
        psi_r_i = psi_r[:,i]
        psi_im_i = psi_im[:,i]
        exp_mom[i] = np.sum(-1j*hbar*(psi_r_i - 1j*psi_im_i)*(np.roll(psi_r_i, -1) - psi_r_i + 1j*(np.roll(psi_im_i, -1) - psi_im_i)))
    """
    return exp_mom + exp_mom_2

def expectation_value_kinetic_energy(psi_r, psi_im, a):
    #exp_kin = -hbar**2/(2*m)*np.sum((psi_r - 1j*psi_im)*(np.roll(psi_r,1,1)+np.roll(psi_r,-1,1)-2*psi_r + 1j*(np.roll(psi_im,1,1)+np.roll(psi_im,-1,1)-2*psi_im)),0)/delta_y
    exp_kin = -hbar**2/(2*m)*np.sum((psi_r - 1j*psi_im)*(np.roll(psi_r,1,0)+np.roll(psi_r,-1,0)-2*psi_r + 1j*(np.roll(psi_im,1,0)+np.roll(psi_im,-1,0)-2*psi_im)),0)/delta_y
    exp_kin_2 = -1j*hbar*q/m*a*np.sum((psi_r-1j*psi_im)*(psi_r+1j*psi_im - np.roll(psi_r+1j*psi_im, 1, 0)),0)
    exp_kin_3 = q**2/(2*m)*a**2

    """
    exp_kin = np.zeros(n_t)
    for i in range(n_t):
        psi_r_i = psi_r[:,i]
        psi_im_i = psi_im[:,i]
        exp_kin[i] = -hbar**2/(2*m)*np.sum((psi_r_i - 1j*psi_im_i)*(np.roll(psi_r_i,1)+np.roll(psi_r_i,-1)-2*psi_r_i + 1j*(np.roll(psi_im_i,1)+np.roll(psi_im_i,-1)-2*psi_im_i)))/delta_y
    """
    return exp_kin + exp_kin_2 + exp_kin_3

def expectation_value_potential_energy(psi_r, psi_im):
    V = potential_diag()
    V = np.transpose(np.array([[V for t in range(safe_points)] for n in range(n_sheets)]))
    exp_pot = np.sum(((psi_r)**2 + (psi_im)**2)*V,0)*delta_y

    """
    exp_pot = np.zeros(n_t)
    V = potential_diag()
    for i in range(n_t):
        exp_pot[i] = np.sum(((psi_r[:,i])**2 + (psi_im[:,i])**2)*V)*delta_y
    """
    return exp_pot

def expectation_value_energy(psi_r, psi_im):

    exp_energy = np.sum(1j*hbar*(psi_r - 1j*psi_im)*(psi_r - np.roll(psi_r,1,1) + 1j*(psi_im - np.roll(psi_im,1,1))),0)*delta_y/(delta_t*safe_frequency)
    """
    exp_energy2 = np.zeros((safe_points, n_sheets))
    for sheet in range(n_sheets):
        exp_energy2[:,sheet] = np.sum(1j*hbar*(psi_r[:,:,sheet] - 1j*psi_im[:,:,sheet])*(psi_r[:,:,sheet] - np.roll(psi_r[:,:,sheet],1) + 1j*(psi_im[:,:,sheet] - np.roll(psi_im[:,:,sheet],1))),0)*delta_y/(delta_t*safe_frequency)

    exp_energy = np.zeros(n_t-2)
    for i in range(1, n_t-1):
        exp_energy[i-1] = np.sum(1j*hbar*(psi_r[:,i] - 1j*psi_im[:,i])*(psi_r[:,i] - psi_r[:,i-1] + 1j*(psi_im[:,i] - psi_im[:,i-1])))*delta_y/delta_t
    """
    return exp_energy[1:-1,:]

def check_continuity_equation(psi_r, psi_im):
    error = np.zeros(n_t-1)
    for i in range(1,n_t):
        dPdt = ((psi_r[:,i]**2 + psi_im[:,i]**2) - (psi_r[:,i-1]**2 + psi_im[:,i-1]**2))/delta_t

        term1 = psi_r[:,i]*(np.roll(psi_im[:,i], 1) + np.roll(psi_im[:,i], -1) - 2*psi_im[:,i])/delta_y**2
        term2 = psi_im[:,i]*(np.roll(psi_r[:,i], 1) + np.roll(psi_r[:,i], -1) - 2*psi_r[:,i])/delta_y**2
        term1_with_const = hbar/(m)*term1
        term2_with_const = hbar/(m)*term2
        term_dif = term1_with_const - term2_with_const
        error = dPdt + (term1_with_const - term2_with_const)
        relative_error = np.divide(error, dPdt)
        relative_error_min = np.divide(dPdt - (term1_with_const - term2_with_const), dPdt)
        print(np.mean(relative_error))
        print(np.sum(relative_error))
        print(np.sum(error))
        print('again')
        print(np.sum(dPdt))

        print(np.sum(term_dif))
        if i == 1000:
            print('new')
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
            print('ok')

def check_continuity_equation_new(psi_r, psi_im):
    error = np.zeros(n_t-1)
    for i in range(1,n_t):
        dPdt = ((psi_r[:,i]**2 + psi_im[:,i]**2) - (psi_r[:,i-1]**2 + psi_im[:,i-1]**2))/delta_t

        term1 = psi_r[:,i]*(np.roll(psi_im[:,i], 1) + np.roll(psi_im[:,i], -1) - 2*psi_im[:,i])/delta_y**2
        term2 = psi_im[:,i]*(np.roll(psi_r[:,i], 1) + np.roll(psi_r[:,i], -1) - 2*psi_r[:,i])/delta_y**2
        term1_with_const = hbar/(m)*term1
        term2_with_const = hbar/(m)*term2

        error[i] = dPdt + hbar/(m)*(term1 - term2)
    error_tot = np.sum(error)
    return error_tot

psi_r,psi_im,norm,ey,hz,psi_squared,psi_squared_cut, a_list = run(coupling)


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

#check_continuity_equation(psi_r, psi_im)

print('started with energy')
exp_energy =  expectation_value_energy(psi_r, psi_im)
print('started with potential energy')
exp_pot =  expectation_value_potential_energy(psi_r, psi_im)
print('started with kinetic energy')
exp_kin =  expectation_value_kinetic_energy(psi_r, psi_im, a_list)

pos = expectation_value_position(psi_r, psi_im, y_axis)
mom = expectation_value_momentum(psi_r, psi_im, a_list)


"""
print(exp_energy[200:210])
print(exp_pot[200:210])
print(exp_kin[200:210])
"""
for sheet in range(n_sheets):
    plt.plot([i*delta_t/10**(-9) for i in range(safe_points-2)], exp_energy[:,sheet], label = 'energy')
    plt.plot([i*delta_t/10**(-9) for i in range(safe_points)], exp_pot[:,sheet], label = 'potential energy')
    plt.plot([i*delta_t/10**(-9) for i in range(safe_points)], exp_kin[:,sheet], label = 'kinetic energy')
    plt.title(f'expectation value of the energy of quantum sheet {sheet}')
    plt.xlabel('Time [ns]')
    plt.ylabel('Energy [J]')
    plt.legend()
    plt.show()


plt.plot([i*delta_t/10**(-9) for i in range(safe_points)], pos, label = [str(i) for i in range(n_sheets)])
plt.title('expectation value of the position')
#plt.xlabel(r'$$ \<X\> [m] $$')
plt.xlabel('X')
plt.ylabel('Energy [J]')
plt.legend()
plt.show()

plt.plot([i*delta_t/10**(-9) for i in range(safe_points)], mom, label = [str(i) for i in range(n_sheets)])
plt.title('expectation value of the momentum')
#plt.xlabel(r'$$ \<P\> [kg m/s] $$')
plt.xlabel('P')
plt.ylabel('Energy 1 [J]')
plt.legend()
plt.show()


for sheet in range(n_sheets):
    plt.imshow(psi_squared_cut[:,:,sheet])
    plt.title(f'Probability of wave in quantum sheet {sheet}')
    plt.xlabel('Time [iteration]')
    plt.ylabel('y-position [m/delta y]')
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
print(np.shape(ey))

animation_speed = 2500//safe_frequency

fig, ax = plt.subplots()
ax.set_xlabel('x position [m]')
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.set_ylabel('ey [V/m]')
    ax.plot(x_axis, ey[:,int(i*animation_speed)], c = 'black')
    for sheet in range(n_sheets):
        ax.axvline(x = x_place_qd[sheet], c = 'red', label = f'quantum dot {sheet}')
    ax.set_title(f'n = {int(i*animation_speed)}')

    #ax2 = ax.twinx()
    #ax2.set_ylabel('y')
    #ax2.scatter([x_place_qd for i in range(len(y_axis))], y_axis, c = psi_r[:,int(i*animation_speed)]**2 + psi_im[:,int(i*animation_speed)]**2)

anim = FuncAnimation(fig, animate)
plt.show()


animation_speed = 2500//safe_frequency

fig, ax = plt.subplots()
ax.set_xlabel('y-position [m]')
ax.set_ylabel('Probability')
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.plot(y_axis, psi_r[:,int(i*animation_speed)]**2 + psi_im[:,int(i*animation_speed)]**2, c = 'black', label = [str(i) for i in range(n_sheets)])
    ax.set_title(f'n = {int(i*animation_speed)}')
anim = FuncAnimation(fig, animate)
plt.show()