import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity as sparse_identity
import time
from scipy.optimize import curve_fit

def potential_diag():
    V = np.zeros(n_y)
    for i in range(n_y):
        y = y_start + i*delta_y
        V[i] = (m/2)*(omega_HO**2)*(y**2)
    return V

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
f = 2 # factor by which the 'normal' width of the gaussian pulse is normalized
Eg_0 = 5*10**6*np.sqrt(f) # Amplitude of the PW-pulse if it is Gaussian
Es_0 = 1*10**5 # Amplitude of the PW-pulse if it is a sine wave
sigma_t = 10*10**(-15)*f # Width of the gaussian pulse
sigma_ramping = 30*10**(-15)
t0 = 20*10**(-15)*15 # Center of the gaussian pulse

alpha = 1 #should be between 0.9 and 1.1
omega_EM = alpha*omega_HO
#omega_EM = 7.5398*10**(15) # This makes the period equal to 500 time steps
delta_x = 1*10**(-6) # grid size in the x-direction in meter
delta_y = 0.5*10**(-9) # grid size in the y-direction in meter
n_y = int(L_y/delta_y) + 1 # Number of y grid cells
n_x = int(L_x/delta_x) + 1 # Number of x grid cells
t_sim = 10**(-12)*2.5 # Total simulated time in seconds
#provide location of structure through boundary of y-domain
y_start = -L_y/2
Courant = 1 # Courant number
delta_t_EM = 1/c*Courant*delta_x # time step based on the Courant number. With the current definitions, this means that delta_t = 1.6667*10**(-18)
Vmax = np.max(potential_diag())
delta_t_QM = 2*Courant/(8/3*hbar/m/delta_y**2 + Vmax/hbar)
delta_t = min(delta_t_EM, delta_t_QM)

n_t = int(t_sim/delta_t) # Total number of time steps
y_axis = np.linspace(y_start,y_start + (n_y-1)*delta_y,n_y)
x_axis = np.linspace(0, (n_x-1)*delta_x,n_x)

n_sheets = 1 # The number of wanted sheets of quantum dots


t0_gauss = t0//delta_t # Convert t0 to iteration number
sigma_gauss = sigma_t//delta_t # Convert sigma_gauss to iteration number

safe_frequency = 1 # All values are saved after this amount of time steps (except Norm, this is saved at all time steps)
safe_points = n_t//safe_frequency+1 # Denotes how many times the variables will be saved.

source = 'sine' # Type of EM wave. This should be either 'gaussian', 'sine', or 'None'
source_location = int(5*n_x/12) # Position of the source in the x-direction


x_place_qd = [(sheet+1)*L_x/(n_sheets+1) for sheet in range(n_sheets)] # place of the quantum dots. Should have length == n_sheets
#x_place_qd = [450*delta_x, 500*delta_x, 550*delta_x]

x_qd = [int(x_place_qd[sheet]/delta_x) for sheet in range(n_sheets)] # y-coordinate of the quantum dots. Should have length == n_sheets
#x_qd = [450, 500, 550] # y-coordinate of the quantum dots

starting_n_point = [(n_y//4 + sheet*n_y//24)*0 for sheet in range(n_sheets)] # starting points of the quantum dots. Should have length == n_sheets
#starting_n_point = [0, n_y//4, 0]


print(f'Zero-point energy is {omega_HO*hbar/2}')

coupling = True # Whether there is EM --> QM coupling
back_coupling = False # Whether there is QM --> EM coupling. This can only be True if coupling == True
gauge = 'velocity' # The gauge that will be used. This should be either 'length' or 'gauge'

norm_every_step = False # Denotes whether the wavefunction is normalized at every time step. Default is False.

def update_matrix():
    A = np.zeros((n_y,n_y))
    for i in range(n_y):
        A[i,i] = -30
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


def get_H_int(a):
    H_int = np.zeros((n_y,n_y))
    for i in range(n_y):
        y = y_start + i*delta_y
        H_int[i,i] = q**2/(2*m)*a**2
        H_int[i,i+1] = 0
    H_int = csr_matrix(H_int)
    H_int.eliminate_zeros()
    return H_int


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
        A_plus = -hbar*delta_t/(24*m*delta_y**2)*A_vel + V*delta_t/hbar
        B_matrix = q*delta_t/(24*m*delta_y)*B_vel


    psi_r_new = np.zeros((n_y, n_sheets))
    psi_im_new = np.zeros((n_y, n_sheets))

    for sheet in range(n_sheets):
        psi_r_new[:,sheet] = (m*omega_HO/(constants.pi*hbar))**(1/4)*np.exp(-m*omega_HO/2/hbar*(y_axis - starting_n_point[sheet]*delta_y)**2)
        psi_im_new[:,sheet] = np.zeros(n_y)

    psi_r[:,0,:] = psi_r_new
    psi_im[:,0,:] = psi_im_new
    Norm[0,:] = (np.sum(psi_r_new**2 + psi_im_new**2,0))*delta_y
    psi_squared_cut[:,0] = (psi_r_new)**2 + (psi_im_new)**2

    if gauge == 'velocity':
        a_squared_int = np.zeros(n_sheets)

    ey_new = np.zeros(n_x)
    hz_new = np.zeros(n_x)
    Jy = np.zeros(n_x)

    for i in range(1,n_t):
        ey_old = ey_new
        hz_old = hz_new
        psi_r_old = psi_r_new
        psi_im_old = psi_im_new

        hz_new = hz_old - delta_t/(mu*delta_x)*(ey_old - np.roll(ey_old, 1))

        Jy = np.zeros(n_x)
        if source == 'gaussian':
            if i > t0_gauss - 5*sigma_gauss and i < t0_gauss + 5*sigma_gauss:
                Jy[source_location] = -1110.6833660953741*np.sqrt(1/f)*Eg_0*np.exp(-(i-t0_gauss)**2/(2*sigma_gauss**2))
        elif source == 'sine':
            Jy[source_location] = -5308.993524411968*Es_0*np.sin(omega_EM*i*delta_t)*(1+np.tanh((i*delta_t-t0)/sigma_ramping))/2
        elif source == 'None':
            pass
        else:
            print('wrong source')

        if gauge == 'length':
            if coupling == False:
                for sheet in range(n_sheets):
                    psi_r_new[:,sheet] = psi_r_old[:,sheet] - hbar*delta_t*(A @ psi_im_old[:,sheet])/(24*m*(delta_y**2)) + delta_t*(V @ psi_im_old[:,sheet])/hbar
                    psi_im_new[:,sheet] = psi_im_old[:,sheet] + hbar*delta_t/24/(m*delta_y**2)*(A @ psi_r_new[:,sheet]) - delta_t/hbar*(V @ psi_r_new[:,sheet])
                ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy
            else:
                if back_coupling == False:
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
                        
                        j_q[x_qd[sheet]] = q*hbar*N*L_x_size_quantum_dot/(2*m*delta_x)*np.mean(psi_r_new[:,sheet]*np.roll(psi_im_new[:,sheet]+psi_im_old[:,sheet],-1) - np.roll(psi_r_new[:,sheet],-1)*(psi_im_new[:,sheet]+psi_im_old[:,sheet]))

                    ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy - delta_t/epsilon*j_q

        if gauge == 'velocity':
            if coupling == False:
                raise Exception("No coupling is not yet implemented in the velocity gauge. This can easily be done by specifying coupling = True, back_coupling = False, and source = 'None'.")
            else:
                if back_coupling == False:
                    ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*Jy

                    for sheet in range(n_sheets):
                        a[sheet] += -ey_old[x_qd[sheet]]*delta_t

                        # B_plus = sparse_identity(n_y) + a[sheet]*B_matrix
                        # B_min = sparse_identity(n_y) - a[sheet]*B_matrix
                        B_plus = sparse_identity(n_y) - a[sheet]*B_matrix
                        B_min = sparse_identity(n_y) + a[sheet]*B_matrix

                        psi_r_new[:,sheet] = spsolve(B_plus, A_plus @ psi_im_old[:,sheet] + B_min @ psi_r_old[:,sheet])
                        psi_im_new[:,sheet] = spsolve(B_plus, -A_plus @ psi_r_new[:,sheet] + B_min @ psi_im_old[:,sheet])
                else:
                    j_q = np.zeros(n_x)
                    for sheet in range(n_sheets):
                        #B_plus = sparse_identity(n_y) + q*a[sheet]*delta_t/(24*m*delta_y)*B_vel
                        #B_min = sparse_identity(n_y) - q*a[sheet]*delta_t/(24*m*delta_y)*B_vel
                        B_plus = sparse_identity(n_y) - q*a[sheet]*delta_t/(24*m*delta_y)*B_vel
                        B_min = sparse_identity(n_y) + q*a[sheet]*delta_t/(24*m*delta_y)*B_vel

                        psi_r_new[:,sheet] = spsolve(B_plus, A_plus @ psi_im_old[:,sheet] + B_min @ psi_r_old[:,sheet])
                        psi_im_new[:,sheet] = spsolve(B_plus, -A_plus @ psi_r_new[:,sheet] + B_min @ psi_im_old[:,sheet])
                        j_q[x_qd[sheet]] = q*hbar*N*L_x_size_quantum_dot/(2*m*delta_x)*np.mean(psi_r_new[:,sheet]*np.roll(psi_im_new[:,sheet]+psi_im_old[:,sheet],-1) - np.roll(psi_r_new[:,sheet],-1)*(psi_im_new[:,sheet]+psi_im_old[:,sheet]))
                        j_q[x_qd[sheet]] += -(q**2*a[sheet]*N/m)*(L_x_size_quantum_dot/delta_x)

                    ey_new = ey_old - delta_t/(epsilon*delta_x) * (np.roll(hz_new, -1) - hz_new) - (delta_t/epsilon)*(Jy+j_q)
                    for sheet in range(n_sheets):
                        a[sheet] += -ey_new[x_qd[sheet]]*delta_t
            a_squared_int += a**2*delta_t

        if norm_every_step:
            norm_new = np.sqrt(np.sum(psi_r_old**2+psi_im_old**2)*delta_y,0)
            psi_r_new = np.divide(psi_r_new,norm_new)
            psi_im_new = np.divide(psi_im_new,norm_new)
            
        S = c*delta_t/delta_x
        ey_new[0] = ey_old[1] + (1-S)/(1+S)*(ey_old[0]-ey_new[1])
        ey_new[-1] = ey_old[-2] + (1-S)/(1+S)*(ey_old[-1]-ey_new[-2])

        if i % safe_frequency == 0:
            ey[:,i//safe_frequency] = ey_new
            hz[:,i//safe_frequency] = hz_new
            psi_squared_cut[:,i//safe_frequency] = (psi_r_new)**2 + (psi_im_new)**2

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

        Norm[i,:] = (np.sum(psi_r_old**2 + psi_im_old**2,0))*delta_y

        if i%1000 == 0:
            print(f'Done iteration {i} of {n_t}')
    return psi_r,psi_im, Norm, ey, hz, psi_squared, psi_squared_cut, a_list


def expectation_value_position(psi_r, psi_im, y_axis):
    y_axis = np.transpose(np.array([[y_axis for t in range(safe_points)] for n in range(n_sheets)]))

    exp_pos = np.sum((psi_r**2 + psi_im**2) * y_axis * delta_y,0)
    return exp_pos

def expectation_value_momentum(psi_r, psi_im, a):
    exp_mom = np.sum(-1j*hbar*(psi_r - 1j*psi_im)*(np.roll(psi_r, -1,0) - psi_r + 1j*(np.roll(psi_im, -1,0) - psi_im)),0)
    exp_mom_2 = q*a
    return exp_mom + exp_mom_2

def expectation_value_kinetic_energy(psi_r, psi_im, a):
    exp_kin = -hbar**2/(2*m)*np.sum((psi_r - 1j*psi_im)*(np.roll(psi_r,1,0)+np.roll(psi_r,-1,0)-2*psi_r + 1j*(np.roll(psi_im,1,0)+np.roll(psi_im,-1,0)-2*psi_im)),0)/delta_y
    exp_kin_2 = -1j*hbar*q/m*a*np.sum((psi_r-1j*psi_im)*(psi_r+1j*psi_im - np.roll(psi_r+1j*psi_im, 1, 0)),0)
    exp_kin_3 = q**2/(2*m)*a**2

    return exp_kin + exp_kin_2 + exp_kin_3

def expectation_value_potential_energy(psi_r, psi_im):
    V = potential_diag()
    V = np.transpose(np.array([[V for t in range(safe_points)] for n in range(n_sheets)]))
    exp_pot = np.sum(((psi_r)**2 + (psi_im)**2)*V,0)*delta_y
    return exp_pot

def expectation_value_energy(psi_r, psi_im):
    exp_energy = np.sum(1j*hbar*(psi_r - 1j*psi_im)*(psi_r - np.roll(psi_r,1,1) + 1j*(psi_im - np.roll(psi_im,1,1))),0)*delta_y/(delta_t*safe_frequency)
    return exp_energy[1:-1]

def check_continuity_final(psi_r, psi_im):
    A, B = update_matrix_velocity()
    norm_list = np.zeros((n_t, n_y))
    for i in range(0,n_t): # The result should not be interpreted for i = 0

        psi_r_new = psi_r[:,i+1][:,0] # timestep i+3/2
        psi_im_new = psi_im[:,i][:,0] # timestep i+1
        psi_r_old = psi_r[:,i][:,0] # timestep i+1/2
        psi_im_old = psi_im[:,i-1][:,0] # timestep i

        psi_r_middle = (psi_r_new+psi_r_old)/2 # timestep i+1
        psi_im_middle = (psi_im_new+psi_im_old)/2 # timestep i+1/2

        psi_norm = (psi_r_middle**2 + psi_im_new**2) # at timestep i+1
        norm_list[i,:] = psi_norm

        J = (hbar/m)*(psi_r_old*(B @ psi_im_middle)-psi_im_middle*(B @ psi_r_old))/(12*delta_y)
        # the above was evaluated at timestep i+1/2, thus using psi_r_old and psi_im_middle

        dPdt = (psi_norm - norm_list[i-1,:])/(safe_frequency*delta_t) # evaluated at timestep i+1/2
        grad_J = (B @ J)/(12*delta_y) # Gradient of J
        results_cont = dPdt+grad_J
        rel_error = np.divide(results_cont, dPdt)
        plt.plot(y_axis[7:-7], rel_error[7:-7])
        plt.xlabel('y [m]', fontsize = 15)
        plt.ylabel('Relative error', fontsize = 15)
        plt.title('Checking of the continuity equation', fontsize = 15)
        plt.show()


psi_r,psi_im,norm,ey,hz,psi_squared,psi_squared_cut, a_list = run(coupling)

plt.plot([i*delta_t/10**(-15) for i in range(n_t)], norm, label = ['quantum sheet' + str(i) for i in range(n_sheets)])
plt.title('Norm of the wave functions')
plt.xlabel('Time [fs]')
plt.ylabel('Norm')
plt.legend()
plt.show()

#check_continuity_final(psi_r, psi_im)



print('started with energy')
exp_energy =  expectation_value_energy(psi_r, psi_im)
print('started with potential energy')
exp_pot =  expectation_value_potential_energy(psi_r, psi_im)
print('started with kinetic energy')
exp_kin =  expectation_value_kinetic_energy(psi_r, psi_im, a_list)

pos = expectation_value_position(psi_r, psi_im, y_axis)
mom = expectation_value_momentum(psi_r, psi_im, a_list)

for sheet in range(n_sheets):
    plt.plot([i*safe_frequency*delta_t/10**(-15) for i in range(safe_points-2)], exp_energy[:,sheet], label = 'energy')
    plt.plot([i*safe_frequency*delta_t/10**(-15) for i in range(safe_points)], exp_pot[:,sheet], label = 'potential energy')
    plt.plot([i*safe_frequency*delta_t/10**(-15) for i in range(safe_points)], exp_kin[:,sheet], label = 'kinetic energy')
    plt.title(f'expectation value of the energy of quantum sheet {sheet}', fontsize = 15)
    plt.xlabel('Time [fs]', fontsize = 15)
    plt.ylabel('Energy [J]', fontsize = 15)
    plt.legend()
    plt.show()


"""
# Fitting of the exponential

def exponential(x, y0, a, b):
    return y0 + a*np.exp(-b*x)

curve_fit_x = [i*safe_frequency*delta_t/10**(-15) for i in range(safe_points-2)]
params, cov = curve_fit(exponential, curve_fit_x, exp_energy[:,0], p0 = [5.7*10**(-22), 4.35*10**(-21), 6.3*10**(-4)])
print(f'params are {params}')
print(f'cov is {cov}')

for sheet in range(n_sheets):
    plt.plot([i*safe_frequency*delta_t/10**(-15) for i in range(safe_points-2)], exp_energy[:,sheet], label = 'energy')
    plt.plot([i*safe_frequency*delta_t/10**(-15) for i in range(safe_points)], exp_pot[:,sheet], label = 'potential energy')
    plt.plot([i*safe_frequency*delta_t/10**(-15) for i in range(safe_points)], exp_kin[:,sheet], label = 'kinetic energy')
    plt.plot(curve_fit_x, [exponential(x, params[0], params[1], params[2]) for x in curve_fit_x], label = 'fitted energy')
    plt.title(f'expectation value of the energy of quantum sheet {sheet}')
    plt.xlabel('Time [fs]')
    plt.ylabel('Energy [J]')
    plt.legend()
    plt.show()

"""



plt.plot([i*safe_frequency*delta_t/10**(-15) for i in range(safe_points)], pos, label = ['quantum sheet ' + str(i) for i in range(n_sheets)])
plt.title('expectation value of the position')
plt.xlabel('Time [fs]')
plt.ylabel('X [m]')
plt.legend()
plt.show()

plt.plot([i*safe_frequency*delta_t/10**(-15) for i in range(safe_points)], mom, label = ['quantum sheet ' + str(i) for i in range(n_sheets)])
plt.title('expectation value of the momentum')
plt.xlabel('Time [fs]')
plt.ylabel('P [kg m/s]')
plt.legend()
plt.show()


for sheet in range(n_sheets):
    plt.imshow(psi_squared_cut[:,:,sheet], extent=[0,int(t_sim*10**(15)),int(L_y*10**(9)),0], aspect='auto')
    plt.title(f'Probability density of wave in quantum sheet {sheet}', fontsize = 15)
    plt.xlabel('Time [fs]', fontsize = 15)
    plt.ylabel('y-position [nm]', fontsize = 15)
    plt.show()


animation_speed = 2500//safe_frequency # speed of the animation

fig, ax = plt.subplots()
ax.set_xlabel('x position [m]', fontsize = 15)
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.set_ylabel('ey [V/m]', fontsize = 15)
    ax.plot(x_axis, ey[:,int(i*animation_speed)], c = 'black')
    for sheet in range(n_sheets):
        ax.axvline(x = x_place_qd[sheet], c = 'red', label = f'quantum dot {sheet}')
    ax.set_title(f'timestep = {int(i*animation_speed*safe_frequency)}. Time = {round(delta_t/10**(-15)*i*animation_speed*safe_frequency,3)} fs', fontsize = 15)
    ax.set_xlabel('x-position [m]', fontsize = 15)
    ax.set_ylabel('ey field [V/m]', fontsize = 15)
    plt.legend()

anim = FuncAnimation(fig, animate)
plt.show()


animation_speed = 2500//safe_frequency # speed of the animation

fig, ax = plt.subplots()
#ax.set_aspect('equal', adjustable='box')
def animate(i):
    ax.clear()
    ax.plot(y_axis, psi_r[:,int(i*animation_speed)]**2 + psi_im[:,int(i*animation_speed)]**2, label = ['quantum sheet ' + str(i) for i in range(n_sheets)])
    ax.set_title(f'timestep = {int(i*animation_speed*safe_frequency)}. Time = {round(delta_t/10**(-15)*i*animation_speed*safe_frequency,3)} fs')
    ax.set_xlabel('y-position [m]')
    ax.set_ylabel('Probability density')
    plt.legend()
anim = FuncAnimation(fig, animate)
plt.show()