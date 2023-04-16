import numpy as np
from scipy.sparse import csr_matrix
import math

def def_jz(J0, source, M, N, x_point, y_point, iterations, delta_t, tc, sigma_source, period, delta_value):
    # Definition of the source
    jz = np.zeros((M, N, iterations))
    if J0 == 0:
        return jz
    elif source == 'gaussian_modulated_dirac':
        omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps
        for n in range(iterations):
            jz[x_point, y_point, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.sin(omega_c*n*delta_t)*delta_value
    elif source == 'gaussian_modulated':
        for n in range(iterations):
            for i in range(max(0, x_point - math.ceil(5*sigma_source)), min(M, x_point + math.ceil(5*sigma_source) + 1)):
                for j in range(max(0, y_point - math.ceil(5*sigma_source)), min(N, y_point + math.ceil(5*sigma_source) + 1)):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.exp(-((i-x_point)**2 + (j-y_point)**2)/(2*sigma_source**2))
    elif source == 'gaussian':
        delta_t = delta_value
        omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps
        for n in range(iterations):
            for i in range(x_point - math.ceil(5*sigma_source), x_point + math.ceil(5*sigma_source) + 1):
                for j in range(y_point - math.ceil(5*sigma_source), y_point + math.ceil(5*sigma_source) + 1):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.sin(omega_c*n*delta_t)*np.exp(-((i-x_point)**2 + (j-y_point)**2)/(2*sigma_source**2))
    elif source == 'sine':
        delta_t = delta_value
        omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps
        for n in range(iterations):
            for i in range(x_point - math.ceil(5*sigma_source), x_point + math.ceil(5*sigma_source) + 1):
                for j in range(y_point - math.ceil(5*sigma_source), y_point + math.ceil(5*sigma_source) + 1):
                    jz[i, j, n] = J0*np.sin(omega_c*n*delta_t)*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    elif source == 'dirac':
        jz[x_point, y_point, 0] = delta_value
    else:
        print('Invalid source name')
    return jz


def def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, M):
    # definition of the UCHIE update matrices.
    # PBC are implemented
    A = np.zeros((2*M, 2*M))
    B = np.zeros((2*M, 2*M))

    for i in range(M):

        if i != M-1:
            A[i,i+1] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,i+1] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) + sigma[i+1,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M+i+1] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) - sigma[i+1,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M+i+1] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])

        else:
            A[i,0] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M] = -mu[0,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,0] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M] = -mu[0,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,0] = (epsilon[0,0]/(2*delta_t) + sigma[0,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,0] = (epsilon[0,0]/(2*delta_t) - sigma[0,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])
    return [A, B]



def update_implicit(ez_old, hy_old, bx, n, A_inv, B, delta_t, delta_y_matrix, M, N, jz, mu):
    # code to implicitly update ez and hy in the UCHIE regions

    bx_term = -(delta_t/2)*np.divide(bx, np.multiply(mu, delta_y_matrix))
    C_term_base = np.roll(bx_term, -1, 0) + bx_term - np.roll(np.roll(bx_term, -1, 0), 1, 1) - np.roll(bx_term, 1, 1)
    C_term = np.concatenate((np.zeros((M, N)), C_term_base))
    # should be delta_y_star_matrix
    jz_n = -np.multiply(delta_y_matrix, jz[:,:,n])/4
    jz_nm1 = -np.multiply(delta_y_matrix, jz[:,:,n-1])/4
    D_term_base = np.roll(jz_n, -1, 0) + jz_n + np.roll(jz_nm1, -1, 0) + jz_nm1
    D_term = np.concatenate((np.zeros((M, N)), D_term_base))

    new_values = np.dot(A_inv, (np.dot(B, np.concatenate((ez_old, hy_old))) + C_term + D_term))
    ez_new = new_values[:M,:]
    hy_new = new_values[M:,:]
    return [new_values[:M,:], new_values[M:,:]]

def update_implicit_faster(ez_old, hy_old, bx, n, A_inv, A_invB, delta_t, delta_y_matrix, M, N, jz, mu):
    # code to implicitly update ez and hy in the UCHIE regions
    # this works marginally faster than update_implicit, as the matrix A^{-1} B is precomputed.

    bx_term = -(delta_t/2)*np.divide(bx, np.multiply(mu, delta_y_matrix))
    C_term_base = np.roll(bx_term, -1, 0) + bx_term - np.roll(np.roll(bx_term, -1, 0), 1, 1) - np.roll(bx_term, 1, 1)
    C_term = np.concatenate((np.zeros((M, N)), C_term_base))
    # should be delta_y_star_matrix
    jz_n = -np.multiply(delta_y_matrix, jz[:,:,n])/4
    jz_nm1 = -np.multiply(delta_y_matrix, jz[:,:,n-1])/4
    D_term_base = np.roll(jz_n, -1, 0) + jz_n + np.roll(jz_nm1, -1, 0) + jz_nm1
    D_term = np.concatenate((np.zeros((M, N)), D_term_base))

    new_values = np.dot(A_invB, np.concatenate((ez_old, hy_old))) + np.dot(A_inv, C_term + D_term)
    ez_new = new_values[:M,:]
    hy_new = new_values[M:,:]
    return [new_values[:M,:], new_values[M:,:]]


### The underlying code can be ignored, as they correspond to outdated functions.
"""
def update_implicit_hybrid_OLD(ez_old, hy_old, bx, n, A_inv, B, delta_t, delta_y_matrix, M, N, jz, mu, delta_x_Yee_left, delta_x_Yee_right, Hy_Yee_left, Hy_Yee_right, Ez_Yee_left, Ez_Yee_right):

    bx_term = -(delta_t/2)*np.divide(bx, np.multiply(mu, delta_y_matrix))
    C_term_base = np.roll(bx_term, -1, 0) + bx_term - np.roll(np.roll(bx_term, -1, 0), 1, 1) - np.roll(bx_term, 1, 1)
    C_term = np.concatenate((np.zeros((M, N)), C_term_base))
    # should be delta_y_star_matrix
    jz_n = -np.multiply(delta_y_matrix, jz[:,:,n])/4
    jz_nm1 = -np.multiply(delta_y_matrix, jz[:,:,n-1])/4
    D_term_base = np.roll(jz_n, -1, 0) + jz_n + np.roll(jz_nm1, -1, 0) + jz_nm1
    D_term = np.concatenate((np.zeros((M, N)), D_term_base))

    term1_left = - 1/(delta_x_Yee_left)*Ez_Yee_left
    term1_right = 1/(delta_x_Yee_right)*Ez_Yee_right

    bx_term_boundary = np.divide(bx, np.multiply(mu, delta_y_matrix))
    term2_left = -2/(delta_x_Yee_left)*Hy_Yee_left - jz[0,:,n]
    term2_left[1:] = term2_left[1:] - bx_term_boundary[0,1:] + bx_term_boundary[0,:-1]

    term2_right = 2/(delta_x_Yee_right)*Hy_Yee_right - jz[-1,:,n]
    term2_right[1:] = term2_right[1:] - bx_term_boundary[-1,1:] + bx_term_boundary[-1,:-1]

    C_term[0,:] = term1_left
    C_term[M-1,:] = term1_right
    C_term[M,:] = term2_left
    C_term[2*M-1,:] = term2_right

    D_term[0,:] = np.zeros(N)
    D_term[M-1,:] = np.zeros(N)
    D_term[M,:] = np.zeros(N)
    D_term[2*M-1,:] = np.zeros(N)

    new_values = np.dot(A_inv, (np.dot(B, np.concatenate((ez_old, hy_old))) + C_term + D_term))
    return [new_values[:M,:], new_values[M:,:]]

def update_implicit_hybrid_new_OLD(ez_old, hy_old, bx, n, A_inv, B, delta_t, delta_y_matrix, M, N, jz, mu, delta_x_Yee_left, delta_x_Yee_right, Ez_left_new, Ez_left_old, Ez_right_new, Ez_right_old, Hy_left, Hy_right):

    bx_term = -(delta_t/2)*np.divide(bx, np.multiply(mu, delta_y_matrix))
    C_term_base = np.roll(bx_term, -1, 0) + bx_term - np.roll(np.roll(bx_term, -1, 0), 1, 1) - np.roll(bx_term, 1, 1)
    C_term = np.concatenate((np.zeros((M, N)), C_term_base))
    # should be delta_y_star_matrix
    jz_n = -np.multiply(delta_y_matrix, jz[:,:,n])/4
    jz_nm1 = -np.multiply(delta_y_matrix, jz[:,:,n-1])/4
    D_term_base = np.roll(jz_n, -1, 0) + jz_n + np.roll(jz_nm1, -1, 0) + jz_nm1
    D_term = np.concatenate((np.zeros((M, N)), D_term_base))

    tot_term = C_term + D_term

    tot_term[0,:] = 1/(2*delta_x_Yee_left)*(Ez_left_new + Ez_left_old)
    tot_term[M-1,:] = 1/(2*delta_x_Yee_right)*(Ez_right_new + Ez_right_old)

    bx_term = np.divide(bx, np.multiply(mu, delta_y_matrix))
    tot_term[M,:] = -2/(delta_x_Yee_left)*Hy_left - bx_term[0,:] + np.roll(bx_term, 1, 1)[0,:] - jz[0,:,n]
    tot_term[2*M-1,:] = 2/(delta_x_Yee_right)*Hy_right - bx_term[-1,:] + np.roll(bx_term, 1, 1)[-1,:] - jz[-1,:,n]

    new_values = np.dot(A_inv, (np.dot(B, np.concatenate((ez_old, hy_old))) + tot_term))
    return [new_values[:M,:], new_values[M:,:]]

def def_update_matrices_sparse_OLD(epsilon, mu, sigma, delta_x, delta_y, delta_t, M):
    A = csr_matrix((2*M, 2*M))
    B = csr_matrix((2*M, 2*M))

    for i in range(M):

        if i != M-1:
            A[i,i+1] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,i+1] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) + sigma[i+1,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M+i+1] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) - sigma[i+1,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M+i+1] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])

        else:
            A[i,0] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M] = -mu[0,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,0] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M] = -mu[0,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,0] = (epsilon[0,0]/(2*delta_t) + sigma[0,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,0] = (epsilon[0,0]/(2*delta_t) - sigma[0,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])
    return [A, B]


def def_update_matrices_new_OLD(epsilon, mu, sigma, delta_x, delta_y, delta_t, M):
    A = np.zeros((2*M, 2*M))
    B = np.zeros((2*M, 2*M))

    for i in range(M):

        if i != M-1:
            A[i,i+1] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,i+1] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) + sigma[i+1,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M+i+1] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) - sigma[i+1,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M+i+1] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])

        else:
            A[i,0] = 0
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M] = 0
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,0] = 0
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M] = 0
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,0] = 0
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M] = 0
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,0] = 0
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M] = 0
            B[M+i,M+i] = -1/(2*delta_x[i])
    return [A, B]

def update_implicit_hybrid_zeros_OLD(ez_old, hy_old, bx, n, A_inv, B, delta_t, delta_y_matrix, M, N, jz, mu, delta_x_Yee_left, delta_x_Yee_right, Ez_left_new, Ez_left_old, Ez_right_new, Ez_right_old, Hy_left, Hy_right):

    bx_term = -(delta_t/2)*np.divide(bx, np.multiply(mu, delta_y_matrix))
    C_term_base = np.roll(bx_term, -1, 0) + bx_term - np.roll(np.roll(bx_term, -1, 0), 1, 1) - np.roll(bx_term, 1, 1)
    C_term = np.concatenate((np.zeros((M, N)), C_term_base))
    # should be delta_y_star_matrix
    jz_n = -np.multiply(delta_y_matrix, jz[:,:,n])/4
    jz_nm1 = -np.multiply(delta_y_matrix, jz[:,:,n-1])/4
    D_term_base = np.roll(jz_n, -1, 0) + jz_n + np.roll(jz_nm1, -1, 0) + jz_nm1
    D_term = np.concatenate((np.zeros((M, N)), D_term_base))


    new_values = np.dot(A_inv, (np.dot(B, np.concatenate((ez_old, hy_old))) + C_term + D_term))

    values_left_old = np.concatenate((ez_old[0,:], hy_old[0,:]))



    ez_new = new_values[:M,:]
    hy_new = new_values[M:,:]
    return [new_values[:M,:], new_values[M:,:]]

def def_update_matrices_hybrid_new_OLD(epsilon, mu, sigma, delta_x, delta_y, delta_t, M, delta_x_Yee_left, delta_x_Yee_right):
    A = np.zeros((2*M, 2*M))
    B = np.zeros((2*M, 2*M))

    for i in range(M):

        if i != M-1 and i != 0:
            A[i,i+1] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,i+1] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) + sigma[i+1,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M+i+1] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) - sigma[i+1,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M+i+1] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])

        elif i == 0:
            A[i,i] = 1/(2*delta_x_Yee_left)
            A[i,M+i] = -mu[i,0]/delta_t

            B[i,i] = -1/(2*delta_x_Yee_left)
            B[i,M+i] = -mu[i,0]/delta_t

            A[M+i,i] = epsilon[i,0]/delta_t + sigma[i,0]/2
            A[M+i,M+i] = -1/delta_x_Yee_left #sign?

            B[M+i,i] = epsilon[i,0]/delta_t - sigma[i,0]/2
            B[M+i,M+i] = 1/delta_x_Yee_left # sign?
        elif i == M-1:
            A[i,i] = 1/(2*delta_x_Yee_right)
            A[i,M+i] = mu[i,0]/delta_t

            B[i,i] = -1/(2*delta_x_Yee_right)
            B[i,M+i] = mu[i,0]/delta_t

            A[M+i,i] = epsilon[i,0]/delta_t + sigma[i,0]/2
            A[M+i,M+i] = 1/delta_x_Yee_right

            B[M+i,i] = epsilon[i,0]/delta_t - sigma[i,0]/2
            B[M+i,M+i] = -1/delta_x_Yee_right

        else:
            print('something went terribly wrong')
    return [A, B]

def def_update_matrices_hybrid_OLD(epsilon, mu, sigma, delta_x, delta_y, delta_t, M, delta_x_Yee_left, delta_x_Yee_right):
    A = np.zeros((2*M, 2*M))
    B = np.zeros((2*M, 2*M))

    for i in range(M):
        if i != M-1 and i != 0:
            A[i,i+1] = delta_y[0]/(2*delta_x[i])
            A[i, i] = -delta_y[0]/(2*delta_x[i])
            A[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            A[i,M+i] = -mu[i,0]/(2*delta_t)

            B[i,i+1] = -delta_y[0]/(2*delta_x[i])
            B[i,i] = delta_y[0]/(2*delta_x[i])
            B[i,M+i+1] = -mu[i+1,0]/(2*delta_t)
            B[i,M+i] = -mu[i,0]/(2*delta_t)

            # delta_y should be delta_y*
            A[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) + sigma[i+1,0]/4)*delta_y[0]
            A[M+i,i] = (epsilon[i,0]/(2*delta_t) + sigma[i,0]/4)*delta_y[0]
            A[M+i,M+i+1] = -1/(2*delta_x[i])
            A[M+i,M+i] = 1/(2*delta_x[i])

            B[M+i,i+1] = (epsilon[i+1,0]/(2*delta_t) - sigma[i+1,0]/4)*delta_y[0]
            B[M+i,i] = (epsilon[i,0]/(2*delta_t) - sigma[i,0]/4)*delta_y[0]
            B[M+i,M+i+1] = 1/(2*delta_x[i])
            B[M+i,M+i] = -1/(2*delta_x[i])

        elif i == M-1:
            A[i,M+i] = mu[i,0]/delta_t
            B[i,i] = -1/delta_x_Yee_right
            B[i,M+i] = mu[i,0]/delta_t

            # delta_y should be delta_y*
            A[M+i,i] = (epsilon[i,0]/delta_t - sigma[i,0]/2)#*delta_y[0]
            A[M+i,M+i] = 1/delta_x_Yee_right

            B[M+i,i] = (epsilon[i,0]/delta_t + sigma[i,0]/2)#*delta_y[0]
            B[M+i,M+i] = -1/delta_x_Yee_right
        elif i == 0:
            A[i,M+i] = mu[i,0]/delta_t
            B[i,i] = 1/delta_x_Yee_left
            B[i,M+i] = mu[i,0]/delta_t

            # delta_y should be delta_y*
            A[M+i,i] = (epsilon[i,0]/delta_t + sigma[i,0]/2)#*delta_y[0]
            A[M+i,M+i] = -1/delta_x_Yee_left

            B[M+i,i] = (epsilon[i,0]/delta_t - sigma[i,0]/2)#*delta_y[0]
            B[M+i,M+i] = 1/delta_x_Yee_left

        else:
            print('something went terribly wrong!')
    return [A, B]

"""