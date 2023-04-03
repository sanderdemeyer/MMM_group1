import numpy as np

def def_jz(source, M, N, iterations, delta_value):
    J0 = 1
    jz = np.zeros((M, N, iterations))
    if source == 0:
        return jz
    elif source == 'gaussian_modulated':
        tc = 5
        sigma_source = 1
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    elif source == 'gaussian':
        delta_t = delta_value
        tc = 5
        sigma_source = 1
        period = 10
        omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.exp(-(n-tc)**2/(2*sigma_source**2))*np.sin(omega_c*n*delta_t)*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    elif source == 'sine':
        delta_t = delta_value
        tc = 5
        sigma_source = 1
        period = 10
        omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps
        for n in range(iterations):
            for i in range(M):
                for j in range(N):
                    jz[i, j, n] = J0*np.sin(omega_c*n*delta_t)*np.exp(-(i**2 + j**2)/(2*sigma_source**2))
    elif source == 'dirac':
        jz[M//3, N//4, 0] = delta_value
    return jz

def def_update_matrices(epsilon, mu, sigma, delta_x, delta_y, delta_t, M):
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
