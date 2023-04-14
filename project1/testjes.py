import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
import numpy.fft as fft

import pickle


with open('testje.pkl', 'rb') as f:
    [A, M_U] = pickle.load(f)
print(M_U)
print(A)




print(a)

M_U = 3

A = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
print(A)

A[[1, M_U-1]] = A[[M_U-1, 1]]
A[:,[1, M_U-1]] = A[:,[M_U-1, 1]]

print(A)






print(a)


with open('testje.pkl', 'wb') as f:
    pickle.dump([A, M_U], f)



print(a)

lijst = fft.fftfreq(60, 10**(-9))
print(len(lijst))
print(lijst)


print(a)

plt.plot([special.hankel2(0,x) for x in range(100)])
plt.show()

print(a)

omega = 10**6
delta_t = 10**(-7)
tijds = [1] + [0 for n in range(150)]

print(tijds)
print(fft.fft(tijds))


print(a)

d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

dflat = d.flatten()/8
print((dflat))

print(np.transpose([dflat, 1-dflat]))


assert 0 == 1

print(d)
print(np.roll(d, 1, 0))
print(np.roll(d, -1, 0))
print(np.roll(d, 1, 1))
print(np.roll(d, -1, 1))

print(np.multiply(d, np.roll(d, 1, 0)))
#1: laatste naar eerste (voor i-1)
#-1: eerste naar laatste (voor i+1)

#0: behoud rijen, verander kolommen (voor i)
#1: behoud kolommen, verander rijen (voor j)


print(np.repeat(np.array([1, 2, 3, 4, 5]), 7, axis=0))

M = 3
N = 3

zz = np.array([1, 2, 3])
print(np.array([zz for i in range(10)]))


print(np.repeat(5, 7))

print(np.array([np.repeat(zz[i], M) for i in range(N)]))


print(d*7)


A = np.zeros((5, 4))
print(A[4, 2])
print(A[(4, 2)])


A = np.array([[0, 1], [2, 3]])
print(A[(0, 1)])

print(([(2, 2)]))

print('fdjk' == 'fdsjqklm')

def hankel(x):
    return special.hankel2(0, x)

d = np.linspace(0, 5*10**(1), 1000)
mu = 1
c = 1
w = 1

#plt.plot(d, -w*mu/4*hankel((w/c)*d))
#plt.show()


#print(fft.fft([1 for i in range(100)]))

#%%

epsilon_0 = 8.85*10**(-12)
mu_0 = 1.25663706*10**(-6)
c = 3*10**8
M = 100
N = 100

iterations = 80

#source should be either 'sine', 'gaussian_modulated', or 'gaussian'


# last 'extra' element should be the same as the 0th.
epsilon = np.ones((M,N))*epsilon_0
mu = np.ones((M,N))*mu_0
sigma = np.ones((M,N))*0

delta_x = np.ones(M)*10**(-1)
delta_y = np.ones(N)*10**(-1)
delta_x_matrix = np.array([np.repeat(delta_x[i], N) for i in range(M)])
delta_y_matrix = np.array([delta_y for i in range(M)])
delta_t = 10**(-11)

courant_number = 1
delta_t = np.max(delta_y)/(c)*courant_number
print(delta_t)

source = 'sine'
J0 = 1
tc = 5
sigma_source = 1
period = 10
omega_c = (2*np.pi)/(period*delta_t) # to have a period of 10 time steps

jz = np.zeros((M, N, iterations))

for n in range(iterations):
    for i in range(M):
        for j in range(N):
            jz[i, j, n] = J0*np.sin(omega_c*n*delta_t)*np.exp(-(i**2 + j**2)/(2*sigma_source**2))

fft_nd = fft.fftn(jz, axes = [2])

plt.plot([i for i in range(80)], fft_nd[0,0,:], label = '2')
plt.show()


# inverse

four = [J0 for i in range(iterations)]
inverse = fft.ifft(four)

print(inverse)
plt.plot([i for i in range(80)], inverse, label = '2')
plt.show()


print(fft.fft([1] + [0 for i in range(iterations-1)]))



a = np.array([[1, 2], [3, 4]])
print(a)
print(np.divide(1, a))





