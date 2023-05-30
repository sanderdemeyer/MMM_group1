import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
"""
print(d)
print(np.roll(d, 1, 0))
print(np.roll(d, -1, 0))
print(np.roll(d, 1, 1))
print(np.roll(d, -1, 1))

print(np.multiply(d, np.roll(d, 1, 0)))
"""
#1: laatste naar eerste (voor i-1)
#-1: eerste naar laatste (voor i+1)

#0: behoud rijen, verander kolommen (voor i)
#1: behoud kolommen, verander rijen (voor j)


print(np.eye(5, 5, -2))

print(a)



print(np.array([1+1j, 2j, 3]).imag)

print((1 + 1j).imag)
print(1j * 1j)

x = np.linspace(-10, 10, 1000)
plt.plot(x, x/2*(1+x/np.sqrt(x**2 + 0.01**2)))
plt.show()


t0 = 200000
sigma_ramping = 90000

x = np.linspace(-10, 10, 1000)
plt.plot(x, np.exp(-x**2))
plt.plot(x, (1+np.tanh(x))/2)
plt.show()



x = np.linspace(0, 600000, 10**6)
plt.plot(x, np.tanh((x-t0)/sigma_ramping))
plt.show()



print(a)

n_t = 20
y_axis = np.array([0, 1, 2, 3, 4, 5, 6, 7])

delta_y_matrix = np.transpose(np.array([y_axis for i in range(n_t)]))
print(delta_y_matrix)



"""
print(np.linspace(0, 10, 10))
n = 10
delta = 5
y_axis = np.linspace(0,0 + n*delta,n+1)
print(y_axis)
"""
a = np.array([0, 1, 2, 3, 4, 5])
print(a)
print(np.roll(a,1)) # voor i-1
print(np.roll(a,-1)) # voor i+1


A = np.zeros((5, 5))
A[1, 2] = 5
A[3, 4] = 1
A[1, 1] = -2
print(A)
B = A
A = csr_matrix(A)
A.eliminate_zeros()

y = np.array([1, 2, 3, 4, 5])

print(B)
print(y)
print(B*y)
print(B@y)
