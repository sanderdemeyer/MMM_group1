import numpy as np
from scipy.sparse import csr_matrix
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