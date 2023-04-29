import numpy as np
from scipy.sparse import csr_matrix

"""
print(np.linspace(0, 10, 10))
n = 10
delta = 5
y_axis = np.linspace(0,0 + n*delta,n+1)
print(y_axis)
"""


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
