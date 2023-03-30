import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.concatenate((a, b))



d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

print(d)
print(np.roll(d, 1, 0))
print(np.roll(d, -1, 0))
print(np.roll(d, 1, 1))
print(np.roll(d, -1, 1))

print(np.multiply(d, np.roll(d, 1, 0)))
#1: laatste naar eerste
#-1: eerste naar laatste

#0: behoud rijen, switch kolommen
#1: behoud kolommen, switch rijen


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