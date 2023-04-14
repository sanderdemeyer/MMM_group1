import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as special
from matplotlib.pyplot import pcolormesh
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

import pickle

epsilon_0 = 8.85*10**(-12)  # in units F/m
mu_0 = 1.25663706*10**(-6) # in units N/A^2
c = 3*10**8 # in units m/s

def hankel(x, f, J0=1):
    omega = 2*np.pi*f
    return -(J0*omega*mu_0/4)*special.hankel2(0, (omega*x/c))


with open('latest_test.pkl', 'rb') as f:
    [frequencies, iterations, fft_transform, fft_transform_source, x_source, y_source, delta_t, delta_x, delta_y, dist] = pickle.load(f)


print(frequencies)
print(fft_transform)
"""
plt.plot(2*np.pi*frequencies[1:iterations//2], abs(np.divide(fft_transform[1:iterations//2], fft_transform_source[1:iterations//2])), label = 'computational')
plt.show()
plt.plot(2*np.pi*frequencies[:iterations//2], delta_t*delta_x[x_source]*delta_y[x_source]*np.array([abs(hankel(dist, omega)) for omega in frequencies[:iterations//2]]), label = 'analytical')
plt.legend()
plt.show()
"""

plt.plot(frequencies[:iterations//2], fft_transform[:iterations//2])
plt.title('ez')
plt.xlabel(r'$ \omega $ [Hz]')
plt.ylabel('Frequency respons')
plt.title(f'Frequency respons at distance {dist} m from the source')

plt.show()

plt.plot(frequencies[:iterations//2], fft_transform_source[:iterations//2])
plt.title('source')
plt.show()

#dist = np.sqrt(delta_x[x_source](observation_point[0]-x_source)**2 + delta_y[y_source](observation_point[1]-y_source)**2)

lijst1 = abs(np.divide(fft_transform[1:iterations//2], fft_transform_source[1:iterations//2]))
lijst2 = delta_t*delta_x[x_source]*delta_y[x_source]*np.array([abs(hankel(dist, omega)) for omega in frequencies[:iterations//2]])
print(lijst1)
print(lijst2)
plt.plot(2*np.pi*frequencies[:iterations//2], abs(np.divide(fft_transform[:iterations//2], fft_transform_source[:iterations//2])), label = 'computational')
plt.plot(2*np.pi*frequencies[:iterations//2], delta_t*delta_x[x_source]*delta_y[x_source]*np.array([abs(hankel(dist, omega)) for omega in frequencies[:iterations//2]]), label = 'analytical')
plt.legend()
#plt.xlim(0, 6*10**9)
#plt.ylim(top = 3*10**(-9))
plt.show()

print(a)

plt.plot(2*np.pi*frequencies[1:iterations//2], fft_transform[1:iterations//2], label = 'computational')
plt.show()
plt.plot(2*np.pi*frequencies[1:iterations//2], fft_transform_source[1:iterations//2], label = 'computational')
plt.show()


plt.plot(2*np.pi*frequencies[1:iterations//2], (np.divide(abs(fft_transform[1:iterations//2]), abs(fft_transform_source[1:iterations//2]))), label = 'computational')
plt.plot(2*np.pi*frequencies[:iterations//2], 2*delta_t*delta_x[x_source]*delta_y[x_source]*np.array([abs(hankel(dist, 2*np.pi*freq)) for freq in frequencies[:iterations//2]]), label = 'analytical')
plt.legend()
plt.show()


comp = np.array([abs(hankel(dist, omega)) for omega in frequencies[1:iterations//2]])
ana = abs(np.divide(fft_transform[1:iterations//2], fft_transform_source[1:iterations//2]))

print(np.divide(comp, ana))

x_lijst = [x/100 for x in range(400)]
plt.plot(x_lijst, [x*special.hankel2(0,x) for x in x_lijst])
plt.show()