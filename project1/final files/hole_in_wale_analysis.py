import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('transmissions.pkl', 'rb') as f:
    [transmission_list, transmission_list, alpha_list, observation_points_ez_t, observation_points_ez_b, ez_b_list_observe, ez_t_list_observe] = pickle.load(f)

maxima_points = [220, 226, 230, 240, 247, 255, 260, 265, 270, 278, 283, 290, 300, 305, 313, 318, 320, 325, 333, 349, 349, 349, 349, 349, 349, 349]

print(len(alpha_list))
print(len(observation_points_ez_t))
print(len(maxima_points))
transmission_list_new = []

transmission_base = np.max(np.abs(ez_b_list_observe[:40,0]))

print(transmission_base)

iterations = 350
for i, (maxima, point) in enumerate(zip(maxima_points, observation_points_ez_t)):
    plt.plot(range(iterations), ez_t_list_observe[:,i])
    plt.xlabel('Iteration')
    plt.ylabel('Ez')
    plt.title(f'Ez at {point} in top UCHIE region')
    #plt.show()

    transmission_list_new.append(np.max(np.abs(ez_t_list_observe[:maxima,i])))
plt.show()

print(transmission_list_new)

plt.plot(np.array(alpha_list)*180/np.pi, np.array(transmission_list_new)/transmission_base)
plt.xlabel(r'$ \alpha $ [degrees]')
plt.ylabel('Relative magnitude of the peak')
plt.title('Transmission through a hole in a PEC')
plt.show()