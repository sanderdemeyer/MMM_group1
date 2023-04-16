import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('transmissions_new.pkl', 'rb') as f:
    [alpha_list, observation_points_ez_Yee, ez_Yee_list_observe] = pickle.load(f)

maxima_points = [220, 226, 230, 240, 247, 255, 260, 265, 270, 278, 283, 290, 300, 305, 313, 318, 320, 325, 333, 349, 349, 349, 349, 349, 349, 349]
maxima_points = [308, 312, 310, 320, 313, 315, 313, 320, 320, 325, 330, 330, 335, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349, 349]
maxima_ponit = [490 for i in range(len(alpha_list))]
transmission_list_new = []

transmission_base = np.max(np.abs(ez_Yee_list_observe[:40,0]))

print(transmission_base)
transmission_base = 1
iterations = 750
for i, (maxima, point) in enumerate(zip(maxima_points, observation_points_ez_Yee)):
    plt.plot(range(iterations), ez_Yee_list_observe[:,i])
    plt.xlabel('Iteration')
    plt.ylabel('Ez')
    plt.title(f'Ez at {point} in top UCHIE region')
    #plt.show()

    transmission_list_new.append(np.max(np.abs(ez_Yee_list_observe[:maxima,i])))
plt.show()

print(transmission_list_new)

plt.plot(np.array(alpha_list)*180/np.pi, np.array(transmission_list_new)/transmission_base)
plt.xlabel(r'$ \alpha $ [degrees]')
plt.ylabel('Relative magnitude of the peak')
plt.title('Transmission through a hole in a PEC')
plt.show()

"""
plt.plot(range(350), ez_Yee_list_observe)
plt.show()
"""