import pyfluid as sph
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

pos = np.ones((100, 3))

for i in range(pos.shape[0]):
    pos[i] = np.array([rng.random(), rng.random(), rng.random()])

h_initial = 0.2

print("final hj: " + str(sph.newton_h_while(0, pos, h_initial)))


'''
d = np.arange(0, 5, 0.01)
M4_d1_h1 = np.zeros(len(d))
M4_d1_h2 = np.zeros(len(d))
M4_d1_h3 = np.zeros(len(d))
M4_d1_h4 = np.zeros(len(d))

for i in range(0, len(d)):
    M4_d1_h1[i] = sph.M4_h_derivative(d[i], 1)
    M4_d1_h2[i] = sph.M4_h_derivative(d[i], 2)
    M4_d1_h3[i] = sph.M4_h_derivative(d[i], 3) 
    M4_d1_h4[i] = sph.M4_h_derivative(d[i], 0.34401218)   

plt.plot(d, M4_d1_h1, 'r', d, M4_d1_h2, 'b', d, M4_d1_h3, 'g', d, M4_d1_h4, 'y')
plt.show()'''