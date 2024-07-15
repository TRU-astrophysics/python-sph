import pyfluid as sph
import numpy as np

pos = np.zeros((2, 3))
pos[1][0] = 1

h_initial = 1

h_new = sph.newton_h(0, pos, h_initial)

print(h_new)