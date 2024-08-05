import numpy as np
import pyfluid as sph

positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0.5]])

h = sph.newton_h(0, positions, sph.INITIAL_H, 1.5)
density = sph.var_density(h)
print(h)
print(density)
print(sph.density(positions[0], positions))