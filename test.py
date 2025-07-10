import numpy as np
import sph_energy as erg
import matplotlib.pyplot as plt
import sph_physicalmethods as phys

a1 = np.array([1,2,3])
a2 = np.array([4,5])
a3 = np.hstack((a1,a2))
a4 = np.hstack((a3,a1))
print(f"a1: {a1}")
print(f"a2: {a2}")
print(f"a3: {a3}")
print(f"a4: {a4}")
