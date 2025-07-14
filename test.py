import numpy as np
import sph_energy as erg
import matplotlib.pyplot as plt
import sph_physicalmethods as phys

pos = np.load("temp/pos0 where viscosity_sum is negative.npy")

N = len(pos[:,0])
for i in range(N):
    for j in range(N):
        if i != j:
            dist = phys.distance(pos[i,:],pos[j,:])
            if dist==0:
                print("zero distance found")
            elif dist<=0:
                print("negative distance found")
print("Completed")
