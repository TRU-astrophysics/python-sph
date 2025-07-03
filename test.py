import numpy as np
import sph_energy as erg
import matplotlib.pyplot as plt

position = np.load("temp/pos.npy")
velocity = np.load("temp/vel.npy")
int_energy = np.load("temp/ener.npy")
hs = np.load("temp/hs.npy")
ttime = np.load("temp/time.npy")
print("Position array size: ", position.shape)
print("Velocity array size: ", velocity.shape)
print("internal energy array size: ", int_energy.shape)
print("smooth length array size: ", hs.shape)

total_energy = np.zeros(len(ttime))
pressure_array = np.zeros((len(total_energy), len(position[0,:,0])))
density = np.zeros((len(total_energy), len(position[0,:,0])))

print("Total Energy array size: ", total_energy.shape)
print("Pressure array size: ", pressure_array.shape)
print("Density energy array size: ", density.shape)


for t in range(len(total_energy)):
    total_energy+= erg.total_Energy(position[t,:,:],velocity[t,:,:],pressure_array[t,:],density[t,:],hs[t])

plt.scatter(ttime,total_energy)
plt.show()