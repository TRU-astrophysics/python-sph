import numpy as np
import sph_energy as erg
import matplotlib.pyplot as plt
import sph_physicalmethods as phys

position = np.load("temp/pos.npy")
velocity = np.load("temp/vel.npy")
int_energy = np.load("temp/ener.npy")
hs = np.load("temp/hs.npy")
ttime = np.load("temp/time.npy")
print("Position array size: ", position.shape)
print("Velocity array size: ", velocity.shape)
print("internal energy array size: ", int_energy.shape)
print("smooth length array size: ", hs.shape)

Nt = len(ttime)
total_energy = np.zeros(Nt)
pressure_array = np.zeros((Nt, len(position[0,:,0])))
density = np.zeros((Nt, len(position[0,:,0])))


print("Total Energy array size: ", total_energy.shape)
print("Pressure array size: ", pressure_array.shape)
print("Density energy array size: ", density.shape)

# Need to define the pressure and density arrays as nonzero before proceeding
for t in range(Nt):
    density = phys.density_arr(position[t,:,:],hs[t,:])
    pressure = phys.pressure_arr(int_energy[t,:], density)
    total_energy[t]+= erg.total_Energy(
        position[t,:,:],
        velocity[t,:,:],
        pressure,
        density,
        hs[t,:])

plt.scatter(ttime,total_energy)
plt.show()