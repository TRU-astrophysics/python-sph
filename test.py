import numpy as np
import sph_energy as erg
import matplotlib.pyplot as plt
import sph_physicalmethods as phys

total_time = 1e12
dt = total_time / 200

t = np.arange(0., total_time, dt)

grav_pot = np.load("EnergyTestData/Gravitational_Potential.npy")
int_erg = np.load("EnergyTestData/Internal_Energy.npy")
kin_erg = np.load("EnergyTestData/Kinetic_Energy.npy")
tot_erg = np.load("EnergyTestData/Total_Energy.npy")

#print(grav_pot.shape)
#print(int_erg.shape)
#print(kin_erg.shape)
#print(tot_erg.shape)

energy_sum = grav_pot + int_erg + kin_erg



plt.title("Energies over time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.plot(t, tot_erg, label="Total Energy")
plt.plot(t, energy_sum, label="Energy sums")
plt.grid()
plt.legend()
plt.savefig("EnergyPlots/total energies over time.jpg")
plt.show()

print(energy_sum[0])
print(tot_erg[0])

'''
Total energy gathered from both methods are the same to 15 decimal places
'''
################################
# Density and Pressure testing #
################################
# going to see if the behaviour of total pressure and total density
# Density requires position array and smooth length array

pos_arr = np.load("TestingParameters/pos.npy") # shape: (200, 100, 3)
h_arr = np.load("TestingParameters/smoothlen.npy") # (200,100)
interg_arr = np.load("TestingParameters/int_erg.npy") # (200,100)
#print(pos_arr.shape)
#print(h_arr.shape)
#print(interg_arr.shape)
Nt = len(t)
N = 100
dens_arr = np.zeros(Nt)
pressure_arr = np.zeros(Nt)

for tstep in range(Nt):
    for particle in range(N):
        dens_arr[tstep] += phys.density(particle, pos_arr[tstep,:,:],h_arr[tstep,particle])
        pressure_arr[tstep] += phys.pressure(interg_arr[tstep,particle], dens_arr[tstep])

plt.title("Total Pressure over time")
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.plot(t, pressure_arr)
plt.grid()
plt.savefig("EnergyPlots/total pressure over time.jpg")
plt.show()

plt.title("Total Density over time")
plt.xlabel("Time")
plt.ylabel("Density")
plt.plot(t, dens_arr)
plt.grid()
plt.savefig("EnergyPlots/total density over time.jpg")
plt.show()

