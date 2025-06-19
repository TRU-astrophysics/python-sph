import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pyfluid as sph
import time

# NOTE: positions defined with [time][particle][dimension]

# Initial definitions

# Avogadro's number
Na = 6.02214076e23

# Molecular mass in kg. Assuming H2
molecular_mass_kg = 2.016e-3 / Na
# Molecular mass in Solar masses.
molecular_mass = molecular_mass_kg / sph.SOLAR_MASS_IN_KG

# Simulating the solar system. Mass is 1 Solar Mass.
total_mass = 1

# Current diameter of solar system in AU (Oort cloud).
# Could go bigger as we are starting from a sparse molecular cloud.
# Wikipedia claims protoplanetary disks to be around hundreds of AU,
# so considerably smaller than this. Could try that.
# https://en.wikipedia.org/wiki/Formation_and_evolution_of_the_Solar_System
total_size = 2e5

# 3 Million years. Solar system took 600 Myrs to form but disk formed
# in the first 3 million. See here:
# https://spacemath.gsfc.nasa.gov/Grade35/10Page6.pdf
#total_time = 3e8  # 
#total_time = 1e8  # 30 minutes
#total_time = 3e7  # 10 minutes
#total_time = 1e7  # 3 minutes
total_time = 3e6  # 1 minute

dt = 1e5  # Nt is approximately total_time // dt
t = np.arange(0., total_time, dt)

# Intial temperature.
# Wikipedia claims protoplanetary disks are "cool": about 1000 K 
# at their hottest. But should be cooler before collapsing into a disk?
# https://en.wikipedia.org/wiki/Formation_and_evolution_of_the_Solar_System
#T0 = 300  # Kelvin
#T0 = 100  # Kelvin
T0 = 10  # Kelvin

N = 100  # Number of particles
# This overwrites the constant during runtime. But it is ugly,
# we should make classes for everything.
# we should make particle mass a non-constant in the modules, I think we can make objects that take in the mass and sets it for each particle

# JG: I think we can make particle objects and do oject oriented programming, that way the particle class would have its own physical traits
# This might be a good idea as then we wouldn't have to save so many arrays they'd be all stored into a singular array of particles and accessed by methods

sph.PARTICLE_MASS = total_mass / N  # Mass per particle
print("Particle mass: {0:0.5e}".format(sph.PARTICLE_MASS))

# Could think of making the initial distribution more spherically symmetric.
# A cube is kind of weird, but I don;t think it matters much.
pos = (np.random.rand(N, 3) - 0.5) * total_size #N rows by 3 columns
#np.save("temp/pos0", pos)
#pos = np.load("temp/pos0.npy")

# Might as well start with zeros for now. However, this will not 
# give me a disk, as there is no angular momentum.
#vels = np.zeros((N, 3))

# Total angular momentum of solar system seems to be
# L = 3.3212 x 10^45 kg m^2 s^-1 or
# L = 2.3536 SM AU^2 / yr
L = 2.3536 #  Had to multiply by 100 to "see" it rotating.

# Angular speed of a solid sphere of same size and mass.
w = 5 * L / (2 * total_mass * total_size**2)

# Velocities are omega * z_hat cross r_i
vels = w * np.cross(np.array([0, 0, 1]), pos)
np.save("temp/velocities", vels)

# Keep it uniform energy for now.
engs = np.ones(N) * (1 / (sph.ADIABATIC_INDEX - 1) * sph.K_BOLTZMANN 
                                            * T0 / molecular_mass)

# Intial guess should be eta times mean distance between particles:
initial_h = np.ones(N) * sph.COUPLING_CONST * total_size / N**(1/3)

#import os 
#file_path = 'densities_test.txt'
#if os.path.exists(file_path): 
#    os.remove(file_path)
#file_path = 'smoothlengths_test.txt'
#if os.path.exists(file_path): 
#    os.remove(file_path)


print("Total_mass", total_mass)
print("Total_size", total_size)
print("Total_time", total_time)
print("T0", T0)
print("dt", dt)
print("N", N)
print("L", L)

# Sim
start = time.time()
sim_results = sph.var_smoothlength_sim(t, pos, vels, engs, initial_h)
end = time.time()
print("Runtime: {0:0.3e}".format(end - start))


np.save("temp/pos", sim_results[0])
np.save("temp/vel", sim_results[1])
np.save("temp/ener", sim_results[2])
#np.save("hs", sim_results[3])

