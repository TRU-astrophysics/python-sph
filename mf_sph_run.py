import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pyfluid as sph
import time

# NOTE: positions defined with [time][particle][dimension]

# Initial definitions

# Simulating the solar system. Mass is 1 Solar Mass.
total_mass = 1

# Current diameter of solar system in AU (Oort cloud).
# Could go bigger as we are starting from a sparse moilecular cloud.
# Wikipedia claims protoplanetary disks to be around hundreds of AU,
# so considerably smaller than this. Could try that.
# https://en.wikipedia.org/wiki/Formation_and_evolution_of_the_Solar_System
total_size = 2e5

# 3 Million years. Solar system took 600 Myrs to form but disk formed
# in the first 3 million. See here:
# https://spacemath.gsfc.nasa.gov/Grade35/10Page6.pdf
#total_time = 1e8
#total_time = 3e7  # 10 minutes
total_time = 1e7  # 3 minutes
#total_time = 3e6  # 1 minute

# Intial temperature.
# Wikipedia claims protoplanetary disks are "cool": about 1000 K 
# at their hottest. But should be cooler before collapsing into a disk?
# https://en.wikipedia.org/wiki/Formation_and_evolution_of_the_Solar_System
T0 = 300  # Kelvin

dt = 1e5  # Nt is approximately total_time // dt
t = np.arange(0., total_time, dt)

N = 50  # Number of particle
# This overwrites the constant during runtime. But it is ugly,
# we should make classes for everything.
sph.PARTICLE_MASS = total_mass / N  # Mass per particle
print("Particle mass: {0:0.5e}".format(sph.PARTICLE_MASS))

# Could think of making the initial distribution more spherically simetric.
# A cube is kind of weird, but I don;t think it matters much.
pos = (np.random.rand(N, 3) - 0.5) * total_size
pos[0, :] = 0  # Put first particle in the centre just for testing
np.save("positions", pos)

# Might as well start with zeros for now. However, this will not 
# give me a disk, as there is no angular momentum.
vels = np.zeros((N, 3))

# Keep it uniform energy for now.
engs = np.ones(N) * (1 / (sph.ADIABATIC_INDEX - 1) * sph.K_BOLTZMANN 
                                            * T0 / sph.PARTICLE_MASS)

# Intial guess should be eta times mean distance between particles:
initial_h = np.ones(N) * sph.COUPLING_CONST * total_size / N**(1/3)

# Sim
start = time.time()
sim_results = sph.var_smoothlength_sim(t, pos, vels, engs, initial_h)
end = time.time()
print("Runtime: {0:0.3e}".format(end - start))

np.save("pos", sim_results[0])
#np.save("vel", sim_results[1])
#np.save("ener", sim_results[2])
#np.save("hs", sim_results[3])

