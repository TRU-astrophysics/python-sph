import numpy as np
import pyfluid as sph

# NOTE: positions defined with [time][particle][dimension]

TOTAL_MASS = 1
RADIUS = 1e5
TEMP_0 = 1e4

TOTAL_TIME = 1e7
dt = 2e5
t = np.arange(0., TOTAL_TIME, dt)


N = 50
sph.PARTICLE_MASS = TOTAL_MASS / N 
pos = (np.random.rand(N, 3)) * RADIUS

# Zero inital vels
vels = np.zeros((N, 3))

# Should particle mass be the mass of hydrogen?
engs = np.ones(N) * (1 / (sph.ADIABATIC_INDEX - 1) * sph.K_BOLTZMANN 
                                            * TEMP_0 / sph.PARTICLE_MASS)

# How does diameter / N**1/3 give us mean distance?
initial_h = np.ones(N) * sph.COUPLING_CONST * RADIUS / N**(1/3)

# Sim
sim_results = sph.var_smoothlength_sim(t, pos, vels, engs, initial_h)
np.save("pr_pos", sim_results[0])