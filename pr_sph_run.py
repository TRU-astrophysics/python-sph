import numpy as np
import pyfluid as sph

# NOTE: positions defined with [time][particle][dimension]

TOTAL_MASS = 20
RADIUS = 1e2
TEMP_0 = 100

TOTAL_TIME = 18
dt = 1
t = np.arange(0., TOTAL_TIME, dt)


N = 20
sph.PARTICLE_MASS = TOTAL_MASS / N 
pos = (np.random.rand(N, 3)) * RADIUS

# Zero inital vels
vels = np.zeros((N, 3))

engs = np.ones(N) * (1 / (sph.ADIABATIC_INDEX - 1) * sph.K_BOLTZMANN 
                                            * TEMP_0 / sph.HYDROGEN_MASS)

initial_h = np.ones(N) * sph.COUPLING_CONST * RADIUS / N**(1/3)

# NO VISCOSITY
sph.ALPHA_SPH = 0
sph.BETA_SPH = 0

# Sim
sim_results = sph.var_smoothlength_sim(t, pos, vels, engs, initial_h)
np.save("pos_test", sim_results[0])