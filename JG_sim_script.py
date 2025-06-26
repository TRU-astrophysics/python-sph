# Generates a simple system to test SPH functionalities and gather simple
# data for analysis
# Other Dependencies
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# SPH Specific imports
import sph_sim as sim
import sph_physicalmethods as phys
import sph_energy as erg

# Define everything similarly to mf_sph_run.py and anim.py
# Positions are defined as [time][particle][dimension]
# time from 0 to total_time in step sizes dt
# particle defined from 0 to N, unit steps
# dimension defined from 0 to 2, 0 == x, 1 == y, 2 == z

##########################
# Defining Particle Mass #
##########################
# Avogadro's Number
Na = 6.02214076e23
# Molecular mass in kg. Assuming H2 molecules
molecular_mass_kg = 2.016e-3 / Na
# Molecular mass in Solar masses.
molecular_mass = molecular_mass_kg / phys.SOLAR_MASS_IN_KG

# Simulating the solar system. Mass is 1 Solar Mass.
total_mass = 1

N = 100

particleMass = total_mass/N
# I have to fix this in both modules, but for now I'll just redefine both
# particle masses in phys and erg
phys.PARTICLE_MASS = particleMass
erg.PARTICLE_MASS = particleMass

#################
# Defining Time #
#################
# 3 Million years. Solar system took 600 Myrs to form but disk formed
# in the first 3 million. See here:
# https://spacemath.gsfc.nasa.gov/Grade35/10Page6.pdf
total_time = 3e6


dt = 1e5 # initial test to see if any errors are found
# running the simulation multiple times with smaller time steps to determine
# if energy is conserved
#dt = [1e5, 1e4, 1e3]
t = np.arange(0., total_time, dt)
Nt = total_time/dt

###############################
# Defining Particle Positions #
###############################

# Current diameter of solar system in AU (Oort cloud).
# https://en.wikipedia.org/wiki/Formation_and_evolution_of_the_Solar_System
total_size = 2e5

#pos = (np.random.rand(N,3) - 0.5) * total_size
# Saving initial positions for later use
#np.save("temp/pos0", pos)
pos = np.load("temp/pos0.npy")

################################
# Defining Particle Velocities #
################################


# Total angular momentum of solar system seems to be
# L = 3.3212 x 10^45 kg m^2 s^-1 or
# L = 2.3536 SM AU^2 / yr
# L = 2.3536 #  Had to multiply by 100 to "see" it rotating.
L = 0 # arbitrary, I am seeing only expansion, not contraction
# Angular speed of a solid sphere of same size and mass.
w = 5 * L / (2 * total_mass * total_size**2)

# Velocities are omega * z_hat cross r_i
# IDE states this line of code is unreachable
# I am finding that this method of getting initial velocities causes an expansion of the gas cloud
#vels = w * np.cross(np.array([0, 0, 1]), pos)
vels = np.zeros((N,3))

##############################
# Defining Particle Energies #
##############################

# Initial Temperature
T0 = 10
# Keep it uniform energy for now.
#engs = np.ones(N) * (1 / (phys.ADIABATIC_INDEX - 1) * erg.K_BOLTZMANN
#                                            * T0 / molecular_mass)
engs = np.zeros(N)
# Intial guess should be eta times mean distance between particles:
initial_h = np.ones(N) * phys.COUPLING_CONST * total_size / N**(1/3)

##################################
# Showing all defined parameters #
##################################
print("Total_mass", total_mass)
print("Total_size", total_size)
print("Total_time", total_time)
print("T0", T0)
print("dt", dt)
print("N", N)
print("L", L)
print("time steps", Nt)


# Sim
start = time.time()
sim_results = sim.var_smoothlength_sim(t, pos, vels, engs, initial_h)
end = time.time()
print("Runtime: {0:0.3e}".format(end - start))


np.save("temp/pos", sim_results[0])
np.save("temp/vel", sim_results[1])
np.save("temp/ener", sim_results[2])
#np.save("hs", sim_results[3])

# Saving required parameters
np.save("temp/velocities", vels)


################
# 3D animation #
################

print("Animating simulation...")
df = 5  # decimation factor
pos = np.load("temp/pos.npy")
# pos = np.load("run_logs/run002/pos.npy")

xx = pos[::df, :, 0]
yy = pos[::df, :, 1]
zz = pos[::df, :, 2]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')
scat = ax.scatter(xx[0, :], yy[0, :], zz[0, :], marker='.', alpha=0.8)



def update(t):
    x = xx[t, :]
    y = yy[t, :]
    z = zz[t, :]
    scat._offsets3d = (x, y, z)
    return scat

ani = anim.FuncAnimation(fig=fig, func=update, frames=xx.shape[0], interval=100)

writer = anim.PillowWriter(fps=30)
ani.save(filename="SPH_3D.gif", writer="pillow")

print("Animation Complete")