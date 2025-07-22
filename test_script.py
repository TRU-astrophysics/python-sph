#
# Using this script to set up all the relevant variables and collect the total energy
# There will be no animations in this script
# Generates a simple system to test SPH functionalities and gather simple data for analysis
# Saves all parameters and outputs

# Required Imports
# Other Dependencies #
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
phys.PARTICLE_MASS = total_mass / N

###############################
# Defining Particle Positions #
###############################

# Current diameter of solar system in AU (Oort cloud).
# https://en.wikipedia.org/wiki/Formation_and_evolution_of_the_Solar_System
#total_size = 2e5
total_size = 1e11

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
#L = 2.3536  #  Had to multiply by 100 to "see" it rotating.
L = 0  # gives zero angular speed and zero velocity
# Angular speed of a solid sphere of same size and mass.
w = 5 * L / (2 * total_mass * total_size ** 2)

# Velocities are omega * z_hat cross r_i
# vels = w * np.ones((N,3))
# print(vels[0,:])
vels = w * np.cross(np.array([0, 0, 1]), pos)
# pycharm states that the above function is not reachable?
# This is an IDE error and I tested to see if there would be a change before and after

##############################
# Defining Particle Energies #
##############################

# Initial Temperature
# T0 = 10 # standard starting temperature
# T0 = 1  # for testing energy
T0 = 0
# Keep it uniform energy for now.
engs = np.ones(N) * (1 / (phys.ADIABATIC_INDEX - 1) * erg.K_BOLTZMANN
                     * T0 / molecular_mass)
#engs = np.zeros(N)
# Intial guess should be eta times mean distance between particles:
initial_h = np.ones(N) * phys.COUPLING_CONST * total_size / N ** (1 / 3)

#################
# Defining Time #
#################
# 3 Million years. Solar system took 600 Myrs to form but disk formed
# in the first 3 million. See here:
# https://spacemath.gsfc.nasa.gov/Grade35/10Page6.pdf
total_time = 1e12
# Defining Nt as an array, then going to run each simulation based on this
# Found out that there is a limit to the number of large time steps. Nt = 5 raises an error
#Nt = np.array([10,20,50,100,200,500,1000])
#Nt = np.array([10,20,50,100])
#Nt = np.array([10, 20]) # used for testing loop and plotting functionality
Nt = 200  # from my plots this was the smallest number of steps required for the smallest energy delta
# Nt = 10  -->  56s runtime
# Nt = 20  --> 115s runtime
# Nt = 50  --> 269s runtime
# Nt = 100 --> 532s runtime
# Overall runtime: 16 minutes and 12 seconds

#Nt = np.array([10])

dt = total_time/Nt # initial test to see if any errors are found
# running the simulation multiple times with smaller time steps to determine
# if energy is conserved
#dt = [1e5, 1e4, 1e3]
#t = np.arange(0., total_time, dt)
#Nt = total_time/dt

##################################
# Showing all defined parameters #
##################################
#print("Total_mass", total_mass)
#print("Total_size", total_size)
#print("Total_time", total_time)
#print("T0", T0)
# print("dt", dt)
#print("N", N)
#print("L", L)
#print("time steps", Nt)
#print("Initial smooth length: ", initial_h[0])

#energy_delta = np.zeros(Nt)
#stepper = 0  # steps through the energy_delta
#####################################################
# Running the simulation and collecting energy data #
#####################################################
'''
for n in Nt:
    t = np.linspace(0., total_time, n)  # creates a time array of size n from 0 to total_time
    total_energy = np.zeros(len(t))
    # run simulation
    start = time.time()
    pos_arr, vel_arr, erg_arr, h_arr = sim.var_smoothlength_sim(t, pos, vels, engs, initial_h)
    end = time.time()
    print("Runtime: {0:0.3e}".format(end - start))
    # calculate total energy
    for i in range(n):
        density = phys.density_arr(pos_arr[i, :, :], h_arr[i, :])
        pressure = phys.pressure_arr(erg_arr[i, :], density)
        total_energy[i] += erg.total_Energy(
            pos_arr[i, :, :],
            vel_arr[i, :, :],
            pressure,
            density,
            h_arr[i, :])
    energy_delta[stepper] = total_energy[-1] - total_energy[0]
    stepper += 1
    # plot energy over time
    plt.plot(t, total_energy, label=f"{n} steps")
    np.save(f"temp/total_energy_{n}", total_energy)
np.save("temp/Nt", Nt)
np.save("temp/energy_delta", energy_delta)
'''
#####################################
# Simulation for different energies #
#####################################
t = np.arange(0., total_time, dt)
total_energy = np.zeros(len(t))
grav_pot = np.zeros(len(t))
kin_erg = np.zeros(len(t))
int_erg = np.zeros(len(t))
start = time.time()
pos_arr, vel_arr, erg_arr, h_arr = sim.var_smoothlength_sim(t, pos, vels, engs, initial_h)
end = time.time()
print("Runtime: {0:0.3e}".format(end - start))
for i in range(len(t)):
    density = phys.density_arr(pos_arr[i, :, :], h_arr[i, :])
    pressure = phys.pressure_arr(erg_arr[i, :], density)
    total_energy[i] += erg.total_Energy(
        pos_arr[i, :, :],
        vel_arr[i, :, :],
        pressure,
        density,
        h_arr[i, :])
    for particle in range(N):
        grav_pot[i] += erg.grav_potential(particle, pos_arr[i, :, :], h_arr[i, particle])
        kin_erg[i] += erg.kinetic_energy(vel_arr[i, particle, :])
        int_erg[i] += erg.internal_energy(pressure[particle], density[particle])

###############
# Saving data #
###############
np.save("TestingParameters/pos", pos_arr)
np.save("TestingParameters/vel", vel_arr)
np.save("TestingParameters/int_erg", erg_arr)
np.save("TestingParameters/smoothlen", h_arr)

np.save("EnergyTestData/Gravitational_Potential", grav_pot)
np.save("EnergyTestData/Kinetic_Energy", kin_erg)
np.save("EnergyTestData/Internal_Energy", int_erg)
np.save("EnergyTestData/Total_Energy", total_energy)

######################################
# Plotting the various energy graphs #
######################################
plt.title("Total Energy over time")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.plot(t, total_energy)
plt.grid()
plt.savefig("EnergyPlots/Total Energy over time.jpg")
plt.show()

plt.title("Gravitational Potential Energy over time")
plt.xlabel("Time")
plt.ylabel("Gravitational Potential")
plt.plot(t, grav_pot)
plt.grid()
plt.savefig("EnergyPlots/Gravitational Potential Energy over time.jpg")
plt.show()

plt.title("Kinetic Energy over time")
plt.xlabel("Time")
plt.ylabel("Kinetic Energy")
plt.plot(t, kin_erg)
plt.grid()
plt.savefig("EnergyPlots/Kinetic Energy over time.jpg")
plt.show()

plt.title("Internal Energy over time")
plt.xlabel("Time")
plt.ylabel("Internal Energy")
plt.plot(t, int_erg)
plt.grid()
plt.savefig("EnergyPlots/Internal Energy over time.jpg")
plt.show()

plt.title("Energies over time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.plot(t, int_erg, label="Internal Energy")
plt.plot(t, grav_pot, label="Gravitational Energy")
plt.plot(t,kin_erg, label="Kinetic Energy")
plt.grid()
plt.legend()
plt.savefig("EnergyPlots/Energies over time.jpg")
plt.show()

#########################
# Plotting Energy Delta #
#########################
'''
plt.scatter(Nt, energy_delta)
plt.title("Energy delta over time steps")
plt.xlabel("Time Steps")
plt.ylabel("Energy Delta")
plt.grid()
plt.savefig("Energy Delta over time steps.jpg")
plt.show()
'''
