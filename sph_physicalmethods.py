import numpy as np

# NOTE: Need to create an array of masses if we want differing particle masses
# ALL FUNCTIONS currently assume uniform masses
PARTICLE_MASS = 1
DEFAULT_SMOOTHLENGTH = 2

# Coupling constant is eta in Cossins's thesis (see 3.98)
COUPLING_CONST = 1.3 # used in variable smoothing length 

# Conversion Factors, gathered from Wikipedia on May13, 2025
AU_IN_METERS = 1.495978707e11
SOLAR_MASS_IN_KG = 1.988416e30
YEAR_IN_SECONDS = 31536000
JOULE_IN_ASTRONOMICAL = 2.2349e-38 # in AU, Solar mass and years, 1 Joule conversion, calculated by hand

# Physical Constants
# G in units of years, solar masses, AU
# 39.4324 would be slightly more accurate
# MF: Hmm, I got 39.4227
#G = 4 * np.pi**2
G = 39.4227

# Boltzmann's constant in units of years, solar masses, AU, Kelvin
K_BOLTZMANN = 3.0856e-61

# Adiabatic index (gamma):
ADIABATIC_INDEX = 5/3

# Viscosity
ALPHA_SPH = 1
BETA_SPH = 2
EPSILON = 0.01  # TODO: We need to check that this value is suitable.


'''
inputs:
position vector of two particles

output:
the scalar magnitude of the distance between the two particles
'''
def distance(rj, ri):
    diff = rj - ri
    dist = np.sqrt(diff.dot(diff))

    return dist
'''
inputs: 
position vectors of two particles

outputs: 
the unit direction vector from i to j
'''
def direction_i_j(rj, ri):
    if np.array_equal(rj, ri):
        return np.zeros(3)

    return (rj-ri)/distance(rj, ri)

# MF: Functions like this one that are one line long are not helpful I think.
def density_comp(j, i, position_arr, smoothlength_j):
    return PARTICLE_MASS * M4(distance(position_arr[j], position_arr[i]), smoothlength_j)

def density(j, position_arr, smoothlength_j):
    density = 0
    
    for i in range(position_arr.shape[0]):
        density += density_comp(j, i, position_arr, smoothlength_j)

    return density

def density_arr(position_arr, smoothlength_arr):
    density_arr = np.zeros(position_arr.shape[0])
    
    for i in range(position_arr.shape[0]):
        density_arr[i] = density(i, position_arr, smoothlength_arr[i])

    return density_arr


# New & improved density
# Cossins 3.98
def var_density(smoothlength):
    return PARTICLE_MASS * (COUPLING_CONST/smoothlength)**3

def var_density_arr(smoothlength_arr):
    density_arr = np.zeros(len(smoothlength_arr))
    for i in range(len(smoothlength_arr)):
        density_arr[i] = var_density(smoothlength_arr[i])
    
    return density_arr


def pressure(energy, density):
    return (ADIABATIC_INDEX - 1) * energy * density

def pressure_arr(energy_arr, density_arr):
    pressure_arr = np.zeros(energy_arr.shape[0])

    for i in range(energy_arr.shape[0]):
        pressure_arr[i] = pressure(energy_arr[i], density_arr[i])

    if np.any(pressure_arr < 0):
        raise RuntimeError("Obtained negative pressure.")
    return pressure_arr


# Cossins 3.128?
def smoothed_gravity_acceleration_comp(j, i, position_arr, smoothlength_arr):
    
    if np.array_equal(position_arr[j], position_arr[i]):
        return 0

    dist = distance(position_arr[j], position_arr[i])
    
    # TODO: Should we put this G/2 multiplication outside the sum in acceleration()? 
    # This would mean having different for loops for each term. Not sure.
    # If we decide to do it, the same should happen to smoothed_gravity_correction_comp().
    return -G/2 * PARTICLE_MASS * ((grav_force(dist, smoothlength_arr[j]) 
                                  + grav_force(dist, smoothlength_arr[i]))
                                  * direction_i_j(position_arr[j], position_arr[i]))


# Cossins 3.133
def smoothed_gravity_correction_comp(j, i, position_arr, smoothlength_arr, omega_arr, xi_arr):

    return -G/2 * PARTICLE_MASS * (xi_arr[j]/omega_arr[j] * dellM4(position_arr[j], position_arr[i], smoothlength_arr[j])
                                  + xi_arr[i]/omega_arr[i] * dellM4(position_arr[j], position_arr[i], smoothlength_arr[i]))


# Newton's Law of Gravitation
def basic_gravity_acceleration_comp(j, i, position_arr):
    
    if np.array_equal(position_arr[j], position_arr[i]):
        return 0
    
    return -direction_i_j(position_arr[j], position_arr[i]) * G/distance(position_arr[j], position_arr[i])**2


# MF: This function is deprecated. The new version (with viscosity) needs the velocities as inout.
# TODO: Should probably delete it (we can always remove viscosity by setting constants to zero).
def fluid_acceleration_comp(j, i, position_arr, density_arr, pressure_arr, smoothlength_arr, omega_arr):
    
    if np.array_equal(position_arr[j], position_arr[i]):
        return 0

    return -PARTICLE_MASS * (pressure_arr[j]/(omega_arr[j] * density_arr[j]**2) * dellM4(position_arr[j], position_arr[i], smoothlength_arr[j])
                            + pressure_arr[i]/(omega_arr[i] * density_arr[i]**2) * dellM4(position_arr[j], position_arr[i], smoothlength_arr[i]))


# Acceleration
def fluid_accel_viscosity_comp(j, i, position_arr, velocity_arr, density_arr, pressure_arr, smoothlength_arr, omega_arr):
    
    if np.array_equal(position_arr[j], position_arr[i]):
        return 0

    dellM4_j = dellM4(position_arr[j], position_arr[i], smoothlength_arr[j])
    dellM4_i = dellM4(position_arr[j], position_arr[i], smoothlength_arr[i])

    accel = (pressure_arr[j]/(omega_arr[j] * density_arr[j]**2) * dellM4_j
            + pressure_arr[i]/(omega_arr[i] * density_arr[i]**2) * dellM4_i)

    viscosity_correction = Pi(j, i, position_arr, velocity_arr, 
                              pressure_arr, density_arr, 
                              smoothlength_arr) * 0.5 * (dellM4_j + dellM4_i)

    return -PARTICLE_MASS * (accel + viscosity_correction)


def acceleration(j, position_arr, density_arr, velocity_arr, pressure_arr, smoothlength_arr, omega_arr, xi_arr):
    acc = np.array([0., 0., 0.])

    for i in range(position_arr.shape[0]):
        acc += (fluid_accel_viscosity_comp(j, i, position_arr, velocity_arr, density_arr, pressure_arr, smoothlength_arr, omega_arr) 
              + smoothed_gravity_acceleration_comp(j, i, position_arr, smoothlength_arr)
              + smoothed_gravity_correction_comp(j, i, position_arr, smoothlength_arr, omega_arr, xi_arr))

    return acc

def acceleration_arr(position_arr, density_arr, velocity_arr, pressure_arr, smoothlength_arr, omega_arr, xi_arr):
    acc_arr = np.zeros((position_arr.shape[0], 3))
    
    for i in range(position_arr.shape[0]):
        acc_arr[i] = acceleration(i, position_arr, density_arr, velocity_arr, pressure_arr, smoothlength_arr, omega_arr, xi_arr)

    return acc_arr