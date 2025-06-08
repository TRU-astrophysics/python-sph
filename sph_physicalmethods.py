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

def Pi(j, i, position_arr, velocity_arr, pressure_arr, density_arr, smoothlength_arr):
    """ 
    Cossins eq. 3.85.
    This function is called N^2 times. However, being simmetric, there are only
    N(N-1)/2 different quantities. Theres is potential for economy.
    (Except that we only do it for convergent flow, so maybe we are fine?)
    """
    v_dot_r = np.dot(position_arr[j] - position_arr[i], velocity_arr[j] - velocity_arr[i])
    if v_dot_r < 0:  # Bulk viscosity only for convergent flow.
        # Compute mu
        mean_h = 0.5 * (smoothlength_arr[j] + smoothlength_arr[i])
        dist_ij = distance(position_arr[j], position_arr[i])
        mu = mean_h * v_dot_r / (dist_ij**2 + EPSILON * mean_h**2)
        # Compute average sound speed
        csj = (ADIABATIC_INDEX * pressure_arr[j] / density_arr[j])**0.5
        csi = (ADIABATIC_INDEX * pressure_arr[i] / density_arr[i])**0.5
        cs = 0.5 * (csi + csj)
        # Compute mean density
        mean_density = 0.5 * (density_arr[j] + density_arr[i])
        Pi_value = (- ALPHA_SPH * cs * mu + BETA_SPH * mu**2) / mean_density
        if Pi_value < 0:
            raise RuntimeError("Obtained negative value for Pi")
        return Pi_value
    else:
        return 0
		
# MF: This function and "energy_evolve_arr" are deprecated.
# The new functions (including viscosity) work by returning the energy rate
# and letting the leapfrog function do the evolution.
# TODO: Either delete (we can always remove viscosity by setting constants to zero)
# or turn into an energy-rate.
def energy_evolve(j, position_arr, velocity_arr, energy_arr, pressure_arr, density_arr, smoothlength_j, dt):
    """
    This is energy evolution without variable smoothing length and without viscosity. Cossins eq. (3.74).
    PR: energy is currently just evolved w/ Euler integration - maybe not ideal?
    MF: I think we don't have to evolve energy here. Just compute dy/dt and leave the evolution for the leapfrog integrator.
    This is what I did for the version with viscosity.
    """
    density_change = 0
    
    for i in range(velocity_arr.shape[0]):
        density_change += PARTICLE_MASS * np.dot(velocity_arr[j] - velocity_arr[i], dellM4(position_arr[j], position_arr[i], smoothlength_j))
    
    energy_change = pressure_arr[j] / density_arr[j]**2 * density_change
    
    return energy_arr[j] + energy_change * dt

def energy_evolve_arr(position_arr, velocity_arr, energies_initial, pressure_arr, density_arr, smoothlength_arr, omega_arr, dt):
    energies_final = np.zeros(position_arr.shape[0])

    for i in range(position_arr.shape[0]):
        energies_final[i] = energy_evolve(i, position_arr, velocity_arr, energies_initial, pressure_arr, density_arr, smoothlength_arr[i], dt)

    return energies_final

def energy_rate(j, position_arr, velocity_arr, pressure_arr, density_arr, smoothlength_arr, omega_arr):
    """
    This is energy evolution including variable smoothing length 
    and viscosity. This equation was not given in Cossins and was 
    worked out by MF. Need to be tested as there might be errors.
    """
    density_change = 0
    viscosity_sum = 0

    for i in range(velocity_arr.shape[0]):
        density_change += PARTICLE_MASS * np.dot(
                        velocity_arr[j] - velocity_arr[i],
                        dellM4(position_arr[j], position_arr[i], 
                                                smoothlength_arr[j]))


        
        mean_dellM4 = 0.5 * (dellM4(position_arr[j], position_arr[i], 
                                    smoothlength_arr[j])
                           + dellM4(position_arr[j], position_arr[i], 
                                    smoothlength_arr[i]))
        # MF: I am unsure if we should have vj, vji or 0.5*vji here.
        # The thesis says vji, but I can't understand this result.
        # Using vji ensures energy only increase due to viscosity.
        # It also makes the overall magnitude of the energy rate due 
        # to viscosity smaller, and more in line with the "density_change"
        # term. I think we need a factor of 0.5 here though.
        # TODO: investigate.
        vji = velocity_arr[j] - velocity_arr[i]
        viscosity_sum += (PARTICLE_MASS * Pi(j, i, position_arr, 
                                             velocity_arr, pressure_arr, 
                                             density_arr, smoothlength_arr) 
                          * np.dot(vji, mean_dellM4))
                          #* np.dot(velocity_arr[j], mean_dellM4))

    # MF: This test could actually be inside the for loop above.
    # But that would be more expensive.
    if np.any(viscosity_sum < 0):
        raise RuntimeError("Obtained negative energy rate due to viscosity.")

    return (pressure_arr[j] / (density_arr[j]**2 * omega_arr[j]) 
            * density_change + viscosity_sum)

def energy_rate_arr(position_arr, velocity_arr, pressure_arr, density_arr, smoothlength_arr, omega_arr):
    energy_rate_array = np.zeros(position_arr.shape[0])

    for i in range(position_arr.shape[0]):
        energy_rate_array[i] = energy_rate(i, position_arr, velocity_arr, pressure_arr, density_arr, smoothlength_arr, omega_arr)

    return energy_rate_array


# Gravity
def grav_potential(dist, smoothlength):
    """
    Cossins eq. 3.149
    """
    x = dist/smoothlength
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [1/smoothlength * (2/3*x**2 - 3/10*x**3 + 1/10*x**5 - 7/5), 
                         1/smoothlength * (4/3*x**2 - x**3 + 3/10*x**4 - 1/30*x**5 - 8/5 + 1/(15*x)),
                         -1/dist])    

def grav_force(dist, smoothlength):
    """
    Cossins eq. 3.148
    """
    x = dist/smoothlength
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [1/(smoothlength**2) * (4/3*x - 6/5*x**3 + 1/2*x**4), 
                         1/(smoothlength**2) * (8/3*x - 3*x**2 + 6/5*x**3 - 1/6*x**4 - 1/(15*x**2)),
                         1/(dist**2)]) 

def grav_potential_smoothlength_derivative(dist, smoothlength):
    x = dist/smoothlength
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [-1/(smoothlength**2) * (2*x**2 - 12/10*x**3 + 6/10*x**5 - 7/5), 
                         -1/(smoothlength**2) * (4*x**2 - 4*x**3 + 3/2*x**4 - 1/5*x**5 - 8/5),
                         0])

def xi(j, position_arr, density_j, smoothlength_j):
    
    sum  = 0

    for i in range(position_arr.shape[0]):
        dist = distance(position_arr[j], position_arr[i])
        # TODO: numpy.piecewise can get an array for input. We should use it instead of this for loop!
        sum += PARTICLE_MASS * grav_potential_smoothlength_derivative(dist, smoothlength_j)

    return - smoothlength_j/(3*density_j) * sum

def xi_arr(density_arr, smoothlength_arr, position_arr):
    xi_arr = np.zeros(position_arr.shape[0])

    for j in range(position_arr.shape[0]):
        xi_arr[j] = xi(j, position_arr, density_arr[j], smoothlength_arr[j])
    
    return xi_arr

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

def smoothed_gravity_correction_comp(j, i, position_arr, smoothlength_arr, omega_arr, xi_arr):

    return -G/2 * PARTICLE_MASS * (xi_arr[j]/omega_arr[j] * dellM4(position_arr[j], position_arr[i], smoothlength_arr[j])
                                  + xi_arr[i]/omega_arr[i] * dellM4(position_arr[j], position_arr[i], smoothlength_arr[i]))

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