import numpy as np
import sph_gravity as grav
import sph_physicalmethods as phys
import sph_NumericalMethods as num

# ALL FUNCTIONS currently assume uniform masses
PARTICLE_MASS = 1
DEFAULT_SMOOTHLENGTH = 2

# Coupling constant is eta in Cossins's thesis (see 3.98)
COUPLING_CONST = 1.3

# Constants for determining smoothing lengths:
# Newton's method
SMOOTHLENGTH_VARIATION_TOLERANCE = 1e-3
NEWTON_ITERATION_LIMIT = 20
INITIAL_NEWTON_SMOOTHELENGTH = 0.2

# Viscosity
ALPHA_SPH = 1
BETA_SPH = 2
EPSILON = 0.01  # TODO: We need to check that this value is suitable.

# Physical Constants
# G in units of years, solar masses, AU
# 39.4324 would be slightly more accurate
# MF: Hmm, I got 39.4227
#G = 4 * np.pi**2
G = 39.4227

# Boltzmann's constant in units of years, solar masses, AU, Kelvin
K_BOLTZMANN = 3.0856e-61

# Adiabatic index (gamma):
ADIABATIC_INDEX = 5 / 3


def internal_energy(pressure_i, density_i):
    denom = (ADIABATIC_INDEX - 1) * density_i
    u = pressure_i / denom
    return u

def kinetic_energy(v_i):
    E_k = .5*PARTICLE_MASS* np.dot(v_i,v_i)
    return E_k

#Cossins eq.3.118
def grav_potential(i, position_arr, smoothlength_i):
    pot = 0
    for j in range(position_arr.shape[0]):
        pot += PARTICLE_MASS * grav.grav_kernal(position_arr[i]-position_arr[j], smoothlength_i)
    return 0.5*G*pot
    
def total_Energy(position_arr, vel_arr, press_arr, density_arr, smoothlength_arr):
    E = 0
    for i in range(position_arr.shape[0]):
        E+= (internal_energy(press_arr[i],density_arr[i]) + kinetic_energy(vel_arr[i]) + grav_potential(i, position_arr, smoothlength_arr[i]))
    return E

def Pi(j, i, position_arr, velocity_arr, pressure_arr, density_arr, smoothlength_arr):
    """ 
    Cossins eq. 3.85.
    This function is called N^2 times. However, being symmetric, there are only
    N(N-1)/2 different quantities. Theres is potential for economy.
    (Except that we only do it for convergent flow, so maybe we are fine?)
    """
    v_dot_r = np.dot(position_arr[j] - position_arr[i], velocity_arr[j] - velocity_arr[i])
    if v_dot_r < 0:  # Bulk viscosity only for convergent flow.
        # Compute mu
        mean_h = 0.5 * (smoothlength_arr[j] + smoothlength_arr[i])
        dist_ij = phys.distance(position_arr[j], position_arr[i])
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
        density_change += PARTICLE_MASS * np.dot(velocity_arr[j] - velocity_arr[i], num.dellM4(position_arr[j], position_arr[i], smoothlength_j))
    
    energy_change = pressure_arr[j] / density_arr[j]**2 * density_change
    
    return energy_arr[j] + energy_change * dt


def energy_evolve_arr(position_arr, velocity_arr, energies_initial, pressure_arr, density_arr, smoothlength_arr, omega_arr, dt):
    energies_final = np.zeros(position_arr.shape[0])

    for i in range(position_arr.shape[0]):
        energies_final[i] = energy_evolve(i, position_arr, velocity_arr, energies_initial, pressure_arr, density_arr, smoothlength_arr[i], dt)

    return energies_final


# du/dt?
def energy_rate(j, position_arr, velocity_arr, pressure_arr, density_arr, smoothlength_arr, omega_arr):
    """
    This is energy evolution including variable smoothing length 
    and viscosity. This equation was not given in Cossins and was 
    worked out by MF. Need to be tested as there might be errors.
    """
    density_change = 0
    viscosity_sum = 0

    for i in range(velocity_arr.shape[0]):
        
        vji = velocity_arr[j] - velocity_arr[i]
        # d/dt of rho
        density_change += PARTICLE_MASS * np.dot(
                        vji,
                        num.dellM4(position_arr[j], position_arr[i], smoothlength_arr[j]))



        mean_dellM4 = (num.dellM4(position_arr[j], position_arr[i],
                                    smoothlength_arr[j])
                           + num.dellM4(position_arr[j], position_arr[i],
                                    smoothlength_arr[i]))
        # MF: I am unsure if we should have vj, vji or 0.5*vji here.
        # The thesis says vji, but I can't understand this result.
        # Using vji ensures energy only increase due to viscosity.
        # It also makes the overall magnitude of the energy rate due 
        # to viscosity smaller, and more in line with the "density_change"
        # term. I think we need a factor of 0.5 here though.
        # JG: I found an oddity, not sure if it's this term. I am removing the 0.5 factor and
        # seeing if this changes what I found
        # TODO: investigate.
        
        #d/dt of kappa
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
