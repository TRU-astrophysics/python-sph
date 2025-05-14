import numpy as np

# NOTE: Need to create an array of masses if we want differing particle masses
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

# Bisection 
BISECTION_ITERATION_LIMIT = 100
BISECTION_TOLERANCE = 1e-10
INITIAL_BISECTION_SMOOTHLENGTH_A = 1e3
INITIAL_BISECTION_SMOOTHLENGTH_B = 1e-3

# Conversion Factors
AU_IN_METERS = 1.495978707e11
SOLAR_MASS_IN_KG = 1.988416e30 # As of May 13, 2025
YEAR_IN_SECONDS = 31536000
JOULE_IN_ASTRONOMICAL = 2.2349e-38 # in AU, Solar mass and years, 1 Joule conversion

# Physical Constants

# G in units of years, solar masses, AU
# 39.4324 would be slightly more accurate
# MF: Hmm, I got 39.4227
# JG: I got a value of 39.423025 using values from wikipedia
#G = 4 * np.pi**2
G = 39.4227

# Boltzmann's constant in units of years, solar masses, AU, Kelvin
K_BOLTZMANN = 3.0856e-61

# Adiabatic index (gamma):
ADIABATIC_INDEX = 5/3


def distance(rj, ri):
    diff = rj - ri
    dist = np.sqrt(diff.dot(diff))

    return dist

# Note: direction_i_j gives direction vector FROM i TO j
def direction_i_j(rj, ri):
    if np.array_equal(rj, ri):
        return np.zeros(3)

    return (rj-ri)/distance(rj, ri)

def M4(dist, smoothlength):
    q = dist/smoothlength
    w = 1/(np.pi*smoothlength**3)*np.piecewise(q, 
                                    [q >= 2, q >= 1 and q < 2, q < 1], 
                                    [0, 0.25*(2 - q)**3, 1 - 3/2*q**2 + 3/4*q**3])
    
    return w

def M4_d1(dist, smoothlength):
    q = dist/smoothlength
    wp = 1/(np.pi*smoothlength**4)*np.piecewise(q, 
                                       [q >= 2, q >= 1 and q < 2, q < 1], 
                                       [0, -0.75*(2 - q)**2, 9/4 * q**2 - 3*q])
    
    return wp

def M4_d2(dist, smoothlength):
    q = dist/smoothlength
    ## should have some coeff
    wpp = np.piecewise(q, 
                       [q >= 2, q >= 1 and q < 2, q < 1], 
                       [0, 1.5*(2 - q), (1.5*(2 - q) - 6*(1 - q))])
    
    return wpp

def dellM4(rj, ri, smoothlength):
    return M4_d1(distance(rj, ri), smoothlength) * direction_i_j(rj, ri)

# Cossins 3.107
def M4_smoothlength_derivative(dist, smoothlength):
    x = dist/smoothlength
    return -x*M4_d1(dist, smoothlength) - 3/smoothlength * M4(dist, smoothlength)


# Cossins 3.106
def omega_j(j, position_arr, density_j, smoothlength_j):
    sum = 0
    for i in range(position_arr.shape[0]):
        dist = distance(position_arr[j], position_arr[i])
        sum += PARTICLE_MASS * M4_smoothlength_derivative(dist, smoothlength_j)
    
    return 1 + smoothlength_j/(3 * density_j) * sum

def omega_arr(position_arr, density_arr, smoothlength_arr):
    omega_arr = np.zeros(position_arr.shape[0])

    for j in range(position_arr.shape[0]):
        omega_arr[j] = omega_j(j, position_arr, density_arr[j], smoothlength_arr[j])
    
    return omega_arr

# Cossins 3.112 - 3.115
def zeta_j(smoothlength_j, density_j):
    zeta = var_density(smoothlength_j) - density_j
    return zeta

# NOTE: We may be able to find zetaprime with 
# -3 * density_j * omega_j / smoothlength_j
def zetaprime(density_j, omega_j, smoothlength_j):
    mj = PARTICLE_MASS
    return (- 3 * density_j / smoothlength_j * (omega_j - 1) 
            - 3 * mj * (COUPLING_CONST / smoothlength_j)**3 / smoothlength_j)

def smoothlength_variation(new_h, old_h):
    return np.abs(new_h - old_h)/np.abs(old_h)

# Bisection should perhaps be broken down into a series of smaller functions, like newton's method is. 
def bisection_h(j, position_arr):
    h_a = INITIAL_BISECTION_SMOOTHLENGTH_A
    h_b = INITIAL_BISECTION_SMOOTHLENGTH_B
    i = 0

    while i < BISECTION_ITERATION_LIMIT:
        density_a = density(j, position_arr, h_a)
        zeta_a = zeta_j(h_a, density_a)
        density_b = density(j, position_arr, h_b)
        zeta_b = zeta_j(h_b, density_b)

        # zeta(h_a) * zeta(h_b) must be < 0 because one must be pos and one neg, so there is a root of zeta between
        if zeta_a * zeta_b > 0:
            print("ERROR: Root out of bisection bounds.")
        
        root_candidate = (h_a + h_b)/2
        zeta_root = zeta_j(root_candidate, density(j, position_arr, root_candidate))
        
        if np.abs(zeta_root) < BISECTION_TOLERANCE:
            return root_candidate
        
        else: 
            if zeta_a * zeta_root < 0:
                h_b = root_candidate
        
            elif zeta_b * zeta_root < 0:
                h_a = root_candidate
        
            i += 1
        
    print("ERROR: Failure to converge in Bisection_new_h.")
        
def newton_smoothlength_iteration(j, position_arr, old_smoothlength_j):

    old_density_j = density(j, position_arr, old_smoothlength_j)
    omega = omega_j(j, position_arr, old_density_j, old_smoothlength_j)
    zeta = zeta_j(old_smoothlength_j, old_density_j)
    zetap = zetaprime(old_density_j, omega, old_smoothlength_j)
    new_smoothlength_j = newton_new_h(old_smoothlength_j, zeta, zetap)

    return new_smoothlength_j

def newton_smoothlength_while(j, position_arr, initial_smoothlength_j):
    old_smoothlength_j = initial_smoothlength_j

    i = 0
    while i < NEWTON_ITERATION_LIMIT:
        current_smoothlength_j = newton_smoothlength_iteration(j, position_arr, old_smoothlength_j)

        if smoothlength_variation(old_smoothlength_j, current_smoothlength_j) < SMOOTHLENGTH_VARIATION_TOLERANCE and current_smoothlength_j > 0:
            return current_smoothlength_j
        else:
            old_smoothlength_j = current_smoothlength_j
            i += 1
    
    print("WARNING: Failure to converge in newton_new_h.")
    return bisection_h(j, position_arr)

def newton_smoothlength_recursive(j, position_arr, old_smoothlength_j, old_old_smoothlength_j=0):
    new_smoothlength_j = newton_smoothlength_iteration(j, position_arr, old_smoothlength_j)
    
# Can't get the convergence failure warning to work, it goes off every time. 
    '''
    if smoothlength_variation(new_smoothlength_j, old_smoothlength_j) > smoothlength_variation(old_smoothlength_j, old_old_smoothlength_j):
        print("Error: Failure to Converge in newton_smoothlength_recursive")
        return
    '''    
    
    if smoothlength_variation(new_smoothlength_j, old_smoothlength_j) < SMOOTHLENGTH_VARIATION_TOLERANCE:
        return new_smoothlength_j
    
    else:
        new_smoothlength_j = newton_smoothlength_while(j, position_arr, new_smoothlength_j, old_smoothlength_j)
        return new_smoothlength_j

# We may be able to use new_h = old_h * (1 + zeta/(3 * density * omega))
def newton_new_h(old_h, zeta, zetap):
    new_h = old_h - zeta / zetap
    return new_h

def newton_smoothlength_arr(position_arr, old_smoothlength_arr):
    smoothlength_arr = np.zeros(position_arr.shape[0])
    for i in range(len(smoothlength_arr)):
        smoothlength_arr[i] = newton_smoothlength_while(
                                i, position_arr, old_smoothlength_arr[i])
    
    if np.any(smoothlength_arr < 0):
        raise RuntimeError("Obtained negative smoothing length.")
    return smoothlength_arr


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
    return (ADIABATIC_INDEX - 1) * energy * density * JOULE_IN_ASTRONOMICAL # added the constant to convert the energy from Joules in SI to the required astronomical units

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


# Integration and Simulation
# Cossins 3.151
def var_smoothlength_leapfrog(pos0, vel0, energy0, h0, dt):
    """
    Leap-frogging in the manner of equations 3.152 to 3.155 in Cossins.
    """
    den0 =      var_density_arr(h0)
    omega0 =    omega_arr(pos0, den0, h0)
    xi0 =       xi_arr(den0, h0, pos0)
    press0 =    pressure_arr(energy0, den0)
    energy_rate0 = energy_rate_arr(pos0, vel0, press0, den0, h0, omega0)
    acc0 =      acceleration_arr(pos0, den0, vel0, press0, h0, omega0, xi0)

    # MF: Need to check memory usage and whether we need to overwrite
    # the "0" arrays when writting the "mid" arrays, to save memory.
    pos_mid = pos0 + vel0 * 0.5 * dt
    vel_mid = vel0 + acc0 * 0.5 * dt
    energy_mid = energy0 + energy_rate0 * 0.5 * dt
    if np.any(energy_mid < 0):
        raise RuntimeError("Obtained negative energy.")
    h_mid = newton_smoothlength_arr(pos_mid, h0)
    den_mid = var_density_arr(h_mid)
    omega_mid = omega_arr(pos_mid, den_mid, h_mid)
    xi_mid = xi_arr(den_mid, h_mid, pos_mid)
    press_mid = pressure_arr(energy_mid, den_mid)
    energy_rate_mid = energy_rate_arr(pos_mid, vel_mid, press_mid, den_mid, h_mid, omega_mid)
    acc_mid = acceleration_arr(pos_mid, den_mid, vel_mid, press_mid, h_mid, omega_mid, xi_mid)

    vel1 = vel0 + acc_mid * dt
    pos1 = pos0 + 0.5 * (vel0 + vel1) * dt
    energy1 = energy0 + energy_rate_mid * dt
    if np.any(energy1 < 0):
        raise RuntimeError("Obtained negative energy.")
    # Might as well use h_mid as a guess. 
    h1 = newton_smoothlength_arr(pos1, h_mid)

    return pos1, vel1, energy1, h1


def var_smoothlength_sim(time_arr, positions0, velocities0, energies0, smoothlength_approx):
    """Performs the time integration.

    Arguments
    ---------
    time_arr : ndarray(nt)
        The times at which to compute the evolved system. nt is the number of time steps.
        Currently assumed to be uniformly spaced.
        TODO: we currently don't use this except to get dt and nt. we should have these as our inputs then.
    positions0 : ndarray(N, 3)
        Initial positions. N is the number of particles. All arrays are expected to be in 3D.
    velocities0 : ndarray(N, 3)
        Initial velocities.
    energies0 : ndarray(N)
        Initial internal energies.
    smoothlength_approx : ndarray(N)
        Initial guess for the smoothing lengths. Does not have to be precise, 
        as initial positions will eventually determine the starting values for
        smoothing lengths and densities.

    Returns
    -------
    positions : ndarray(nt, N, 3)
        Positions.
    velocities : ndarray(nt, N, 3)
        Velocities.
    energies : ndarray(nt, N)
        Internal energies.
    """
    dt = time_arr[1] - time_arr[0]
    Nt = len(time_arr)
    N = len(energies0)
    # MF: The best order is (nt, N, 3), as we want all dimmensions allways
    # then all particles, and never all times.
    # See for details:
    # https://agilescientific.com/blog/2018/12/28/what-is-the-fastest-axis-of-an-array
    positions_with_time = np.zeros((Nt, N, 3))
    velocities_with_time = np.zeros((Nt, N, 3))
    energies_with_time = np.zeros((Nt, N))
    smoothlengths_with_time = np.zeros((Nt, N))

    # Assign initial conditions
    positions_with_time[0, :, :] = positions0
    velocities_with_time[0, :, :] = velocities0
    energies_with_time[0, :] = energies0
    # Get the first smoothing length from an initial guess and the initial positions.
    smoothlengths_with_time[0, :] = newton_smoothlength_arr(
                                        positions_with_time[0, :, :],
                                        smoothlength_approx)

    # MF: We might not be able to keep all the data like this.
    # Think of a scheme to write to disk, maybe every many time steps.
    # We could also just keep positions. No one is going to plot velocities?
    # Energies might be interesting though?
    for t in range(Nt-1):
        print("time step:", t, "of", Nt)
        (positions_with_time[t+1, :, :],
         velocities_with_time[t+1, :, :],
         energies_with_time[t+1, :],
         smoothlengths_with_time[t+1, :]) = var_smoothlength_leapfrog(
                                                positions_with_time[t, :, :],
                                                velocities_with_time[t, :, :],
                                                energies_with_time[t, :],
                                                smoothlengths_with_time[t, :],
                                                dt)

    return (positions_with_time, velocities_with_time,
            energies_with_time, smoothlengths_with_time)


# MF: The static leapfrog and simulation functions are probably broken now.
# Should decide whether to keep (and fix) them or not.
def static_smoothlength_leapfrog(pos0, vel0, energy0, dt):
    static_smoothlength_arr = DEFAULT_SMOOTHLENGTH*np.ones(pos0.shape[0])

    den0 =      density_arr(pos0, static_smoothlength_arr)
    press0 =    pressure_arr(energy0, den0)
    acc0 =      acceleration_arr(pos0, den0, press0, static_smoothlength_arr)
    pos1 = pos0 + vel0*dt + 0.5*acc0*dt**2

    energy1 =   energy_evolve_arr(pos0, vel0, energy0, press0, den0, static_smoothlength_arr, dt)
    den1 =      density_arr(pos1, static_smoothlength_arr)
    press1 =    pressure_arr(energy1, den1)
    acc1 =      acceleration_arr(pos1, den1, press1, static_smoothlength_arr)
    vel1 = vel0 + 0.5*(acc0 + acc1)*dt

    return pos1, vel1, energy1

def static_smoothlength_sim(time, positions_with_time, velocities_with_time, energies_with_time):
    dt = time[1] - time[0]

    for t in range(len(time)-1):
        positions_with_time[:, :, t+1], velocities_with_time[:, :, t+1], energies_with_time[:, t+1] = static_smoothlength_leapfrog(positions_with_time[:, :, t], 
                                                                                                                        velocities_with_time[:, :, t], 
                                                                                                                        energies_with_time[:, t], 
                                                                                                                        dt)

# RK2 is wrong; should be using new position_arr to calculate a1. We aren't using it anyways. 
def rk2(r0, position_arr, v0, dt):

    a0 = acceleration(r0, position_arr)

    r1 = r0 + v0 * dt/2
    v1 = v0 + a0 * dt/2

    a1 = acceleration(r1, position_arr)

    r2 = r0 + v1 * dt
    v2 = v0 + a1 * dt

    return r2, v2
