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

# Bisection
BISECTION_ITERATION_LIMIT = 100
BISECTION_TOLERANCE = 1e-10
INITIAL_BISECTION_SMOOTHLENGTH_A = 1e3
INITIAL_BISECTION_SMOOTHLENGTH_B = 1e-3

# G in units of years, solar masses, AU
# 39.4324 would be slightly more accurate
G = 4 * np.pi**2

# Boltzmann's constant in units of years, solar masses, AU, Kelvin
K_BOLTZMANN = 3.08e-61

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

# NOTE: We may be able to find zetaprime with -3 * density_j * omega_j / smoothlength_j
def zetaprime(density_j, omega_j, smoothlength_j):
    mj = PARTICLE_MASS
    return -3 * density_j / smoothlength_j * (omega_j - 1) - 3 * mj * (COUPLING_CONST / smoothlength_j)**3 / smoothlength_j

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
    
    print("ERROR: Failure to converge in newton_new_h.")
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
        smoothlength_arr[i] = newton_smoothlength_while(i, position_arr, old_smoothlength_arr[i])
    
    return smoothlength_arr


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
    return 2/3 * energy * density

def pressure_arr(energy_arr, density_arr):
    pressure_arr = np.zeros(energy_arr.shape[0])

    for i in range(energy_arr.shape[0]):
        pressure_arr[i] = pressure(energy_arr[i], density_arr[i])

    return pressure_arr

# PR: energy is currently just evolved w/ Euler integration - maybe not ideal?
def energy_evolve(j, position_arr, velocity_arr, energy_arr, pressure_arr, density_arr, smoothlength_j, dt):
    density_change = 0
    
    for i in range(velocity_arr.shape[0]):
        density_change += PARTICLE_MASS * np.dot(velocity_arr[j] - velocity_arr[i], dellM4(position_arr[j], position_arr[i], smoothlength_j))
    
    energy_change = pressure_arr[j] / density_arr[j]**2 * density_change
    
    return energy_arr[j] + energy_change * dt

def energy_evolve_arr(position_arr, vels, energies_initial, pressure_arr, density_arr, smoothlength_arr, dt):
    energies_final = np.zeros(position_arr.shape[0])
    
    for i in range(position_arr.shape[0]):
        energies_final[i] = energy_evolve(i, position_arr, vels, energies_initial, pressure_arr, density_arr, smoothlength_arr[i], dt)
    
    return energies_final


# Gravity
def grav_potential(dist, smoothlength):
    x = dist/smoothlength
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [1/smoothlength * (2/3*x**2 - 3/10*x**3 + 1/10*x**5 - 7/5), 
                         1/smoothlength * (4/3*x**2 - x**3 + 3/10*x**4 - 1/30*x**5 - 8/5 + 1/(15*x)),
                         -1/dist])    

def grav_force(dist, smoothlength):
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


# Acceleration
def fluid_acceleration_comp(j, i, position_arr, density_arr, pressure_arr, smoothlength_arr, omega_arr):
    
    if np.array_equal(position_arr[j], position_arr[i]):
        return 0

    return -PARTICLE_MASS * (pressure_arr[j]/(omega_arr[j] * density_arr[j]**2) * dellM4(position_arr[j], position_arr[i], smoothlength_arr[j])
                           + pressure_arr[i]/(omega_arr[i] * density_arr[i]**2) * dellM4(position_arr[j], position_arr[i], smoothlength_arr[i]))

def acceleration(j, position_arr, density_arr, pressure_arr, smoothlength_arr, omega_arr, xi_arr):
    acc = np.array([0., 0., 0.])

    for i in range(position_arr.shape[0]):
        acc += (fluid_acceleration_comp(j, i, position_arr, density_arr, pressure_arr, smoothlength_arr, omega_arr) 
              + smoothed_gravity_acceleration_comp(j, i, position_arr, smoothlength_arr)
              + smoothed_gravity_correction_comp(j, i, position_arr, smoothlength_arr, omega_arr, xi_arr))

    return acc

def acceleration_arr(position_arr, density_arr, pressure_arr, smoothlength_arr, omega_arr, xi_arr):
    acc_arr = np.zeros((position_arr.shape[0], 3))
    
    for i in range(position_arr.shape[0]):
        acc_arr[i] = acceleration(i, position_arr, density_arr, pressure_arr, smoothlength_arr, omega_arr, xi_arr)

    return acc_arr


# Integration and Simulation
# Cossins 3.151
def var_smoothlength_leapfrog(pos0, vel0, energy0, smoothlength_approx, dt):
    h0 =        newton_smoothlength_arr(pos0, smoothlength_approx)
    den0 =      var_density_arr(h0)
    omega0 =    omega_arr(pos0, den0, h0)
    xi0 =       xi_arr(den0, h0, pos0)
    press0 =    pressure_arr(energy0, den0)
    acc0 =      acceleration_arr(pos0, den0, press0, h0, omega0, xi0)
    pos1 = pos0 + vel0*dt + 0.5*acc0*dt**2

    h1 =        newton_smoothlength_arr(pos1, h0) 
    # energy1 is (almost certainly) supposed to take h0, not h1! Don't change!
    energy1 =   energy_evolve_arr(pos0, vel0, energy0, press0, den0, h0, dt)
    den1 =      density_arr(pos1, h1)
    omega1 =    omega_arr(pos1, den1, h1)
    xi1 =       xi_arr(den1, h1, pos1)
    press1 =    pressure_arr(energy1, den1)
    acc1 =      acceleration_arr(pos1, den1, press1, h1, omega1, xi1)
    vel1 = vel0 + 0.5*(acc0 + acc1)*dt

    return pos1, vel1, energy1, h1

def var_smoothlength_sim(time, positions_with_time, velocities_with_time, energies_with_time, smoothlengths_with_time):
    dt = time[1] - time[0]

    for t in range(len(time)-1):
        positions_with_time[:, :, t+1], velocities_with_time[:, :, t+1], energies_with_time[:, t+1], smoothlengths_with_time[:, t+1] = var_smoothlength_leapfrog(positions_with_time[:, :, t], 
                                                                                                                                                                 velocities_with_time[:, :, t], 
                                                                                                                                                                 energies_with_time[:, t], 
                                                                                                                                                                 smoothlengths_with_time[:, t],
                                                                                                                                                                 dt)

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
