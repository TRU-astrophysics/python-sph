import numpy as np
import sph_physicalmethods as phys
import sph_gravity as grav
import sph_energy as erg

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

DEFAULT_SMOOTHLENGTH = 2

#SPH Specific Methods#

def M4(dist, smoothlength):
    q = dist / smoothlength
    w = 1 / (np.pi * smoothlength ** 3) * np.piecewise(q,
                                                       [q >= 2, q >= 1 and q < 2, q < 1],
                                                       [0, 0.25 * (2 - q) ** 3, 1 - 3 / 2 * q ** 2 + 3 / 4 * q ** 3])

    return w


def M4_d1(dist, smoothlength):
    q = dist / smoothlength
    wp = 1 / (np.pi * smoothlength ** 4) * np.piecewise(q,
                                                        [q >= 2, q >= 1 and q < 2, q < 1],
                                                        [0, -0.75 * (2 - q) ** 2, 9 / 4 * q ** 2 - 3 * q])

    return wp


def M4_d2(dist, smoothlength):
    q = dist / smoothlength
    # should have some coeff
    wpp = np.piecewise(q,
                       [q >= 2, q >= 1 and q < 2, q < 1],
                       [0, 1.5 * (2 - q), (1.5 * (2 - q) - 6 * (1 - q))])

    return wpp


def dellM4(rj, ri, smoothlength):
    return M4_d1(phys.distance(rj, ri), smoothlength) * phys.direction_i_j(rj, ri)


# Cossins 3.107
def M4_smoothlength_derivative(dist, smoothlength):
    x = dist / smoothlength
    return -x * M4_d1(dist, smoothlength) - 3 / smoothlength * M4(dist, smoothlength)

def smoothlength_variation(new_h, old_h):
    return np.abs(new_h - old_h) / np.abs(old_h)


# Cossins 3.106
def omega_j(j, position_arr, density_j, smoothlength_j):
    sum = 0
    for i in range(position_arr.shape[0]):
        dist = phys.distance(position_arr[j], position_arr[i])
        sum += phys.PARTICLE_MASS * M4_smoothlength_derivative(dist, smoothlength_j)

    return 1 + smoothlength_j / (3 * density_j) * sum


def omega_arr(position_arr, density_arr, smoothlength_arr):
    omega_arr = np.zeros(position_arr.shape[0])

    for j in range(position_arr.shape[0]):
        omega_arr[j] = omega_j(j, position_arr, density_arr[j], smoothlength_arr[j])

    return omega_arr

# Cossins 3.112 - 3.115
def zeta_j(smoothlength_j, density_j):
    zeta = phys.var_density(smoothlength_j) - density_j
    return zeta


# NOTE: We may be able to find zetaprime with
# -3 * density_j * omega_j / smoothlength_j
def zetaprime(density_j, omega_j, smoothlength_j):
    mj = phys.PARTICLE_MASS
    return (- 3 * density_j / smoothlength_j * (omega_j - 1)
            - 3 * mj * (phys.COUPLING_CONST / smoothlength_j) ** 3 / smoothlength_j)


# Numerical Methods
# creating a general bisecion method, will also make a generalized
# Newton's Method
def bisection(a,b,f):
    h_a = a
    h_b = b
    i = 0
    if f(h_a)*f(h_b)>0:
        print("ERROR: root out of bounds")
    
    while i<BISECTION_ITERATION_LIMIT:
        i +=1
    #return 0
    
def newton():
    return 0


# Bisection should perhaps be broken down into a series of smaller functions, like newton's method is. 
def bisection_h(j, position_arr):
    h_a = INITIAL_BISECTION_SMOOTHLENGTH_A
    h_b = INITIAL_BISECTION_SMOOTHLENGTH_B
    i = 0

    while i < BISECTION_ITERATION_LIMIT:
        density_a = phys.density(j, position_arr, h_a)
        zeta_a = zeta_j(h_a, density_a)
        density_b = phys.density(j, position_arr, h_b)
        zeta_b = zeta_j(h_b, density_b)

        # zeta(h_a) * zeta(h_b) must be < 0 because one must be pos and one neg, so there is a root of zeta between
        if zeta_a * zeta_b > 0:
            print("ERROR: Root out of bisection bounds.")
        
        root_candidate = (h_a + h_b)/2
        zeta_root = zeta_j(root_candidate, phys.density(j, position_arr, root_candidate))
        
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

    old_density_j = phys.density(j, position_arr, old_smoothlength_j)
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
    

######################################################################
# Integration and Simulation
######################################################################
# Cossins 3.151
def var_smoothlength_leapfrog(pos0, vel0, energy0, h0, dt):
    """
    Leap-frogging in the manner of equations 3.152 to 3.155 in Cossins.
    """
    den0 =      phys.var_density_arr(h0)
    omega0 =    omega_arr(pos0, den0, h0)
    xi0 =       grav.xi_arr(den0, h0, pos0)
    press0 =    phys.pressure_arr(energy0, den0)
    energy_rate0 = erg.energy_rate_arr(pos0, vel0, press0, den0, h0, omega0)
    acc0 =      phys.acceleration_arr(pos0, den0, vel0, press0, h0, omega0, xi0)

    # MF: Need to check memory usage and whether we need to overwrite
    # the "0" arrays when writting the "mid" arrays, to save memory.
    pos_mid = pos0 + vel0 * 0.5 * dt
    vel_mid = vel0 + acc0 * 0.5 * dt
    energy_mid = energy0 + energy_rate0 * 0.5 * dt
    if np.any(energy_mid < 0):
        raise RuntimeError("Obtained negative energy.")
    h_mid = newton_smoothlength_arr(pos_mid, h0)
    den_mid = phys.var_density_arr(h_mid)
    omega_mid = omega_arr(pos_mid, den_mid, h_mid)
    xi_mid = grav.xi_arr(den_mid, h_mid, pos_mid)
    press_mid = phys.pressure_arr(energy_mid, den_mid)
    energy_rate_mid = erg.energy_rate_arr(pos_mid, vel_mid, press_mid, den_mid, h_mid, omega_mid)
    acc_mid = phys.acceleration_arr(pos_mid, den_mid, vel_mid, press_mid, h_mid, omega_mid, xi_mid)

    vel1 = vel0 + acc_mid * dt
    pos1 = pos0 + 0.5 * (vel0 + vel1) * dt
    energy1 = energy0 + energy_rate_mid * dt
    if np.any(energy1 < 0):
        raise RuntimeError("Obtained negative energy.")
    # Might as well use h_mid as a guess. 
    h1 = newton_smoothlength_arr(pos1, h_mid)

    return pos1, vel1, energy1, h1
    
# MF: The static leapfrog and simulation functions are probably broken now.
# Should decide whether to keep (and fix) them or not.
def static_smoothlength_leapfrog(pos0, vel0, energy0, dt):
    static_smoothlength_arr = DEFAULT_SMOOTHLENGTH*np.ones(pos0.shape[0])

    den0 =      phys.density_arr(pos0, static_smoothlength_arr)
    press0 =    phys.pressure_arr(energy0, den0)
    acc0 =      phys.acceleration_arr(pos0, den0, press0, static_smoothlength_arr)
    pos1 = pos0 + vel0*dt + 0.5*acc0*dt**2

    energy1 =   erg.energy_evolve_arr(pos0, vel0, energy0, press0, den0, static_smoothlength_arr, dt)
    den1 =      phys.density_arr(pos1, static_smoothlength_arr)
    press1 =    phys.pressure_arr(energy1, den1)
    acc1 =      phys.acceleration_arr(pos1, den1, press1, static_smoothlength_arr)
    vel1 = vel0 + 0.5*(acc0 + acc1)*dt

    return pos1, vel1, energy1