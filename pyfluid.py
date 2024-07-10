import numpy as np
import math 

# NOTE: Need to create an array of masses if we want differing particle masses
# ALL FUNCTIONS currently assume uniform masses
PARTICLE_MASS = 1
INITIAL_H = 2
# Coupling constant is eta in Cossins's thesis (see 3.98)
COUPLING_CONST = 1.3
SMOOTHLENGTH_VARIATION_TOLERANCE = 1e-3
# G in units of years, earth masses, AU 
G = 4 * np.pi

def distance(rj, ri):
    diff = rj - ri
    dist = np.sqrt(diff.dot(diff))

    return dist

# Note: directionij gives direction vector from i to j
def directionij(rj, ri):
    if np.array_equal(rj, ri):
        return np.zeros(3)

    return (rj-ri)/distance(rj, ri)

def M4(dist, h):
    q = dist/h
    w = (1/np.pi*h**3)*np.piecewise(q, 
                                    [q >= 2, q >= 1 and q < 2, q < 1], 
                                    [0, 0.25*(2 - q)**3, 1 - 3/2*q**2 + 3/4*q**3])
    
    return w

def M4_d1(dist, h):
    q = dist/h
    wp = (1/math.pi*h**4)*np.piecewise(q, 
                                       [q >= 2, q >= 1 and q < 2, q < 1], 
                                       [0, -0.75*(2 - q)**2, 9/4 * q**2 - 3*q])
    
    return wp

def M4_d2(dist, h):
    q = dist/h
    ## should have some coeff
    wpp = np.piecewise(q, 
                       [q >= 2, q >= 1 and q < 2, q < 1], 
                       [0, 1.5*(2 - q), (1.5*(2 - q) - 6*(1 - q))])
    
    return wpp

def dellM4(rj, ri, h):
    return M4_d1(distance(rj, ri), h) * directionij(rj, ri)

# Cossins 3.107
def M4_h_derivative(dist, h):
    x = dist/h
    return -x*M4_d1(dist, h) - 3/h * M4(dist, h)


# New & improved density
# Cossins 3.98
def var_density(h):
    return PARTICLE_MASS * (COUPLING_CONST/h)**3

def var_density_arr(h_arr):
    density_arr = np.zeros(len(h_arr))
    for i in range(len(h_arr)):
        density_arr[i] = var_density(h_arr[i])
    
    return density_arr

# Cossins 3.106
def omegaj(j, hj, positions, denj):
    sum = 0
    for i in range(positions.shape[0]):
        dist = distance(positions[j], positions[i])
        sum += PARTICLE_MASS * M4_h_derivative(dist, hj)
    
    return 1 + hj/(3 * denj) * sum

# Cossins 3.112 - 3.115
def zetaj(j, hj, positions):
    return PARTICLE_MASS * (COUPLING_CONST/hj)**3 - density(positions[j], positions, hj)

# Don't really need zetaprime
def zetaprime(densityj, omegaj, hj):
    return -3 * densityj * omegaj / hj

def new_h(old_h, zeta, density, omega):
    return old_h * (1 + zeta/(3 * density * omega))

def smoothlength_variation(h_new, h_old):
    return np.abs(h_new - h_old)/h_old

def newton_h_iteration(j, positions, old_hj):
    new_denj = var_density(old_hj)
    omega = omegaj(j, old_hj, positions, new_denj)
    zeta = zetaj(j, old_hj, positions)
    new_hj = new_h(old_hj, zeta, new_denj, omega)

    return new_hj

# Use INITIAL_H as old_hj when using in code
def newton_h(j, positions, old_hj, old_old_hj):
    new_hj = newton_h_iteration(j, positions, old_hj)
    
# Can't get the convergence failure warning to work, it goes off every time. 
    '''
    if smoothlength_variation(new_hj, old_hj) > smoothlength_variation(old_hj, old_old_hj):
        print("ERROR: failure to converge")
        return
    '''    
    
    if smoothlength_variation(new_hj, old_hj) < SMOOTHLENGTH_VARIATION_TOLERANCE:
        return new_hj
    
    else:
        new_hj = newton_h(j, positions, new_hj, old_hj)
        return new_hj

def newton_h_arr(positions, old_h_arr):
    h_arr = np.zeros(positions.shape[0])
    for i in range(len(h_arr)):
        h_arr[i] = newton_h(i, positions, old_h_arr[i])
    
    return h_arr

# Old density
def density_comp(rj, ri, h):
    return PARTICLE_MASS*M4(distance(ri, rj), h)

def density(rj, positions, h):
    rho_j = 0
    
    for i in range(positions.shape[0]):
        rho_j += density_comp(rj, positions[i], h)

    return rho_j

def density_arr(positions, h_arr):
    density_arr = np.zeros(positions.shape[0])
    
    for i in range(positions.shape[0]):
        density_arr[i] = density(positions[i], positions, h_arr[i])

    return density_arr


def pressure(energy, density):
    return 2/3 * energy * density

def pressure_arr(energies, densities):
    pressure_arr = np.zeros(energies.shape[0])

    for i in range(energies.shape[0]):
        pressure_arr[i] = pressure(energies[i], densities[i])

    return pressure_arr

# PR: energy is currently just evolved w/ Euler integration - maybe not ideal?
def energy_evolve(j, positions, vels, engs, pressures, dens, h, dt):
    deltaDenj = 0
    
    for i in range(vels.shape[0]):
        deltaDenj += PARTICLE_MASS * np.dot(vels[j] - vels[i], dellM4(positions[i], positions[j], h))
    
    delta_energy = pressures[j] / dens[j]**2 * deltaDenj
    
    return engs[j] + delta_energy * dt

def energy_evolve_arr(positions, vels, engs0, pressures, dens, h_arr, dt):
    engs1 = np.zeros(positions.shape[0])
    
    for i in range(positions.shape[0]):
        engs1[i] = energy_evolve(i, positions, vels, engs0, pressures, dens, h_arr[i], dt)
    
    return engs1

# Gravity
def grav_potential(dist, h):
    x = dist/h
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [1/h * (2/3*x**2 - 3/10*x**3 + 1/10*x**5 - 7/5), 
                         1/h * (4/3*x**2 - x**3 + 3/10*x**4 - 1/30*x**5 - 8/5 + 1/(15*x)),
                         -1/dist])    

def grav_force(dist, h):
    x = dist/h
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [1/(h**2) * (4/3*x - 6/5*x**3 + 1/2*x**4), 
                         1/(h**2) * (8/3*x - 3*x**2 + 6/5*x**3 - 1/6*x**4 - 1/(15*x**2)),
                         1/(dist**2)]) 

# Note: This might be wrong. I just quickly did the calculus in my head, so it ought to be looked over more thoroughly. 
def grav_potential_h_derivative(dist, h):
    x = dist/h
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [-1/(h**2) * (2*x**2 - 12/10*x**3 + 6/10*x**5 - 7/5), 
                         -1/(h**2) * (4*x**2 - 4*x**3 + 3/2*x**4 - 1/5*x**5 - 8/5),
                         0])

def xi(j, hj, denj, positions):
    sum  = 0
    for i in range(positions.shape[0]):
        dist = distance(positions[j], positions[i])
        sum += PARTICLE_MASS * grav_potential_h_derivative(dist, hj)
    return - hj/(3*denj) * sum

def smoothed_gravity_acceleration_comp(j, i, positions, hj):
    
    if np.array_equal(positions[j], positions[i]):
        return 0

    return -G * PARTICLE_MASS * grav_force(distance(positions[j], positions[i]), hj) * directionij(positions[j], positions[i])

def basic_gravity_acceleration_comp(j, i, positions):
    
    if np.array_equal(positions[j], positions[i]):
        return 0
    
    return -directionij(positions[j], positions[i]) * G/distance(positions[j], positions[i])**2


# Acceleration
def fluid_acceleration_comp(j, i, positions, densities, pressures, hj):
    
    if np.array_equal(positions[j], positions[i]):
        return 0

    return -PARTICLE_MASS*(pressures[j]/densities[j]**2 + pressures[i]/densities[i]**2) * dellM4(positions[j], positions[i], hj)

def acceleration(j, positions, densities, pressures, h_arr):
    acc = np.array([0., 0., 0.])

    for i in range(positions.shape[0]):
        acc += (fluid_acceleration_comp(j, i, positions, densities, pressures, h_arr[j]) + smoothed_gravity_acceleration_comp(j, i, positions, h_arr[j]))

    return acc

def acceleration_arr(positions, densities, pressures, h_arr):
    acc_arr = np.zeros((positions.shape[0], 3))
    
    for i in range(positions.shape[0]):
        acc_arr[i] = acceleration(i, positions, densities, pressures, h_arr)

    return acc_arr


# Cossins 3.151
def var_h_leapfrog(pos0, vel0, energy0, h_approx, dt):
    h0 =        newton_h_arr(pos0, h_approx)
    den0 =      var_density_arr(h0)
    press0 =    pressure_arr(energy0, den0)
    acc0 =      acceleration_arr(pos0, den0, press0, h0)
    pos1 = pos0 + vel0*dt + 0.5*acc0*dt**2

    h1 =        newton_h_arr(pos1, h0) 
    # energy1 is (almost certainly) supposed to take h0, not h1! Don't change!
    energy1 =   energy_evolve_arr(pos0, vel0, energy0, press0, den0, h0, dt)
    den1 =      density_arr(pos1, h1)
    press1 =    pressure_arr(energy1, den1)
    acc1 =      acceleration_arr(pos1, den1, press1, h1)
    vel1 = vel0 + 0.5*(acc0 + acc1)*dt

    return pos1, vel1, energy1

def var_h_sim(time, positions_with_time, velocities_with_time, energies_with_time, initial_h_with_time):
    dt = time[1 - time[0]]

    for t in range(len(time)-1):
        positions_with_time[:, :, t+1], 
        velocities_with_time[:, :, t+1], 
        energies_with_time[:, t+1] = var_h_leapfrog(positions_with_time[:, :, t], 
                                                       velocities_with_time[:, :, t], 
                                                       energies_with_time[:, t], 
                                                       initial_h_with_time[:, t],
                                                       dt)

def static_h_leapfrog(pos0, vel0, energy0, dt):
    static_h_arr = INITIAL_H*np.ones(pos0.shape[0])

    den0 =      density_arr(pos0, static_h_arr)
    press0 =    pressure_arr(energy0, den0)
    acc0 =      acceleration_arr(pos0, den0, press0, static_h_arr)
    pos1 = pos0 + vel0*dt + 0.5*acc0*dt**2

    energy1 =   energy_evolve_arr(pos0, vel0, energy0, press0, den0, static_h_arr, dt)
    den1 =      density_arr(pos1, static_h_arr)
    press1 =    pressure_arr(energy1, den1)
    acc1 =      acceleration_arr(pos1, den1, press1, static_h_arr)
    vel1 = vel0 + 0.5*(acc0 + acc1)*dt

    return pos1, vel1, energy1

def static_h_sim(time, positions_with_time, velocities_with_time, energies_with_time):
    dt = time[1] - time[0]

    for t in range(len(time)-1):
        positions_with_time[:, :, t+1], 
        velocities_with_time[:, :, t+1], 
        energies_with_time[:, t+1] = static_h_leapfrog(positions_with_time[:, :, t], 
                                                       velocities_with_time[:, :, t], 
                                                       energies_with_time[:, t], 
                                                       dt)

# RK2 is wrong; should be using new positions to calculate a1. We aren't using it anyways. 
def rk2(r0, positions, v0, dt):

    a0 = acceleration(r0, positions)

    r1 = r0 + v0 * dt/2
    v1 = v0 + a0 * dt/2

    a1 = acceleration(r1, positions)

    r2 = r0 + v1 * dt
    v2 = v0 + a1 * dt

    return r2, v2
