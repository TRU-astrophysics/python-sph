import numpy as np
import math 

mi = 1
h = 1
# G in units of 100,000 yrs, 1,000 earth masses, AU 
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


def M4(dist):
    q = dist/h
    w = (1/np.pi*h**3)*np.piecewise(q, 
                                    [q >= 2, q >= 1 and q < 2, q < 1], 
                                    [0, 0.25*(2 - q)**3, 1 - 3/2*q**2 + 3/4*q**3])
    
    return w

def M4_d1(dist):
    q = dist/h
    wp = (1/math.pi*h**4)*np.piecewise(q, 
                                       [q >= 2, q >= 1 and q < 2, q < 1], 
                                       [0, -0.75*(2 - q)**2, 9/4 * q**2 - 3*q])
    
    return wp

def M4_d2(dist):
    q = dist/h
    ## should have some coeff
    wpp = np.piecewise(q, 
                       [q >= 2, q >= 1 and q < 2, q < 1], 
                       [0, 1.5*(2 - q), (1.5*(2 - q) - 6*(1 - q))])
    
    return wpp

def dellM4(rj, ri):  
    return M4_d1(distance(rj, ri)) * directionij(rj, ri)


def density_comp(rj, ri):
    return mi*M4(distance(ri, rj))

def density(rj, positions):
    rho_j = 0
    
    for i in range(positions.shape[0]):
        rho_j += density_comp(rj, positions[i])

    return rho_j

def density_arr(positions):
    density_arr = np.zeros(positions.shape[0])
    
    for i in range(positions.shape[0]):
        density_arr[i] = density(positions[i], positions)

    return density_arr


def pressure(energy, density):
    return 2/3 * energy * density

def pressure_arr(energies, densities):
    pressure_arr = np.zeros(energies.shape[0])

    for i in range(energies.shape[0]):
        pressure_arr[i] = pressure(energies[i], densities[i])

    return pressure_arr

# PR: energy is currently just evolved w/ Euler integration - maybe not ideal?
def energy_evolve(j, positions, vels, engs, pressures, dens, dt):
    deltaDenj = 0
    
    for i in range(vels.shape[0]):
        deltaDenj += mi * np.dot(vels[j] - vels[i], dellM4(positions[i], positions[j]))
    
    delta_energy = pressures[j] / dens[j]**2 * deltaDenj
    
    return engs[j] + delta_energy * dt

def energy_evolve_arr(positions, vels, engs0, pressures, dens, dt):
    engs1 = np.zeros(positions.shape[0])
    
    for i in range(positions.shape[0]):
        engs1[i] = energy_evolve(i, positions, vels, engs0, pressures, dens, dt)
    
    return engs1


def grav_potential(dist):
    x = dist/h
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [1/h * (2/3*x**2 - 3/10*x**3 + 0.1*x**5 -7/5), 
                         1/h * (4/3*x**2 - x**3 + 3/10*x**4 - 1/30*x**5 - 8/5 + 1/(15*x)),
                         -1/dist])    

def grav_force(dist):
    x = dist/h
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [1/h**2 * (4/3*x - 6/5*x**3 + 0.5*x**4), 
                         1/h**2 * (8/3*x - 3*x**2 + 6/5*x**3 - 1/6*x**4 - 1/(15*x**2)),
                         1/dist**2]) 

def smoothed_gravity_acceleration_comp(j, i, positions):
    
    if np.array_equal(positions[j], positions[i]):
        return 0

    return -G * mi * grav_force(distance(positions[j], positions[i])) * directionij(positions[j], positions[i])

def basic_gravity_acceleration_comp(j, i, positions):
    
    if np.array_equal(positions[j], positions[i]):
        return 0
    
    return -directionij(positions[j], positions[i]) * G/distance(positions[j], positions[i])**2

def fluid_acceleration_comp(j, i, positions, densities, pressures):
    
    if np.array_equal(positions[j], positions[i]):
        return 0

    return -mi*(pressures[j]/densities[j]**2 + pressures[i]/densities[i]**2) * dellM4(positions[j], positions[i])

def acceleration(j, positions, densities, pressures):
    acc = np.array([0., 0., 0.])

    for i in range(positions.shape[0]):
        acc += (fluid_acceleration_comp(j, i, positions, densities, pressures) + smoothed_gravity_acceleration_comp(j, i, positions))

    return acc

def acceleration_arr(positions, densities, pressures):
    acc_arr = np.zeros((positions.shape[0], 3))
    
    for i in range(positions.shape[0]):
        acc_arr[i] = acceleration(i, positions, densities, pressures)

    return acc_arr

# From equation 3.151 in Pete Cossin's thesis
def leapfrog(pos0, vel0, energy0, dt):

    den0 =      density_arr(pos0)
    press0 =    pressure_arr(energy0, den0)
    acc0 =      acceleration_arr(pos0, den0, press0)
    pos1 = pos0 + vel0*dt + 0.5*acc0*dt**2

    energy1 =   energy_evolve_arr(pos0, vel0, energy0, press0, den0, dt)
    den1 =      density_arr(pos1)
    press1 =    pressure_arr(energy1, den1)
    acc1 =      acceleration_arr(pos1, den1, press1)
    vel1 = vel0 + 0.5*(acc0 + acc1)*dt

    return pos1, vel1, energy1

# This is wrong; should be using new positions to calculate a1
def rk2(r0, positions, v0, dt):

    a0 = acceleration(r0, positions)

    r1 = r0 + v0 * dt/2
    v1 = v0 + a0 * dt/2

    a1 = acceleration(r1, positions)

    r2 = r0 + v1 * dt
    v2 = v0 + a1 * dt

    return r2, v2

def sim(time, positions_with_time, velocities_with_time, energies_with_time):
    dt = time[1] - time[0]

    for t in range(len(time)-1):
        positions_with_time[:, :, t+1], velocities_with_time[:, :, t+1], energies_with_time[:, t+1] = leapfrog(positions_with_time[:, :, t], velocities_with_time[:, :, t], energies_with_time[:, t], dt)
