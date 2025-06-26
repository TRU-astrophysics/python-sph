import numpy as np
import sph_physicalmethods as phys

# Physical Constants
# G in units of years, solar masses, AU
# 39.4324 would be slightly more accurate
# MF: Hmm, I got 39.4227
#G = 4 * np.pi**2
G = 39.4227


# Gravity

def grav_kernal(dist, smoothlength):
    """
    Cossins eq. 3.149
    """
    x = dist/smoothlength
    return np.piecewise(x, 
                        [x < 1,
                         1 <= x < 2,
                         x >= 2],
                        [1/smoothlength * (2/3*x**2 - 3/10*x**3 + 1/10*x**5 - 7/5), 
                         1/smoothlength * (4/3*x**2 - x**3 + 3/10*x**4 - 1/30*x**5 - 8/5 + 1/(15*x)),
                         -1/dist])    

    

def grav_kernal_spacial_derivative(dist, smoothlength):
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


# derived quantity, not given in paper, confirmed correct 16/06/2025
def grav_kernal_smoothlength_derivative(dist, smoothlength):
    x = dist/smoothlength
    return np.piecewise(x, 
                        [x < 1,
                         x >= 1 and x < 2,
                         x >= 2],
                        [-1/(smoothlength**2) * (2*x**2 - 12/10*x**3 + 6/10*x**5 - 7/5), 
                         -1/(smoothlength**2) * (4*x**2 - 4*x**3 + 3/2*x**4 - 1/5*x**5 - 8/5),
                         0])


# Cossins Eq. 3.132
def xi(j, position_arr, density_j, smoothlength_j):
    sum  = 0
    for i in range(position_arr.shape[0]):
        dist = phys.distance(position_arr[j], position_arr[i])
        # TODO: numpy.piecewise can get an array for input. We should use it instead of this for loop!
        sum += phys.PARTICLE_MASS * grav_kernal_smoothlength_derivative(dist, smoothlength_j)
    #dh/d rho = -h/3rho from equation 3.103 
    return -smoothlength_j/(3*density_j) * sum

def xi_arr(density_arr, smoothlength_arr, position_arr):
    xi_arr = np.zeros(position_arr.shape[0])

    for j in range(position_arr.shape[0]):
        xi_arr[j] = xi(j, position_arr, density_arr[j], smoothlength_arr[j])
    
    return xi_arr
    
