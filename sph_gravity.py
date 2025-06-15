
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
    
