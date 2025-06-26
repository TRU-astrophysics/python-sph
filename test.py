import numpy as np

pos = np.load("C:\\Users\\jgilc\\OneDrive\\Documents\\GitHub\\python-sph\\temp\\pos0.npy")
L = 2.3536  # Had to multiply by 100 to "see" it rotating.
total_mass = 1
total_size = 2e5
# Angular speed of a solid sphere of same size and mass.
w = 5 * L / (2 * total_mass * total_size ** 2)

# Velocities are omega * z_hat cross r_i
# IDE states this line of code is unreachable
vels = w * np.cross(np.array([0, 0, 1]), pos)
print(vels.ndim)
print(vels.shape)
print(vels[0, :])
