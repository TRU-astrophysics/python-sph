import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pyfluid as sph

# NOTE: positions defined with [particle][dimension][time]

# Initial definitions
n = 5
t = np.arange(0., 10, 0.1)
pos = np.zeros((n, 3, len(t)))
xs = pos[:, 0, :]
ys = pos[:, 1, :]
zs = pos[:, 2, :]
vels = np.zeros((n, 3, len(t)))
engs = np.ones((n, len(t)))*10
initial_h = np.ones((n, len(t)))

# Randomly distributing particles
rng = np.random.default_rng()
for i in range(n):
    pos[i, :, 0] = np.array([rng.random()*10, 
                             rng.random()*10, 
                             0])

pos[0, :, 0] = [10, 0, 0]
vels[0, :, 0] = [0, 10, 0]

pos[1, :, 0] = [0, 10, 0]
vels[1, :, 0] = [-10, 0, 0]

# Sim
sph.var_h_sim(t, pos, vels, engs, initial_h)


# 2D animation
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim(-25, 35)
ax.set_ylim(-25, 35)
scat = ax.scatter(xs[:, 0], ys[:, 0])

def update(t):
    x = xs[:, t]
    y = ys[:, t]

    data = np.stack([x, y]).T
    scat.set_offsets(data)

    return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t), interval=100)
#ani.save(filename="SPH_rotationtesting.gif", writer="pillow")
plt.show()
