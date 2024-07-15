import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pyfluid as sph

# Plot of M4 and its first two derivatives
'''
d = np.arange(0, 5, 0.01)
m4 = np.zeros(len(d))
m4d1 = np.zeros(len(d))
m4d2 = np.zeros(len(d))

for i in range(0, len(d)-1):
    m4[i] = sph.M4(d[i])
    m4d1[i] = sph.M4_d1(d[i])
    m4d2[i] = sph.M4_d2(d[i])
    i += 1

plt.plot(d, m4, 'r', d, m4d1, 'b', d, m4d2, 'g')
plt.show()
'''

# NOTE: positions defined with [particle][dimension][time]

# Initial definitions
n = 20
t = np.arange(0., 20, 0.1)
pos = np.zeros((n, 3, len(t)))
xs = pos[:, 0, :]
ys = pos[:, 1, :]
zs = pos[:, 2, :]
vels = np.zeros((n, 3, len(t)))
engs = np.ones((n, len(t)))*7

# Grid of particles
'''
for i in range(n/2):
    pos[i, 0, 0] = i
for i in range(n/2, n):
    pos[i, 0, 0] = i - n/2
    pos[i, 1, 0] = 1
'''

# Random distribution
rng = np.random.default_rng()
for i in range(n):
    # change third entry of this array to be the same as the first two for a 3D distribution
    pos[i, :, 0] = np.array([rng.random()*10, 
                             rng.random()*10, 
                             0])


sph.sim(t, pos, vels, engs)


# 2D animation

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
scat = ax.scatter(xs[:, 0], ys[:, 0])

def update(t):
    x = xs[:, t]
    y = ys[:, t]

    data = np.stack([x, y]).T
    scat.set_offsets(data)

    return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t), interval=100)
#ani.save(filename="2D_SPH.gif", writer="pillow")
plt.show()



# 3D animation
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
ax.set_zlim(-5, 15)
scat = ax.scatter(xs[:, 0], ys[:, 0], zs[:, 0])

def update(t):
    x = xs[:, t]
    y = ys[:, t]
    z = zs[:, t]

    scat._offsets3d = ([x, y, z])

    return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t), interval=50)
#ani.save(filename="3D_SPH.gif", writer="pillow")
plt.show()
'''