import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# NOTE: positions defined with [time][particle][dimension]

pos = np.load('pos.npy')

fig = plt.figure()
rad = 2e5
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-rad, rad)
ax.set_ylim(-rad, rad)
ax.set_zlim(-rad, rad)
scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], pos[0, :, 2])

def update(t):
    x = pos[t, :, 0]
    y = pos[t, :, 1]
    z = pos[t, :, 2]

    scat._offsets3d = ([x, y, z])

    return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(pos), interval=50)
#ani.save(filename="3D_SPH.gif", writer="pillow")
plt.show()