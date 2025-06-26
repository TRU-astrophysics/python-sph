import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.animation as animation

# 3D animation
df = 5  # decimation factor
pos = np.load("temp/pos.npy")
# pos = np.load("run_logs/run002/pos.npy")

xx = pos[::df, :, 0]
yy = pos[::df, :, 1]
zz = pos[::df, :, 2]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')
scat = ax.scatter(xx[0, :], yy[0, :], zz[0, :], marker='.', alpha=0.8)



def update(t):
    x = xx[t, :]
    y = yy[t, :]
    z = zz[t, :]
    scat._offsets3d = (x, y, z)
    return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=xx.shape[0], interval=100)

writer = animation.PillowWriter(fps=30)
ani.save(filename="SPH_3D.gif", writer="pillow")