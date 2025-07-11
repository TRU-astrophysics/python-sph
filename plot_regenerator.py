import numpy as np
import matplotlib.pyplot as plt

Nt = np.load("temp/Nt.npy")
energy_delta = np.load("temp/energy_delta.npy")

total_time = 3e6

for n in Nt:
    t = np.linspace(0., total_time, n)  # creates a time array of size n from 0 to total_time
    total_energy = np.load(f"temp/total_energy_{n}.npy")
    # plot energy over time
    plt.plot(t, total_energy, label=f"{n} time steps")

##############################
# Setting up the graph #
##############################
plt.title("Total Energy over time, various dt")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.grid()
plt.legend()
#plt.savefig("Energy over time with different time steps.jpg")
plt.show()

#########################
# Plotting Energy Delta #
#########################
plt.scatter(Nt, energy_delta)
plt.title("Energy delta over time steps")
plt.xlabel("Time Steps")
plt.ylabel("Energy Delta")
plt.grid()
#plt.savefig("Energy Delta over time steps.jpg")
plt.show()