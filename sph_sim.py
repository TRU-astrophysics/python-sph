import numpy as np
import sph_NumericalMethods as num
import sph_energy as erg
import sph_physicalmethods as phys

def static_smoothlength_sim(time, positions_with_time, velocities_with_time, energies_with_time):
    dt = time[1] - time[0]

    for t in range(len(time) - 1):
        positions_with_time[:, :, t + 1], velocities_with_time[:, :, t + 1], energies_with_time[:,
                                                                             t + 1] = num.static_smoothlength_leapfrog(
            positions_with_time[:, :, t],
            velocities_with_time[:, :, t],
            energies_with_time[:, t],
            dt)


def var_smoothlength_sim(time_arr, positions0, velocities0, energies0, smoothlength_approx):
    """Performs the time integration.

    Arguments
    ---------
    time_arr : ndarray(nt)
        The times at which to compute the evolved system. nt is the number of time steps.
        Currently assumed to be uniformly spaced.
        TODO: we currently don't use this except to get dt and nt. we should have these as our inputs then.
    positions0 : ndarray(N, 3)
        Initial positions. N is the number of particles. All arrays are expected to be in 3D.
    velocities0 : ndarray(N, 3)
        Initial velocities.
    energies0 : ndarray(N)
        Initial internal energies.
    smoothlength_approx : ndarray(N)
        Initial guess for the smoothing lengths. Does not have to be precise, 
        as initial positions will eventually determine the starting values for
        smoothing lengths and densities.

    Returns
    -------
    positions : ndarray(nt, N, 3)
        Positions.
    velocities : ndarray(nt, N, 3)
        Velocities.
    energies : ndarray(nt, N)
        Internal energies.
    TODO: output phi, test to see if every 5 or 10 timesteps are frequent enough to get meaningful results
    
    """
    dt = time_arr[1] - time_arr[0]
    Nt = len(time_arr)
    N = len(energies0)
    # MF: The best order is (nt, N, 3), as we want all dimensions allways
    # then all particles, and never all times.
    # See for details:
    # https://agilescientific.com/blog/2018/12/28/what-is-the-fastest-axis-of-an-array
    positions_with_time = np.zeros((Nt, N, 3))
    velocities_with_time = np.zeros((Nt, N, 3))
    energies_with_time = np.zeros((Nt, N))
    smoothlengths_with_time = np.zeros((Nt, N))

    # For testing, adding a total energy calculator
    total_energy = np.zeros(Nt)

    # Assign initial conditions
    positions_with_time[0, :, :] = positions0
    velocities_with_time[0, :, :] = velocities0
    energies_with_time[0, :] = energies0
    # Get the first smoothing length from an initial guess and the initial positions.
    smoothlengths_with_time[0, :] = num.newton_smoothlength_arr(
        positions_with_time[0, :, :],
        smoothlength_approx)

    # MF: We might not be able to keep all the data like this.
    # Think of a scheme to write to disk, maybe every many time steps.
    # We could also just keep positions. No one is going to plot velocities?
    # Energies might be interesting though?
    for t in range(Nt - 1):
        print("time step:", t, "of", Nt)
        (positions_with_time[t + 1, :, :],
         velocities_with_time[t + 1, :, :],
         energies_with_time[t + 1, :],
         smoothlengths_with_time[t + 1, :]) = num.var_smoothlength_leapfrog(
            positions_with_time[t, :, :],
            velocities_with_time[t, :, :],
            energies_with_time[t, :],
            smoothlengths_with_time[t, :],
            dt)

        # added quantities for total energy calculations
        density = phys.var_density_arr(smoothlengths_with_time[t,:])

        pressure = phys.pressure_arr(energies_with_time[t,:], density )
        # TODO I need to figure out how to use the position array
        # with the grav_kernal function in sph_gravity
        total_energy[t] = erg.total_Energy(
            positions_with_time[t,:,:],
            velocities_with_time[t,:,:],
            pressure,
            density,
            smoothlengths_with_time[t,:]
        )

    return (positions_with_time, velocities_with_time,
            energies_with_time, smoothlengths_with_time,
            total_energy)
