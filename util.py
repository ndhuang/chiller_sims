import numpy as np
from matplotlib import pyplot as pl


def reduce_sim(T, state, dt=1):
    '''
    Reduce simulation timestreams to a few statistics: the cycle time,
    peak-to-peak temperature range, and mean temperature.

    Parameters
    ----------
    T : array-like
        The temperature to reduce.  Usually the supply temperature that enters
        the load.
    state : array-like
        The compressor state.
    dt : float
        The time step of the simulation
    '''
    cycle_inds = np.where(np.diff(state) < 0)[0]
    n_cycles = len(cycle_inds)
    t_cycle = np.mean(np.diff(cycle_inds)) * dt

    T_min = 0
    T_max = 0
    prev = 0
    for i in cycle_inds:
        T_min += np.min(T[prev:i])
        T_max += np.max(T[prev:i])
        prev = i
    dT = (T_max - T_min) / n_cycles
    return t_cycle, dT, np.mean(T)


def plot_sim(supply, ret=None, refrigerant=None, dt=1):
    '''
    Plot the water supply and return temperatures, and the temperature of the
    chiller refrigerant.  The first argument is usually the dictionary output
    from the simulation, but the temperature TOD can also be passed individually.

    Parameters
    ----------
    supply : array-like or dict
        If a dict, the temperatures to plot are pulled from the dictionary.
        The dict should have keys `'return'` (for the water return temperature),
        `'supply'`, for the supply temperature, and `'refrigerant`' for the
        chiller refrigerant temperature.  In this case, the `ret`, and
        `regfrigerant` arguments are ignored.
        If an array, this argument should contain the water supply temperature.
    ret : array-like
        The water return temperature
    refrigerant : array-like
        The refrigerant temperature
    dt : float
        The time step of the simulation
    '''
    if hasattr(supply, 'keys'):
        # we got a full dict of sim results as the first argument, attempt to
        # unload the relevant data
        ret = supply['return']
        refrigerant = supply['refrigerant']
        supply = supply.get('supply', supply['chiller'])
    x = np.arange(len(ret)) * dt / 60
    pl.plot(x, supply, label='$T_{\\rm supply}$')
    pl.plot(x, ret, label='$T_{\\rm return}$')
    pl.plot(x, refrigerant, label='$T_{\\rm ch}$')
    pl.xlabel('Time [minutes]')
    pl.ylabel('Temp [K]')
    pl.legend()


def plot_reduced(x, cycle_time, dT, meanT=None, axes=[], xlabel=''):
    '''
    Plot the reduced statistics for a set of simulations.

    Parameters
    ----------
    x : array-like
        The values of the x axis, usually whatever was varied in the simulations.
    cycle_time : array-like
        The cycle time of each simulation, in seconds.
    dT : array-like
        The peak-to-peak temperature variation, in K
    meanT : array-like, optional
        The mean temperatire in K.  If `None`, only plot cycle time and Delta T.
    axes : list
        If not an empty list, use these axes; otherwise make a new figure.
    xlabel : str
        The label of the x axis

    Returns the axes of the plots.
    '''
    if len(axes) == 0:
        nax = 2 if meanT is None else 3
        fig, axes = pl.subplots(
            nax, 1, sharex=True, constrained_layout=True, figsize=(6, 6)
        )
    axes[0].plot(x, cycle_time / 60, marker='.')
    axes[0].set_ylabel('Cycle Time [min]')
    axes[1].plot(x, dT, marker='.')
    axes[1].set_ylabel('$\\Delta T_{\\rm supply}$ [K]')
    if meanT is not None:
        axes[2].plot(x, meanT, marker='.')
        axes[2].set_ylabel('$\\overline{T}_{\\rm supply}$')
    if len(xlabel) > 0:
        axes[-1].set_xlabel(xlabel)
    return axes
