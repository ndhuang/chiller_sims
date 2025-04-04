import warnings
from collections import deque
import numpy as np

# specific heat J / (kg K)
C_water = 4.187e3
C_coolant = 3.559e3  # 50% glycol
# C_coolant = 3.747e3 # 40% glycol
# Both water and coolant assumed to have a density of 1 kg / l

# Units
# volume: liter
# mass: kilogram
# time: second
# temperature: Kelvin
# energy: Joule


class SerialSim:
    '''
    Framework for running a simulation containing only serial elements (i.e. no
    parallel paths).  The simulation contains a chiller at instantiation, and
    other elements are added by the user with the `add_element` method.
    Individual elements can implement parallel paths (see `SplitLoad` below).
    '''

    def __init__(self, flow, dt=1, T0=None, **chiller_kwargs):
        '''
        Initialize the simulation, including the chiller.

        Parameters
        ----------
        flow : float
            Coolant flow in kg / s
        dt : float
            Time step in s
        T0 : float, optional
            Initial temperature in K.  Generally this should not be provided by
            the user.  The automatic calculation provides an initial temperature
            that converges reasonably quickly.
        Additional keyword arguments are passed to `Chiller`.

        '''
        self.dt = dt
        self.flow = flow

        self.chiller = Chiller(T0, dt=dt, **chiller_kwargs)
        if T0 is None:
            T0 = self.chiller.T_tank
        self.T = T0
        self.elements = {}
        self.to_save = deque(('chiller', 'compressor', 'refrigerant'))
        self.order = deque()

        self.burn_in = 0
        self.i_step = 0
        self.data = {}

    def terminate(self):
        '''
        Returns `True` if it is time to end the simulation.
        '''
        return (
            self.n_cycles is not None
            and self.chiller.i_cycle >= self.n_cycles + self.burn_in
        ) or (self.n_steps is not None and self.i_step >= self.n_steps)

    def add_element(self, element, name, i=-1, save=False):
        '''
        Add an element to the simulation.  By default, the element is added to
        the end of the chain.

        Parameters
        ----------
        element : callable
            The element to add to the simulation.  It must be a callable that
            takes two arguments: the temperature of the incoming water, and its
            flow rate.  It must return the temperature of the outgoing water.
        name : str
            A useful name for the element.  If data is saved from this element,
            it will be keyed using `name`.  Must be unique.
        i : int, optional
            The position of the element.  By default, the element will be added
            to the end of the simulation.  Passing `i=0` puts the element
            immediately after the chiller.
        save : bool
            If `True`, save the temperature returned by this element at every
            time step.
        '''
        if name in self.elements:
            raise KeyError('{} already exists'.format(name))
        self.elements[name] = element
        if i >= 0:
            self.order.insert(i, name)
        else:
            self.order.append(name)
        if save:
            self.to_save.append(name)

    def _step_elements(self, T, flow, save=False):
        # Run one step of each user-added element
        for name in self.order:
            T = self.elements[name](T, flow)
            if np.isscalar(T):
                tosave = T
            else:
                T, tosave = T
            if save and name in self.to_save:
                self.data[name].append(tosave)
        return T

    def _step(self, T):
        # Run one step of the simulation, starting with the chiller
        T, compressor, refrigerant = self.chiller(T, self.flow * self.dt)
        save = self.chiller.i_cycle >= self.burn_in
        if save:
            self.data['chiller'].append(T)
            self.data['compressor'].append(compressor)
            self.data['refrigerant'].append(refrigerant)
        T = self._step_elements(T, self.flow, save)
        self.i_step += 1
        return T

    def run(self, n_cycles=5, n_steps=None, burn_in=30):
        '''
        Run the simulation.  The parameters of this function determine how long
        the simulation runs.

        Parameters
        ----------
        n_cycles : int
            Stop the simulation after `burn_in + n_cycles` cycles of the chiller.
            Ignored is `n_steps` is set.
        n_steps : int, optional
            Stop the simulation after `n_steps` time steps.
        burn_in : int, optional
            Run the simulation for `burn_in` cycles before recording any data.
            This allows the simulation to enter periodic behavior that is
            independent of the initial conditions.

        Returns
        -------
        A dictionary mapping element names to their temperatures at every time
        step, if the element was added with `save=True`.  Also includes
        temperatures for the exit of the chiller (`'chiller'`), and the
        refrigerant (`'refrigerant'`).  Finally, the output also the state
        (on or off) of the compressor (`'compressor'`).
        '''
        if n_steps is not None:
            burn_in = 0
        self.n_steps = n_steps
        self.n_cycles = n_cycles + self.chiller.i_cycle
        self.burn_in = burn_in
        self.i_step = 0
        self.data = {}
        for name in self.to_save:
            self.data[name] = deque()
        while not self.terminate():
            self.T = self._step(self.T)
        out = {k: np.asarray(v) for k, v in self.data.items()}
        return out


class BufferSim(SerialSim):
    '''
    A subclass of `SerialSim` that implements a buffer tank.  In this scenario,
    the chiller pumps water directly into and out of a tank.  There is a separate
    pump that circulates water through the user-added elements.
    '''

    def __init__(
        self,
        chiller_flow,
        load_flow,
        buffer_vol,
        dt=1,
        T0=None,
        **chiller_kwargs
    ):
        '''
        Initialize the simulation, including the chiller.

        Parameters
        ----------
        chiller_flow : float
            Coolant flow through the chiller in kg / s
        load_flow : float
            Coolant flow through the user-added elements in kg / s
        buffer_vol : float
            The volume of the buffer tank in liters
        dt : float
            Time step in s
        T0 : float, optional
            Initial temperature in K.  Generally this should not be provided by
            the user.  The automatic calculation provides an initial temperature
            that converges reasonably quickly.
        Additional keyword arguments are passed to `Chiller`.

        '''
        self.dt = dt
        self.chiller_flow = chiller_flow
        self.load_flow = load_flow

        self.chiller = Chiller(T0, dt=dt, **chiller_kwargs)
        if T0 is None:
            T0 = self.chiller.T_tank
        self.buffer = BufferTank(T0, buffer_vol, dt)
        self.T_chill = self.T_load = T0

        self.elements = {}
        self.to_save = deque(
            ('chiller', 'compressor', 'refrigerant', 'buffer')
        )
        self.order = deque()

        self.burn_in = 0
        self.i_step = 0
        self.data = {}

    def _step(self, T_chill, T_load):
        # Run one step of the simulation
        T = self.buffer(T_chill, self.chiller_flow, T_load, self.load_flow)
        T_chill, compressor, refrigerant = self.chiller(
            T, self.chiller_flow * self.dt
        )
        save = self.chiller.i_cycle >= self.burn_in
        if save:
            self.data['chiller'].append(T_chill)
            self.data['compressor'].append(compressor)
            self.data['refrigerant'].append(refrigerant)
            self.data['buffer'].append(self.buffer.T)
        T_load = self._step_elements(T, self.load_flow, save)
        self.i_step += 1
        return T_chill, T_load

    def run(self, n_cycles=5, n_steps=None, burn_in=30):
        '''
        Run the simulation.  The parameters of this function determine how long
        the simulation runs.

        Parameters
        ----------
        n_cycles : int
            Stop the simulation after `burn_in + n_cycles` cycles of the chiller.
            Ignored is `n_steps` is set.
        n_steps : int, optional
            Stop the simulation after `n_steps` time steps.
        burn_in : int, optional
            Run the simulation for `burn_in` cycles before recording any data.
            This allows the simulation to enter periodic behavior that is
            independent of the initial conditions.

        Returns
        -------
        A dictionary mapping element names to their temperatures at every time
        step, if the element was added with `save=True`.  Also includes
        temperatures for the exit of the chiller (`'chiller'`), the refrigerant
        (`'refrigerant'`), and the buffer tank `'buffer`'.  Finally, the output
        also the state (on or off) of the compressor (`'compressor'`).

        '''
        if n_steps is not None:
            burn_in = 0
        self.n_steps = n_steps
        self.n_cycles = n_cycles + self.chiller.i_cycle
        self.burn_in = burn_in
        self.i_step = 0
        self.data = {}
        for name in self.to_save:
            self.data[name] = deque()
        while not self.terminate():
            self.T_chill, self.T_load = self._step(self.T_chill, self.T_load)
        out = {k: np.asarray(v) for k, v in self.data.items()}
        return out


##########################################################################
# Elements to add to the simulation


class Chiller:
    '''
    Specialized simulation element implmenting the chiller.  Note that this
    cannot be treated like the other elements: it has additional return values.
    '''

    def __init__(
        self, T0=None, power=18e3, Tc=270, Th=275, C=3.5e5, G=2e3, dt=1
    ):
        '''
        Initialize the chiller.  All parameters are optional, and filled
        with reaonsable defaults for CHORD and CHIME.

        Parameters
        ----------
        T0 : float
            The initial temperature in K.  This should usually be left at its
            automatically calculated default.
        power : float
            The cooling power of the chiller, in W.
        Tc : float
            The low point of the chiller deadband, in K
        Th : float
            The high point of the chiller deadband, in K
        C : float
            The heat capacity of the refrigerant, in J / K
        G : float
            The thermal conductivity between the refrigerant and the water,
            in W / K
        dt : float
            The time step of the simulation in s.
        '''
        self.power = power
        self.Tc = Tc
        self.Th = Th
        self.C = C
        self.G = G
        self.dt = dt

        self.state = 0
        self.i_cycle = 0
        self.V_tank = 50 * 3.79
        self.refrigerant = self.Tc
        if T0 is None:
            T0 = (self.Tc + self.Th) / 2
        self.T_tank = T0

    def _heat_exchange(self, T_in, flow):
        # In low-flow scenarios, the simple constant power exchange
        # can lower the output temp below the regriferant temp.
        # Model this as two thermal masses (one for the refrigerant
        # and one for the water) with a thermal link between them, and
        # constant power removed from the refrigerant by the chiller.
        # This is an analytically solvable model, and the solutions
        # for both temperatures are exponentials with an added slope
        # Under the model where the entire refrigerant is isothermal,
        # this is will be very close to treating the refrigerant as a
        # temperature reservoir.
        # However, we implement the full calculation so that it is
        # easier to treat this more realistically in the future, if
        # needed.

        # Heat capacity of the water that flows through in time dt
        Cw = flow * self.dt * C_coolant
        # Total thermal energy in the refrigerant and water
        Q_total = T_in * Cw + self.refrigerant * self.C
        # Total heat capacity
        C_total = Cw + self.C
        # Thermal time constant of the combined system
        tau = 1 / ((1 / self.C + 1 / Cw) * self.G)

        # First calculate asymptotic temperatures for the thermal masses
        # Final temperature of both masses if no heat is added or removed
        Tf = Q_total / C_total
        # Asymptotic delta T (including heat removed by the chiller)
        delta_Tf = -self.power * self.state / self.C * tau
        # Asymptotic water and refrigerant temperature temperature, not including slope
        Tw_f = Tf - delta_Tf * self.C / C_total
        Trefrig_f = Tf - delta_Tf * Cw / C_total
        # Asymptotic slope
        slope = -self.power * self.state / C_total

        # Now, we can do the real calculation for a time dt
        linear = slope * self.dt
        exponent = np.exp(-self.dt / tau)
        Trefrig = (
            linear + Trefrig_f + (self.refrigerant - Trefrig_f) * exponent
        )
        Tw = linear + Tw_f + (T_in - Tw_f) * exponent
        return Tw, Trefrig

    def __call__(self, T, flow):
        '''
        Run one step of the chiller simulation.

        Parameters
        ----------
        T : float
            The temperature of the water entering the chiller, in K.
        flow : float
            The rate at which the water is entering the chiller, in kg / s

        Returns
        -------
        The output temperature of the water, the state of the compressor
        (1 = on), and the temperature of the refrigerant,
        '''
        T_in = self.T_tank
        self.T_tank = update_tank_temp(
            self.T_tank,
            self.V_tank,
            (T, flow * self.dt),
        )

        T_out, self.refrigerant = self._heat_exchange(T, flow)

        if self.refrigerant <= self.Tc and self.state:
            self.state = 0
            self.i_cycle += 1
        elif self.refrigerant >= self.Th and ~self.state:
            self.state = 1
        return T_out, self.state, self.refrigerant


class MixingTank:
    '''
    Implements an isothermal (well-mixed) tank.
    '''

    def __init__(self, T0, volume, dt):
        '''
        Initialize the tank with its initial temperature (in K), volume (in l),
        and the time step of the simulation.
        '''
        self.T = T0
        self.vol = volume
        self.dt = dt

    def __call__(self, T, flow):
        # Run one step of the tank.  Remove a volume dt * flow, then
        # add the same volume at the input temperature.  Take a
        # volume-weighted average of the input and tank temperature to determine
        # the new tank temperature.  The returned temperature is the initial
        # tank temperature.
        T_out = self.T
        self.T = update_tank_temp(self.T, self.vol, (T, flow * self.dt))
        return T_out


class Pipe:
    '''
    Implement a pipe with no thermal changes.  Temperatures are fixed in the pipe,
    assuming no mixing or thermal conductivity between adjacent water volumes.
    '''

    def __init__(self, T0, volume, flow, dt):
        '''
        Initialize the pipe with a starting temperature (in K), its total volume
        (in l), the flow of the simulation (in l /s) and the time step in s.
        '''
        len_pipe = int(volume / flow / dt)
        len_pipe = max(len_pipe, 1)
        self.pipe = deque([T0] * len_pipe, len_pipe)

    def __call__(self, T, flow):
        # Run one step of the simulation.
        # Implemented as a FILO queue
        Tout = self.pipe[-1]
        self.pipe.appendleft(T)
        return Tout


class BufferTank:
    '''
    Implements a buffer tank with two inputs and two outputs.
    '''

    def __init__(self, T0, volume, dt):
        '''
        Initialize the tank with its initial temperature (in K), volume (in l),
        and the time step of the simulation.
        '''
        self.T = T0
        self.vol = volume
        self.dt = dt

    def __call__(self, T1, flow1, T2, flow2):
        # Run one step of the tank.  Same as the mixing tank, but with two
        # outputs and two inputs.
        T_out = self.T
        self.T = update_tank_temp(
            self.T, self.vol, (T1, flow1 * self.dt), (T2, flow2 * self.dt)
        )
        return T_out


class SplitLoad:
    '''
    Implements a special parallel segment with the heat load split into two
    legs.  One leg has a mixing tank.
    '''

    def __init__(self, T0, load, split_load, flow_fraction, tank_vol, dt):
        '''
        Initialize the load.

        Parameters
        ----------
        T0 : float
            The initial temperature of the water.
        load : float
            The heat load of the leg withOUT a tank (in W).
        split_load : float
            The heat load of the leg WITH a tank (in W).
        flow_fraction : float
            The fraction of the total flow that goes to the leg WITH a tank.
        tank_vol : float
            The volume of the tank, in l.
        dt : float
            The time step of the simulation, in s.
        '''
        self.load = heat_load(load)
        self.split_load = heat_load(split_load)
        self.flow_fraction = flow_fraction
        if tank_vol > 0:
            self.mixing_tank = MixingTank(T0, tank_vol, dt)
        else:
            self.mixing_tank = lambda T, flow: T

    def __call__(self, T, flow):
        # This is effectively a small simulated segment of its own.
        # Run each element (the main load, the tank and the load with the tank).
        # The output temperature is the flow-weighted average of the output from
        # the two loads.
        split_flow = flow * self.flow_fraction
        flow -= split_flow
        T = self.load(T, flow)

        T_split = self.mixing_tank(T, split_flow)
        T_split = self.split_load(T_split, split_flow)
        T_out = np.average((T, T_split), weights=(flow, split_flow))
        return T_out, T_split


def heat_load(power):
    '''
    Create an element that adds `power` (in W) to the water.  This returns a
    callable that the user can add to a simulation.
    '''

    def load_element(T, flow):
        return T + get_water_dT(flow, power)

    return load_element


##########################################################################
# Utility functions


def update_tank_temp(T_tank, vol_tank, *args):
    '''
    Update a tank temperature using appropriately weighted averages.

    Parameters
    ----------
    T_tank : float
        The temperature of the tank
    vol_tank : float
        The volume of the tank
    (T, flow) pairs : (float, float)
        The remaining arguments take the form of pairs of floats which represent
        different flows into the tank.  The first float in each tuple is the
        temperature of the incoming water, and the second is the volume of that
        water that flows into the tank.

    Returns
    -------
    The temperature of the tank after the flows have taken place and mixed.
    '''
    Ts = [T_tank] + [a[0] for a in args]
    flow_vols = [a[1] for a in args]
    total_flow = sum(flow_vols)
    if total_flow > vol_tank:
        raise ValueError(
            'Total flow ({}) in time step is greater than tank volume ({}).'.format(
                total_flow, vol_tank
            )
        )
    if total_flow > vol_tank / 10:
        warnings.warn(
            'A significant volume of the tank is exchanged in one time step.  Tank temperatures may be inaccurate.'
        )
    vols = [vol_tank - sum(flow_vols)] + flow_vols
    return np.average(Ts, weights=vols)


def get_water_dT(flow, power):
    '''
    Calculate the change in temperature (in K) of the water for a flow in kg /s,
    and added power in W.  Negative `power` removes heat from the water.
    '''
    return power / (flow * C_coolant)


def gpm_to_kgs(gpm, density=1):
    '''
    Convert gallons per minute to kg per second
    (assuming a density given in kg / l).
    '''
    return 3.79 / 60 * gpm * density


def make_sim(flow, load, tank_vol=0, pipe_volume=35 * 3.79, **kwargs):
    '''
    Return a simulation with the following elements:
    chiller, mixing tank, pipe, load, pipe.  The elements
    are named such that the temperature entering the
    load is `'supply'`, and the temperature leaving the load
    is `'return'`.

    Parameters
    ----------
    flow : float
        Coolant flow in kg / s
    load : float
        The heat load in W
    tank_vol : float
        The volume of the mixing tank, in l.  If `tank_vol <= 0`, the mixing
        tank is not included in the simulation.
    pipe_volume : float
        The total volume of the two pipes, in l.
    Additional keyword arguments are passed to `SerialSim`.
    '''
    sim = SerialSim(flow, **kwargs)
    pipe_volume /= 2
    if tank_vol > 0:
        sim.add_element(Pipe(sim.T, pipe_volume, flow, sim.dt), 'pipe1')
        sim.add_element(
            MixingTank(sim.T, tank_vol, sim.dt), 'supply', save=True
        )
    else:
        sim.add_element(
            Pipe(sim.T, pipe_volume, flow, sim.dt), 'supply', save=True
        )
    sim.add_element(heat_load(load), 'return', save=True)
    sim.add_element(Pipe(sim.T, pipe_volume, flow, sim.dt), 'pipe2')
    return sim


def make_buffer_sim(load, *sim_args, pipe_volume=35 * 3.79, **kwargs):
    '''
    Return a buffer tank simulation with the following elements:
    chiller, mixing tank, pipe, load, pipe.  The elements
    are named such that the temperature entering the
    load is `'supply'`, and the temperature leaving the load
    is `'return'`.

    Parameters
    ----------
    load : float
        The heat load in W
    chiller_flow : float
        Coolant flow through the chiller in kg / s
    load_flow : float
        Coolant flow through the user-added elements in kg / s
    buffer_vol : float
        The volume of the buffer tank in liters
    pipe_volume : float
        The total volume of the two pipes, in l.
    Additional keyword arguments are passed to `BufferSim`.
    '''
    sim = BufferSim(*sim_args, **kwargs)
    pipe_volume /= 2
    sim.add_element(
        Pipe(sim.T_load, pipe_volume, sim.load_flow, sim.dt),
        'supply',
        save=True,
    )
    sim.add_element(heat_load(load), 'return', save=True)
    sim.add_element(
        Pipe(sim.T_load, pipe_volume, sim.load_flow, sim.dt), 'pipe2'
    )
    return sim


def make_split_sim(
    load,
    split_load,
    flow,
    split_flow,
    tank_vol=0,
    pipe_volume=35 * 3.79,
    **kwargs
):
    '''
    Return a simulation with the following elements:
    chiller, mixing tank, pipe, split_load, pipe.  The elements
    are named such that the temperature entering the
    split load is `'supply'`, and the temperature leaving the split load
    is `'return'`.

    load : float
        The heat load of the leg withOUT a tank (in W).
    split_load : float
        The heat load of the leg WITH a tank (in W).
    flow : float
        The total flow (through both of the split legs) in kg / s.
    flow_fraction : float
        The fraction of the total flow that goes to the leg WITH a tank.
    tank_vol : float
        The volume of the tank, in l.
    pipe_volume : float
        The total volume of the two pipes, in l.
    Additional keyword arguments are passed to `SerialSim`.
    '''
    flow_fraction = split_flow / (split_flow + flow)
    sim = SerialSim(flow, **kwargs)
    pipe_volume /= 2
    sim.add_element(
        Pipe(sim.T, pipe_volume, sim.flow, sim.dt), 'pre_split', save=False
    )
    sim.add_element(
        SplitLoad(sim.T, load, split_load, flow_fraction, tank_vol, sim.dt),
        'supply',
        save=True,
    )
    sim.add_element(Pipe(sim.T, 0, sim.flow, sim.dt), 'return', save=True)
    sim.add_element(Pipe(sim.T, pipe_volume, sim.flow, sim.dt), 'pipe2')
    return sim
