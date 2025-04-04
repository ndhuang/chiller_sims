import numpy as np
from . import chiller_sims as cs, util

# This file contains a set of functions that vary some parameter in the
# simulation, and return the reduced metrics for each simulation
# (see util.reduce_sim).


def vary_load(
    load=(np.arange(5) + 1) * 3e3, flow=cs.gpm_to_kgs(20), **kwargs
):
    # Vary load at fixed flow
    # Returns the load in W and the reduced metrics
    out = []
    for l in load:
        s = cs.make_sim(flow, l, **kwargs)
        d = s.run()
        out.append(util.reduce_sim(d['supply'], d['compressor']))
    return load, np.asarray(out).T


def vary_flow(
    load=9e3, flow=cs.gpm_to_kgs((np.arange(5) + 0.5) * 10), **kwargs
):
    # Vary flow at fixed load
    # Returns the flow kg/s and the reduced metrics
    out = []
    for f in flow:
        s = cs.make_sim(f, load, **kwargs)
        d = s.run()
        out.append(util.reduce_sim(d['supply'], d['compressor']))
    return flow, np.asarray(out).T


def vary_mixing_tank(
    load=9e3,
    flow=cs.gpm_to_kgs(20),
    tank_vol=np.asarray([0, 5, 10, 25, 50, 100, 200]) * 3.79,
    **kwargs
):
    # Vary the volume of the mixing tank at fixed flow and load
    # Returns the mixing tank volume in l and the reduced metrics
    out = []
    for v in tank_vol:
        s = cs.make_sim(flow, load, tank_vol=v)
        d = s.run()
        out.append(util.reduce_sim(d['supply'], d['compressor']))
    return tank_vol, np.asarray(out).T


def buffer_vs_mix(
    load=9e3,
    flow=cs.gpm_to_kgs(20),
    tank_vol=np.linspace(0, 200, num=5) * 3.79,
    chiller_flow=cs.gpm_to_kgs(50),
):
    # Compare the buffer and mixing tank configurations at fixed load and flow
    # Returns the tank volume in l, the reduced metrics for the buffer tank
    # configuration, and the reduced metrics for the mixing tank configuration
    buff = []
    mix = []
    for vol in tank_vol:
        s = cs.make_sim(flow, load, tank_vol=vol)
        d = s.run()
        mix.append(util.reduce_sim(d['supply'], d['compressor']))
        if vol > 0:
            s = cs.make_buffer_sim(load, chiller_flow, flow, vol)
            d = s.run()
            buff.append(util.reduce_sim(d['supply'], d['compressor']))
        else:
            buff.append(mix[-1])
    return tank_vol, np.asarray(buff).T, np.asarray(mix).T


def split_mix(
    load=13e3,
    flow=cs.gpm_to_kgs(28),
    split_load=1e3,
    split_flow=cs.gpm_to_kgs(2),
    tank_vol=np.linspace(0, 10, num=5) * 3.79,
):
    # Vary the volume of the mixing tank in the split configuration
    # at fixed load and flow.
    # Returns the flow kg/s and the reduced metrics
    out = []
    for vol in tank_vol:
        if vol == 0:
            s = cs.make_sim(flow, load)
            d = s.run()
            out.append(util.reduce_sim(d['supply'], d['compressor']))
        else:
            s = cs.make_split_sim(
                load, split_load, flow, split_flow, tank_vol=vol
            )
            d = s.run()
            out.append(util.reduce_sim(d['supply'], d['compressor']))
    return tank_vol, np.asarray(out).T
