This is a very small package to simulate various configurations of the CHORD chilled water system in the receiver huts.  The functionailty is broken into three files:
* `chiller_sims.py` contains the simulation code and framework for building different configurations.
* `util.py` contains functions useful for analyzing the simulation results.
* `sim_variations.py` provides functions that run a set of simulations, varying one parameter at a time.

Installation
============
Place this directory in your PYTHONPATH.

Basic Usage
===========
```
# Run one simulation and plot the resulting TOD
>>> from chiller_sims import chiller_sims, util
>>> sim = chiller_sims.make_sim(chiller_sims.gpm_to_kgs(18), 16e3)
>>> data = sim.run()
>>> util.reduce_sim(data['supply'], data['compressor'])
(np.float64(1487.0),
 np.float64(2.3441037789551955),
 np.float64(277.8374284910213))
>>> util.plot_sim(data)
```

