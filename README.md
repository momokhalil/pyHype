![Alt Text](/logo.png)

# [pyHype](https://github.com/momokhalil/pyHype): Computational Fluid Dynamics in Python

pyHype is a Python framework for developing parallelized Computational Fluid Dynamics software to solve the hyperbolic 2D Euler equations on distributed, multi-block structured grids. It can be used as a solver to generate numerical predictions of 2D inviscid flow fields, or as a platform for developing new CFD techniques and methods. Contributions are welcome! pyHype is in early stages of development, I will be updating it regularly, along with its documentation.

The core idea behind pyHype is flexibility and modularity. pyHype offers a plug-n-play approach to CFD software, where every component of the CFD pipeline is modelled as a class with a set interface that allows it to communicate and interact with other components. This enables easy development of new components, since the developer does not have to worry about interfacing with other components. For example, if a developer is interested in developing a new approximate riemann solver technique, they only need to provide the implementation of the `FluxFunction` abstract class, without having to worry about how the rest of the code works in detail. 

## Explosion Simulation
Here is an example of an explosion simulation performed on one block. The simulation was performed with the following: 
- 600 x 1200 cartesian grid
- Roe approximate riemann solver
- Van-Albada flux limiter
- Second-order Green-Gauss reconstruction
- RK4 time stepping with CFL=0.8
- Reflection boundary conditions

The example in given in the file [examples/explosion.py](https://github.com/momokhalil/pyHype/blob/main/examples/explosion.py). The file is as follows:

```python
from pyHype.solvers import solver

explosion = {'problem_type':          'explosion',
             'IC_type':               'from_IC',
             'flux_function':         'Roe',
             'reconstruction_type':   'Primitive',
             'realplot':              0,
             'makeplot':              0,
             'time_it':               0,
             't_final':               0.06,
             'time_integrator':       'RK4',
             'CFL':                   0.8,
             'finite_volume_method':  'SecondOrderGreenGauss',
             'flux_limiter':          'van_albada',
             'gamma':                 1.4,
             'rho_inf':               1.0,
             'a_inf':                 343.0,
             'R':                     287.0,
             'nx':                    600,
             'ny':                    1200,
             'mesh_name':             'chamber',
             'profile':               False}

exp = solver.Euler2DSolver(explosion)
exp.solve()
```

![Alt Text](/explosion.gif)

## Current work
1. Allow geometry that is not alligned with the cartesian axes.
2. Compile gradient and reconstruction calculations with numba
3. Implement riemann-invariant boundary conditions
4. Create a fully documented simple example to explain usage
5. Documentation!!

## Major future work
1. Use MPI to distrubute computation to multiple processors
2. Adaptive mesh refinement (maybe with Machine Learning :))
3. Interactive gui for mesh design
4. Advanced interactive plotting
