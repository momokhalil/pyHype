![Alt Text](/logo.png)

# [pyHype](https://github.com/momokhalil/pyHype): Computational Fluid Dynamics in Python

pyHype is a Python framework for developing parallelized Computational Fluid Dynamics software to solve the hyperbolic 2D Euler equations on distributed, multi-block structured grids. It can be used as a solver to generate numerical predictions of 2D inviscid flow fields, or as a platform for developing new CFD techniques and methods. Contributions are welcome! pyHype is in early stages of development, I will be updating it regularly, along with its documentation.

The core idea behind pyHype is flexibility and modularity. pyHype offers a plug-n-play approach to CFD software, where every component of the CFD pipeline is modelled as a class with a set interface that allows it to communicate and interact with other components. This enables easy development of new components, since the developer does not have to worry about interfacing with other components. For example, if a developer is interested in developing a new approximate riemann solver technique, they only need to provide the implementation of the `FluxFunction` abstract class, without having to worry about how the rest of the code works in detail.

**NEW**: Geometry not alligned with the cartesian axes is now supported!\
**NEW**: 60% efficiency improvement!\
**COMING UP**: Examples of simulations on various airfoil geometries, and a presentation of the newly added mesh optimization techniques.\
**COMING UP**: Examples of simulations on multi-block meshes.

## Explosion Simulation
Here is an example of an explosion simulation performed on one block. The simulation was performed with the following: 
- 600 x 1200 cartesian grid
- Roe approximate riemann solver
- Venkatakrishnan flux limiter
- Piecewise-Linear second order reconstruction
- Green-Gauss gradient method
- RK4 time stepping with CFL=0.8
- Reflection boundary conditions

The example in given in the file [examples/explosion.py](https://github.com/momokhalil/pyHype/blob/main/examples/explosion.py). The file is as follows:

```python
from pyHype.solvers import Euler2D

# Solver settings
settings = {'problem_type':             'explosion',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'nozzle',
            'write_every_n_timesteps':  40,
            'CFL':                      0.8,
            't_final':                  0.07,
            'realplot':                 False,
            'profile':                  True,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       600,
            'ny':                       1200,
            'nghost':                   1,
            'mesh_name':                'chamber'
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='Roe',
              limiter='Venkatakrishnan',
              integrator='RK4',
              settings=settings)

# Solve
exp.solve()

```
![alt text](/explosion.gif)

## Double Mach Reflection (DMR)
Here is an example of a Mach 10 DMR simulation performed on five blocks. The simulation was performed with the following: 
- 500 x 500 cells per block
- HLLL flux function
- Venkatakrishnan flux limiter
- Piecewise-Linear second order reconstruction
- Green-Gauss gradient method
- Strong-Stability-Preserving (SSP)-RK2 time stepping with CFL=0.4

The example in given in the file [examples/dmr/dmr.py](https://github.com/momokhalil/pyHype/blob/main/examples/dmr/dmr.py). The file is as follows:

```python
from pyHype.solvers import Euler2D

# Solver settings
settings = {'problem_type':             'mach_reflection',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'conservative',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'machref',
            'write_every_n_timesteps':  20,
            'plot_every':               10,
            'CFL':                      0.4,
            't_final':                  0.25,
            'realplot':                 True,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       50,
            'ny':                       50,
            'nghost':                   1,
            'mesh_name':                'wedge_35_four_block',
            'BC_inlet_west_rho':        8.0,
            'BC_inlet_west_u':          8.25,
            'BC_inlet_west_v':          0.0,
            'BC_inlet_west_p':          116.5,
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='HLLL',
              limiter='Venkatakrishnan',
              integrator='RK2',
              settings=settings)

# Solve
exp.solve()
```
![alt text](/examples/dmr/dmr.png)


## High Speed Jet
Here is an example of high-speed jet simulation performed on 5 blocks. The simulation was performed with the following: 
- Mach 2 flow
- 100 x 1000 cell blocks
- HLLL flux function
- Venkatakrishnan flux limiter
- Piecewise-Linear second order reconstruction
- Green-Gauss gradient method
- RK2 time stepping with CFL=0.4

The example in given in the file [examples/jet/jet.py](https://github.com/momokhalil/pyHype/blob/main/examples/jet/jet.py). The file is as follows:

```python
from pyHype.solvers import Euler2D

# Solver settings
settings = {'problem_type':             'subsonic_rest',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'primitive',
            'upwind_mode':              'conservative',
            'write_solution':           True,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'kvi',
            'write_every_n_timesteps':  20,
            'plot_every':               10,
            'CFL':                      0.4,
            't_final':                  25.0,
            'realplot':                 False,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       1000,
            'ny':                       100,
            'nghost':                   1,
            'mesh_name':                'jet',
            'BC_inlet_west_rho':        1.0,
            'BC_inlet_west_u':          0.25,
            'BC_inlet_west_v':          0.0,
            'BC_inlet_west_p':          2.0 / 1.4,
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='HLLL',
              limiter='Venkatakrishnan',
              integrator='RK2',
              settings=settings)

# Solve
exp.solve()
```
Mach Number:
![alt text](/examples/jet/Ma.png)

Density:
![alt text](/examples/jet/rho.png)

## Current work
1. Integrate airfoil meshing and mesh optimization using elliptic PDEs
2. Compile gradient and reconstruction calculations with numba
3. Integrate PyTecPlot to use for writing solution files and plotting
4. Implement riemann-invariant-based boundary conditions
5. Implement subsonic and supersonic inlet and outlet boundary conditions
6. Implement connectivity algorithms for calculating block connectivity and neighbor-finding
7. Create a fully documented simple example to explain usage
8. Documentation!!

## Major future work
1. Use MPI to distrubute computation to multiple processors
2. Adaptive mesh refinement (maybe with Machine Learning :))
3. Interactive gui for mesh design
4. Advanced interactive plotting
