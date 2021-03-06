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

## Explosion on non-cartesian, multiblock mesh simulation
Here is an example of an explosion simulation performed on five blocks. The simulation was performed with the following: 
- 350 x 350 cells per block
- HLLL flux function
- Venkatakrishnan flux limiter
- Piecewise-Linear second order reconstruction
- Green-Gauss gradient method
- Strong-Stability-Preserving (SSP)-RK3 time stepping with CFL=0.6
- Reflection boundary conditions

The example in given in the file [examples/explosion_skewed.py](https://github.com/momokhalil/pyHype/blob/main/examples/explosion_skewed.py). The file is as follows:

```python
from pyHype.solvers import Euler2D

# Solver settings
settings = {'problem_type':             'explosion_3',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'primitive',
            'upwind_mode':              'primitive',
            'write_solution':           True,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'explosion3',
            'write_every_n_timesteps':  15,
            'CFL':                      0.6,
            't_final':                  0.04,
            'realplot':                 False,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       350,
            'ny':                       350,
            'nghost':                   1,
            'mesh_name':                'chamber_skewed_2'
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='HLLL',
              limiter='Venkatakrishnan',
              integrator='RK3SSP',
              settings=settings)

# Solve
exp.solve()
```
![alt text](/explosion3.gif)


## Supersonic Simulation
Here is an example of supersonic ramjet simulation performed on 9 blocks. The simulation was performed with the following: 
- Mach 2 flow
- 300 x 300 cell blocks
- Roe approximate riemann solver
- Venkatakrishnan flux limiter
- Piecewise-Linear second order reconstruction
- Green-Gauss gradient method
- RK2 time stepping with CFL=0.4
- Reflection boundary conditions for top and bottom walls
- Dirichlet input and output boundary conditions

The example in given in the file [examples/ramjet/ramjet.py](https://github.com/momokhalil/pyHype/blob/main/examples/ramjet/ramjet.py). The file is as follows:

```python
from pyHype.solvers import Euler2D

# Solver settings
settings = {'problem_type':             'supersonic_flood',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'write_solution':           True,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'ramjet',
            'write_every_n_timesteps':  40,
            'CFL':                      0.4,
            't_final':                  10.0,
            'realplot':                 False,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       300,
            'ny':                       300,
            'nghost':                   1,
            'mesh_name':                'ramjet',
            'BC_inlet_west_rho':        1.0,
            'BC_inlet_west_u':          2.0,
            'BC_inlet_west_v':          0.0,
            'BC_inlet_west_p':          1 / 1.4,
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='Roe',
              limiter='Venkatakrishnan',
              integrator='RK2',
              settings=settings)
# Solve
exp.solve()
```
![alt text](/ramjet.png)

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
