![Alt Text](/logo.png)

# [pyHype](https://github.com/momokhalil/pyHype): Computational Fluid Dynamics in Python

pyHype is a Python framework for developing parallelized Computational Fluid Dynamics software to solve the hyperbolic 2D Euler equations on distributed, multi-block structured grids. I started writing pyHype in python as a challenge to achieve high performance in scientific applications traditionally written in low level languages like C/C++ and FORTRAN. It can be used as a solver to generate numerical predictions of 2D inviscid flow fields, or as a platform for developing new CFD techniques and methods. Contributions are welcome! pyHype is in early stages of development, I will be updating it regularly, along with its documentation.

The core idea behind pyHype is flexibility and modularity. pyHype offers a plug-n-play approach to CFD software, where every component of the CFD pipeline is modelled as a class with a set interface that allows it to communicate and interact with other components. This enables easy development of new components, since the developer does not have to worry about interfacing with other components. For example, if a developer is interested in developing a new approximate riemann solver technique, they only need to provide the implementation of the `FluxFunction` abstract class, without having to worry about how the rest of the code works in detail.

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

block1 = {'nBLK': 1,
          'NW': [0, 20], 'NE': [10, 20],
          'SW': [0, 0], 'SE': [10, 0],
          'NeighborE': None,
          'NeighborW': None,
          'NeighborN': None,
          'NeighborS': None,
          'BCTypeE': 'Reflection',
          'BCTypeW': 'Reflection',
          'BCTypeN': 'Reflection',
          'BCTypeS': 'Reflection'}

mesh = {1: block1}


# Solver settings
settings = {'problem_type':             'explosion',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'nozzle',
            'write_every_n_timesteps':  40,
            'plot_every':               20,
            'CFL':                      0.8,
            't_final':                  0.07,
            'realplot':                 False,
            'profile':                  True,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       600,
            'ny':                       600,
            'nghost':                   1,
            }

# Create solver
exp = Euler2D(fvm_type='MUSCL',
              fvm_spatial_order=2,
              fvm_num_quadrature_points=1,
              fvm_gradient_type='GreenGauss',
              fvm_flux_function='Roe',
              fvm_slope_limiter='Venkatakrishnan',
              time_integrator='RK4',
              settings=settings,
              mesh_inputs=mesh)

# Solve
exp.solve()
```
![alt text](/examples/explosion/explosion.gif)

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
import numpy as np
from pyHype.solvers import Euler2D
from pyHype.mesh.base import QuadMeshGenerator

k = 1
a = 2 / np.sqrt(3)
d = np.tan(30 * np.pi / 180)

_left_x = [0, 0]
_left_y = [0, a]
_right_x = [4 * k, 4 * k]
_right_y = [3 * d, a + 3 * d]
_x = [0, k, 2 * k, 3 * k, 4 * k]
_top_y = [a, a, a + d, a + 2 * d, a + 3 * d]
_bot_y = [0, 0, d, 2 * d, 3 * d]

BCS = ['OutletDirichlet', 'Slipwall', 'Slipwall', 'Slipwall']

_mesh = QuadMeshGenerator(nx_blk=4, ny_blk=1,
                          BCE='OutletDirichlet', BCW='OutletDirichlet', BCN='OutletDirichlet', BCS=BCS,
                          top_x=_x, bot_x=_x, top_y=_top_y, bot_y=_bot_y,
                          left_x=_left_x, right_x=_right_x, left_y=_left_y, right_y=_right_y)

# Solver settings
settings = {'problem_type':             'mach_reflection',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'primitive',
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
            'nx':                       500,
            'ny':                       500,
            'nghost':                   1,
            'BC_inlet_west_rho':        8.0,
            'BC_inlet_west_u':          8.25,
            'BC_inlet_west_v':          0.0,
            'BC_inlet_west_p':          116.5,
            }

# Create solver
exp = Euler2D(fvm_type='MUSCL',
              fvm_spatial_order=2,
              fvm_num_quadrature_points=1,
              fvm_gradient_type='GreenGauss',
              fvm_flux_function='HLLL',
              fvm_slope_limiter='Venkatakrishnan',
              time_integrator='RK2',
              settings=settings,
              mesh_inputs=_mesh)

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
from pyHype.mesh.base import QuadMeshGenerator

BCE = ['OutletDirichlet', 'OutletDirichlet', 'OutletDirichlet', 'OutletDirichlet', 'OutletDirichlet']
BCW = ['Slipwall', 'Slipwall', 'InletDirichlet', 'Slipwall', 'Slipwall']
BCN = ['OutletDirichlet']
BCS = ['OutletDirichlet']

_mesh = QuadMeshGenerator(nx_blk=1, ny_blk=5,
                          BCE=BCE, BCW=BCW, BCN=BCN, BCS=BCS,
                          NE=(1, 0.5), SW=(0, 0), NW=(0, 0.5), SE=(1, 0))

# Solver settings
settings = {'problem_type':             'subsonic_rest',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'primitive',
            'upwind_mode':              'conservative',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'kvi',
            'write_every_n_timesteps':  20,
            'plot_every':               10,
            'CFL':                      0.4,
            't_final':                  25.0,
            'realplot':                 True,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       1000,
            'ny':                       100,
            'nghost':                   1,
            'BC_inlet_west_rho':        1.0,
            'BC_inlet_west_u':          0.25,
            'BC_inlet_west_v':          0.0,
            'BC_inlet_west_p':          2.0 / 1.4,
            }

# Create solver
exp = Euler2D(fvm_type='MUSCL',
              fvm_spatial_order=2,
              fvm_num_quadrature_points=1,
              fvm_gradient_type='GreenGauss',
              fvm_flux_function='HLLL',
              fvm_slope_limiter='Venkatakrishnan',
              time_integrator='RK2',
              settings=settings,
              mesh_inputs=_mesh)

# Solve
exp.solve()
```
Mach Number:
![alt text](/examples/jet/Ma.gif)

Density:
![alt text](/examples/jet/rho.gif)

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
