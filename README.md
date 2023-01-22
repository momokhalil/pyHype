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

The example in given in the file [examples/explosion/explosion.py](https://github.com/momokhalil/pyHype/blob/main/examples/explosion/explosion.py). The file is as follows:

```python
from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.states import ConservativeState
from pyhype.solvers.base import ProblemInput

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

block1 = {
    "nBLK": 1,
    "NW": [0, 20],
    "NE": [10, 20],
    "SW": [0, 0],
    "SE": [10, 0],
    "NeighborE": None,
    "NeighborW": None,
    "NeighborN": None,
    "NeighborS": None,
    "NeighborNE": None,
    "NeighborNW": None,
    "NeighborSE": None,
    "NeighborSW": None,
    "BCTypeE": "Reflection",
    "BCTypeW": "Reflection",
    "BCTypeN": "Reflection",
    "BCTypeS": "Reflection",
    "BCTypeNE": None,
    "BCTypeNW": None,
    "BCTypeSE": None,
    "BCTypeSW": None,
}
mesh = {1: block1}

inputs = ProblemInput(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="Roe",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK4",
    mesh=mesh,
    problem_type="explosion",
    interface_interpolation="arithmetic_average",
    reconstruction_type=ConservativeState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="nozzle",
    write_every_n_timesteps=40,
    plot_every=2,
    CFL=0.8,
    t_final=0.07,
    realplot=True,
    profile=True,
    fluid=air,
    nx=50,
    ny=50,
    nghost=1,
    use_JIT=True,
)

# Create solver
exp = Euler2D(inputs=inputs)

# Solve
exp.solve()

```
![alt text](/examples/explosion/explosion.gif)

## Shockbox Simulation
Here is an example of an shockbox simulation performed on one block. The simulation was performed with the following: 
- 500 x 500 cartesian grid
- Roe approximate riemann solver
- Venkatakrishnan flux limiter
- Piecewise-Linear second order reconstruction
- Green-Gauss gradient method
- RK2 time stepping with CFL=0.4
- Reflection boundary conditions

The example in given in the file [examples/shockbox/shockbox.py](https://github.com/momokhalil/pyHype/blob/main/examples/shockbox/shockbox.py). The file is as follows:

```python
from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.states import ConservativeState
from pyhype.solvers.base import ProblemInput

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

block1 = {
    "nBLK": 1,
    "NW": [0, 10],
    "NE": [10, 10],
    "SW": [0, 0],
    "SE": [10, 0],
    "NeighborE": None,
    "NeighborW": None,
    "NeighborN": None,
    "NeighborS": None,
    "NeighborNE": None,
    "NeighborNW": None,
    "NeighborSE": None,
    "NeighborSW": None,
    "BCTypeE": "Reflection",
    "BCTypeW": "Reflection",
    "BCTypeN": "Reflection",
    "BCTypeS": "Reflection",
    "BCTypeNE": None,
    "BCTypeNW": None,
    "BCTypeSE": None,
    "BCTypeSW": None,
}

mesh = {1: block1}

# Solver settings
inputs = ProblemInput(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="Roe",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK2",
    mesh=mesh,
    problem_type="shockbox",
    interface_interpolation="arithmetic_average",
    reconstruction_type=ConservativeState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="shockbox",
    write_every_n_timesteps=30,
    plot_every=10,
    CFL=0.4,
    t_final=2.0,
    realplot=True,
    profile=False,
    fluid=air,
    nx=50,
    ny=50,
    nghost=1,
    use_JIT=True,
)

# Create solver
exp = Euler2D(inputs=inputs)

exp.solve()
```
![alt text](/examples/shockbox/rho.gif)

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
from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.states import PrimitiveState
from pyhype.solvers.base import ProblemInput
from pyhype.mesh.base import QuadMeshGenerator

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

BCS = ["OutletDirichlet", "Slipwall", "Slipwall", "Slipwall"]

_mesh = QuadMeshGenerator(
    nx_blk=4,
    ny_blk=1,
    BCE=["OutletDirichlet"],
    BCW=["OutletDirichlet"],
    BCN=["OutletDirichlet"],
    BCS=BCS,
    top_x=_x,
    bot_x=_x,
    top_y=_top_y,
    bot_y=_bot_y,
    left_x=_left_x,
    right_x=_right_x,
    left_y=_left_y,
    right_y=_right_y,
)

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

# Solver settings
inputs = ProblemInput(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="HLLL",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK2",
    problem_type="mach_reflection",
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="machref",
    write_every_n_timesteps=20,
    plot_every=1,
    CFL=0.4,
    t_final=0.25,
    realplot=True,
    profile=False,
    fluid=air,
    nx=50,
    ny=50,
    nghost=1,
    use_JIT=True,
    mesh=_mesh,
)

# Create solver
exp = Euler2D(inputs=inputs)

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
import numpy as np
from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.solvers.base import ProblemInput
from pyhype.mesh.base import QuadMeshGenerator
from pyhype.states import PrimitiveState
from pyhype.boundary_conditions.base import PrimitiveDirichletBC

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

inlet_rho = 1.0
inlet_u = 0.25
inlet_v = 0.0
inlet_p = 1.0 / 1.4

inlet_state = PrimitiveState(
    fluid=air,
    array=np.array(
        [
            inlet_rho,
            inlet_u,
            inlet_v,
            inlet_p,
        ]
    ).reshape((1, 1, 4)),
)
subsonic_inlet_bc = PrimitiveDirichletBC(primitive_state=inlet_state)

BCE = [
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
]
BCW = ["Slipwall", "Slipwall", subsonic_inlet_bc, "Slipwall", "Slipwall"]
BCN = ["OutletDirichlet"]
BCS = ["OutletDirichlet"]

_mesh = QuadMeshGenerator(
    nx_blk=1,
    ny_blk=5,
    BCE=BCE,
    BCW=BCW,
    BCN=BCN,
    BCS=BCS,
    NE=(1, 0.5),
    SW=(0, 0),
    NW=(0, 0.5),
    SE=(1, 0),
)

# Solver settings
inputs = ProblemInput(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="HLLL",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK2",
    mesh=_mesh,
    problem_type="subsonic_rest",
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="kvi",
    write_every_n_timesteps=20,
    plot_every=10,
    CFL=0.4,
    t_final=25.0,
    realplot=True,
    profile=False,
    fluid=air,
    nx=100,
    ny=10,
    nghost=1,
    use_JIT=True,
)

# Create solver
exp = Euler2D(inputs=inputs)

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
