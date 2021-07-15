![Alt Text](/logo.png){:height="50%" width="50%"}

# [pyHype](https://github.com/momokhalil/pyHype): Computational Fluid Dynamics in Python

pyHype is a Python framework for developing parallelized Computational Fluid Dynamics software to solve the hyperbolic 2D Euler equations on distributed, multi-block structured grids. It can be used as a solver to generate numerical predictions of 2D inviscid flow fields, or as a platform for developing new CFD techniques and methods. Contributions are welcome! pyHype is in early stages of development, I will be updating it regularly, along with its documentation.

The core idea behind pyHype is flexibility and modularity. pyHype offers a plug-n-play approach to CFD software, where every component of the CFD pipeline is modelled as a class with a set interface that allows it to communicate and interact with other components. This enables easy development of new components, since the developer does not have to worry about interfacing with other components. For example, if a developer is interested in developing a new approximate riemann solver technique, they only need to provide the implementation of the `FluxFunction` abstract class, without having to worry about how the rest of the code works in detail.

**NEW**: Geometry not alligned with the cartesian axes is now supported! Example coming soon.

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
from pyHype.solvers import solver

# Solver settings
settings = {'problem_type':             'explosion',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'Conservative',
            'CFL':                      0.4,
            't_final':                  0.07,
            'realplot':                 True,
            'makeplot':                 False,
            'time_it':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       600,
            'ny':                       1200,
            'nghost':                   1,
            'mesh_name':                'chamber',
            'profile':                  False}

# Create solver
exp = solver.Euler2D(fvm='SecondOrderPWL', gradient='GreenGauss', flux_function='Roe',
                     limiter='Venkatakrishnan', integrator='RK2', settings=settings)

# Solve
exp.solve()

```
![alt text](/explosion.gif)

## Current work
1. Compile gradient and reconstruction calculations with numba
2. Integrate PyTecPlot to use for writing solution files and plotting
3. Implement riemann-invariant boundary conditions
4. Create a fully documented simple example to explain usage
5. Documentation!!

## Major future work
1. Use MPI to distrubute computation to multiple processors
2. Adaptive mesh refinement (maybe with Machine Learning :))
3. Interactive gui for mesh design
4. Advanced interactive plotting
