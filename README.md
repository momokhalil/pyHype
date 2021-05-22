![Alt Text](/logo.png)

# pyHype

pyHype is a framework for developing computational fluid dynamics software to solve the HYPErbolic 2D Euler equations on distributed, multi-block grids. It can be used as a solver to generate numerical predictions of 2D inviscid flow fields, or as a platform for developing new CFD techniques and methods. Contributions are welcome! pyHype is in early stages of development! I will be updating it regularly, along with its documentation.

Here is an example of an implosion simulation, performed with a rudimentary prototype version of pyHype on one block. This was run with a 650x650 mesh, Roe approximate riemann solver, van-Albada flux limiter, second-order Green-Gauss reconstruction and RK4 time stepping.

![Alt Text](/examples/implosion.gif)

Current work:
1. Allow geometry that is not alligned with the cartesian axes.
2. Compile gradient and reconstruction calculations with numba
3. Create a fully documented simple example to explain usage
4. Documentation!!

Major future work:
1. Use MPI to distrubute computation to multiple processors
2. Adaptive mesh refinement (maybe with Machine Learning :))
3. Interactive gui for mesh design
4. Advanced interactive plotting
