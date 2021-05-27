![Alt Text](/logo.png)

# [pyHype](https://github.com/momokhalil/pyHype): Computational Fluid Dynamics in Python

pyHype is a Python framework for developing computational fluid dynamics software to solve the HYPErbolic 2D Euler equations on distributed, multi-block grids. It can be used as a solver to generate numerical predictions of 2D inviscid flow fields, or as a platform for developing new CFD techniques and methods. Contributions are welcome! pyHype is in early stages of development! I will be updating it regularly, along with its documentation.

Here is an example of an explosion simulation performed on one block. The mesh consisted of 600 x 1200 cells, and the Roe approximate riemann solver, van-Albada flux limiter, second-order Green-Gauss reconstruction, and RK4 time stepping were used.

![Alt Text](/examples/explosion.gif)

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
