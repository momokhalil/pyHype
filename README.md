# pyHype
Python solver for the Hyperbolic 2D Euler equations of inviscid fluids on distributed, multi-block grids.

pyHype is in early stages of development! I will be updating it regularly, along with its documentation. The docs will be done using pdoc (https://github.com/mitmproxy/pdoc), and so docstrings will be tailored towards its capabilities. If you see Latex math in docstrings (for example in pyHype/states.py, dont worry, you will see beautiful equations once all the docs are compiled with pdoc!).

pyHype is an educational project in Computational Fluid Dynamics. It brings together over two years of learning during my graduate studies at the University of Toronto. It provides the capability to numerically solve the 2D Euler equations on multi-block grids. This paves the way for the addition of Adaptive Mesh Refinement, which is planned for the future. pyHype uses the finite volume method to solve the Euler equations. First and second order reconstructions schemes will be available, along with a multitude of time marching schemes, approximate riemann solvers (Roe, HLLE, HLLL) and flux limiters. 
