import inputs
from pyHype import euler_2D

solver = euler_2D.Euler2DExplicitSolver(inputs.E4())
solver.solve()
