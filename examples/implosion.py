from pyHype.input.implosion import implosion
from pyHype.input.chamber import chamber
from pyHype.solvers import solver
import os

os.environ["NUMBA_DISABLE_JIT"] = str(0)

solver = solver.Euler2DSolver(implosion)
solver.solve()
