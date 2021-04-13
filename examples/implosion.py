from pyHype.input.implosion import implosion
from pyHype import solver


solver = solver.Euler2DSolver(implosion)
solver.solve()

stats = solver.profile.sort_stats('tottime')
stats.print_stats()
