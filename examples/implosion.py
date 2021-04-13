from pyHype.input.implosion import implosion
from pyHype import euler_2D


solver = euler_2D.Euler2DSolver(implosion)
solver.solve()

stats = solver.profile.sort_stats('tottime')
stats.print_stats()
