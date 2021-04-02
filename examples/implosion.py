from pyHype.input_files.implosion import implosion
from pyHype import euler_2D


solver = euler_2D.Euler2DSolver(implosion)
solver.solve()

stats = solver.profile.sort_stats('cumtime')
stats.print_stats()
