from pyHype.input_files import E4
from pyHype import euler_2D

problem_inputs = E4.E4

solver = euler_2D.Euler2DSolver(problem_inputs)
solver.solve()

stats = solver.profile.sort_stats('cumtime')
stats.print_stats()
