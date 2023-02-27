from mesh import step_ten_block
from pyhype.solvers import Euler2D
from examples.supersonic_step.config import config

mesh = step_ten_block()

# Create solver
supersonic_step = Euler2D(config=config, mesh_config=mesh)

# Solve
supersonic_step.solve()
