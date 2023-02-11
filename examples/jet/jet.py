from examples.jet.mesh import mesh
from examples.jet.config import config
from pyhype.solvers import Euler2D

# Create solver
jet = Euler2D(config=config, mesh_config=mesh)

# Solve
jet.solve()
