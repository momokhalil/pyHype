from examples.implosion.config import config
from examples.implosion.mesh import mesh
from pyhype.solvers import Euler2D

if __name__ == "__main__":
    implosion = Euler2D(config=config, mesh_config=mesh)
    implosion.solve()
