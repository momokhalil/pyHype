from examples.jet.mesh import mesh
from examples.jet.config import config
from pyhype.solvers import Euler2D

if __name__ == "__main__":
    jet = Euler2D(config=config, mesh_config=mesh)
    jet.solve()
