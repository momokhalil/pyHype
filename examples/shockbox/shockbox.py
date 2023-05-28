from pyhype.solvers import Euler2D
from examples.shockbox.config import config
from examples.shockbox.mesh import mesh

if __name__ == "__main__":
    shockbox = Euler2D(config=config, mesh_config=mesh)
    shockbox.solve()
